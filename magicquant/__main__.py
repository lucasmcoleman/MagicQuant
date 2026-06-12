"""
MagicQuant CLI - Main Entry Point

Usage:
    magicquant analyze <model.gguf>        Analyze model structure and tensor groups
    magicquant probe <model.gguf>          Run sensitivity probes
    magicquant search <model.gguf>         Run evolutionary search
    magicquant hybrid <config.yaml>        Generate hybrid GGUF from YAML config
    magicquant generate <model.gguf>       Generate hybrid GGUFs from search results
"""

import argparse
import sys
import json
from pathlib import Path

from magicquant.logging import configure_logging, get_logger


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze model structure and tensor groups."""
    from magicquant.gguf.reader import GGUFReader
    from magicquant.gguf.tensor_groups import TensorGroupClassifier

    log = get_logger("analyze")

    log.info("Analyzing model", model=args.model)

    reader = GGUFReader(args.model)
    reader.open()

    arch = reader.get_model_architecture()
    params = reader.get_parameter_count()
    size_gb = reader.get_file_size_gb()
    bpw = reader.get_bits_per_weight()
    tensor_names = reader.get_tensor_names()

    print(f"Architecture:    {arch}")
    print(f"Parameters:      {params:,}")
    print(f"File Size:       {size_gb:.2f} GB")
    print(f"Bits/Weight:     {bpw:.2f}")
    print(f"Total Tensors:   {len(tensor_names)}")
    print()

    classifier = TensorGroupClassifier()
    grouped = classifier.classify_tensors(tensor_names)

    group_labels = {
        'E': 'Embeddings',
        'H': 'LM Head',
        'Q': 'Attention Query',
        'K': 'Attention Key/Value',
        'O': 'Attention Output',
        'U': 'FFN Up/Gate',
        'D': 'FFN Down',
        'X': 'MoE Experts',
        'R': 'MoE Router',
        'UNKNOWN': 'Unclassified',
    }

    print("Tensor Group Distribution:")
    for group, tensors in grouped.items():
        if tensors:
            label = group_labels.get(group, group)
            print(f"  [{group}] {label:<22} {len(tensors):>4} tensors")

    is_moe = len(grouped.get('X', [])) > 0 or len(grouped.get('R', [])) > 0
    print()
    print(f"Architecture type: {'Mixture-of-Experts (MoE)' if is_moe else 'Dense'}")

    reader.close()


def cmd_probe(args: argparse.Namespace) -> None:
    """Run sensitivity probes on model."""
    from magicquant.evolution.probing import SensitivityProber, SensitivityAnalysis

    log = get_logger("probe")

    log.info("Running sensitivity probes", model=args.model)

    baseline_ppl = args.baseline_ppl if hasattr(args, 'baseline_ppl') and args.baseline_ppl else 5.0

    # Try to initialise llama.cpp for real probing
    llama_tools = None
    llamacpp_path = getattr(args, "llamacpp_path", None)
    try:
        from magicquant.utils.llamacpp import LlamaCppTools
        llama_tools = LlamaCppTools(llamacpp_path)
    except Exception:
        log.info("llama.cpp not found, using heuristic sensitivity estimates")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prober = SensitivityProber(
        base_model_path=args.model,
        baseline_perplexity=baseline_ppl,
        perplexity_calculator=llama_tools,
        output_dir=str(output_dir / "_probes"),
    )

    groups = ['E', 'H', 'Q', 'K', 'O', 'U', 'D']
    sensitivity = prober.probe_all_groups(
        groups=groups,
        aggressive_scheme="Q4_K_M",
        verbose=True,
    )

    print()
    weights = prober.get_normalized_weights()
    print("Normalized Sensitivity Weights (sum = 1.0):")
    for group, w in weights.items():
        bar = "#" * int(w * 40)
        print(f"  {group}: {w:.4f}  {bar}")

    high_sens = SensitivityAnalysis.recommend_protected_groups(sensitivity, top_n=3)
    print()
    print("Recommended protected groups (keep high precision):")
    for group, score in high_sens:
        print(f"  {group}: sensitivity={score:.4f}")

    # Save results
    output_file = output_dir / "sensitivity.json"
    prober.save_results(str(output_file))

    print(f"\nSensitivity data saved to: {output_file}")


def _settings_from_args(args: argparse.Namespace):
    """Build MagicQuantSettings, honoring env/.env, then override with any
    explicitly-provided CLI values (argparse default None == not provided).

    This unifies the previously-divergent argparse defaults (50/100) with the
    pydantic config defaults (30/80) into a single source of truth: config.py.
    """
    from magicquant.config import MagicQuantSettings

    # Map argparse names -> settings fields. The model path is required.
    overrides = {"source_model_path": args.model}

    def _maybe(field, attr):
        val = getattr(args, attr, None)
        if val is not None:
            overrides[field] = val

    _maybe("output_dir", "output_dir")
    _maybe("llamacpp_path", "llamacpp_path")
    _maybe("adapter_path", "adapter")
    _maybe("target_base_quant", "target_quant")
    _maybe("search_generations", "generations")
    _maybe("population_size", "population")
    _maybe("measurement_rounds", "rounds")
    _maybe("candidates_per_round", "candidates")
    _maybe("patience", "patience")

    # MagicQuantSettings() reads env/.env first; explicit kwargs (CLI) win.
    return MagicQuantSettings(**overrides)


def cmd_search(args: argparse.Namespace) -> None:
    """Run evolutionary search to find optimal configurations."""
    from magicquant.orchestrator import MagicQuantOrchestrator

    settings = _settings_from_args(args)

    orchestrator = MagicQuantOrchestrator(
        source_model_path=settings.source_model_path,
        output_dir=settings.output_dir,
        llamacpp_path=settings.llamacpp_path,
        adapter_path=settings.adapter_path,
    )

    rounds = settings.measurement_rounds

    if rounds > 0:
        # Full measured search: Predict -> Build -> Measure -> Learn
        all_configs, tiered = orchestrator.run_measured_search(
            target_base_quant=settings.target_base_quant,
            search_generations=settings.search_generations,
            population_size=settings.population_size,
            measurement_rounds=rounds,
            candidates_per_round=settings.candidates_per_round,
            verbose=settings.verbose,
            patience=settings.patience,
        )
    else:
        # Prediction-only search (fast, no llama.cpp required)
        all_configs, tiered = orchestrator.run_full_search(
            target_base_quant=settings.target_base_quant,
            max_generations=settings.search_generations,
            population_size=settings.population_size,
            verbose=settings.verbose,
            patience=settings.patience,
        )

    results_path = Path(settings.output_dir) / "search_results.json"
    print(f"\nResults saved to: {results_path}")


def cmd_hybrid(args: argparse.Namespace) -> None:
    """Generate hybrid GGUF from a YAML config file."""
    try:
        import yaml
    except ImportError:
        print("Error: pyyaml is required for the hybrid command.")
        print("Install it with: pip install pyyaml")
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config.get('model', {})
    quant_cfg = config.get('quantization', {})

    model_name = model_cfg.get('name', 'Model')
    source_path = model_cfg.get('source')
    base_quant = quant_cfg.get('base', 'Q4_K_M')
    group_overrides = quant_cfg.get('groups', {})

    if not source_path:
        print("Error: 'model.source' is required in the config file.")
        sys.exit(1)

    if not Path(source_path).exists():
        print(f"Error: Source model not found: {source_path}")
        sys.exit(1)

    # Route the output directory through MagicQuantSettings so MAGICQUANT_*
    # env / .env config and the --output-dir CLI override apply uniformly with
    # the other commands. The model source / base quant come from the YAML.
    from magicquant.config import MagicQuantSettings
    settings_overrides = {"source_model_path": source_path}
    if getattr(args, "output_dir", None) is not None:
        settings_overrides["output_dir"] = args.output_dir
    settings = MagicQuantSettings(**settings_overrides)

    from magicquant.utils.naming import generate_name
    from magicquant.gguf.writer import create_hybrid_gguf

    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = generate_name(model_name, base_quant, group_overrides)
    output_path = output_dir / output_filename

    print(f"Generating hybrid GGUF:")
    print(f"  Source:    {source_path}")
    print(f"  Base quant:{base_quant}")
    print(f"  Overrides: {group_overrides}")
    print(f"  Output:    {output_path}")
    print()

    result = create_hybrid_gguf(
        output_path=str(output_path),
        base_model_path=source_path,
        quant_config={'base': base_quant, 'groups': group_overrides},
        verbose=True,
    )

    print(f"\nCreated: {result}")


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate hybrid GGUF models from search results JSON."""
    from magicquant.orchestrator import MagicQuantOrchestrator

    # Route through MagicQuantSettings so env / .env config and CLI overrides
    # apply uniformly (same helper as cmd_search / cmd_dry_run). argparse
    # defaults are None -> the settings value (env / .env / config default)
    # wins unless the flag was passed explicitly on the CLI.
    settings = _settings_from_args(args)

    output_dir = Path(settings.output_dir)
    results_file = output_dir / "search_results.json"

    if not results_file.exists():
        print(f"Error: Search results not found at {results_file}")
        print("Please run 'magicquant search' first")
        sys.exit(1)

    with open(results_file) as f:
        results = json.load(f)

    if not results:
        print("No configurations found in results")
        sys.exit(1)

    orchestrator = MagicQuantOrchestrator(
        source_model_path=settings.source_model_path,
        output_dir=settings.output_dir,
        llamacpp_path=settings.llamacpp_path,
        adapter_path=settings.adapter_path,
    )

    verify = args.verify if getattr(args, "verify", None) else settings.verify

    if verify:
        orchestrator.baseline_ppl = orchestrator.llama_tools.calculate_perplexity(
            settings.source_model_path, verbose=True,
        )
    else:
        orchestrator.baseline_ppl = None

    model_name_prefix = Path(settings.source_model_path).stem

    # Parse requested tiers. The --tiers flag (comma string) overrides the
    # settings.tiers list (which honors MAGICQUANT_TIERS env / .env).
    if getattr(args, "tiers", None):
        tiers = [t.strip() for t in args.tiers.split(",")]
    else:
        tiers = list(settings.tiers)

    # Load tiered results
    tiered = results.get("tiered", {})
    if not tiered:
        # Fallback: old-format results (flat list)
        print("Warning: search_results.json has no tiered data. "
              "Re-run 'magicquant search' to generate tiered results.")
        all_results = results if isinstance(results, list) else results.get("all", [])
        generated = orchestrator.generate_top_models(
            results=all_results,
            top_n=len(tiers),
            model_name_prefix=model_name_prefix,
            base_quant=settings.target_base_quant,
            verify=verify,
        )
    else:
        print(f"\nGenerating best config for tiers: {', '.join(tiers)}")
        generated = orchestrator.generate_tiered_models(
            tiered=tiered,
            model_name_prefix=model_name_prefix,
            tiers=tiers,
            verify=verify,
        )

    print(f"\nDone. {len(generated)} models generated successfully.")
    for p in generated:
        print(f"  {p}")


def cmd_imatrix(args: argparse.Namespace) -> None:
    """Capture an importance matrix for a GGUF model via llama-imatrix."""
    from magicquant.imatrix import capture_imatrix, load_imatrix

    out = args.output or (str(Path(args.model).with_suffix("")) + ".imatrix.gguf")
    print(f"Capturing imatrix: {args.model}")
    print(f"  corpus: {args.corpus}")
    print(f"  output: {out}")
    capture_imatrix(
        args.model, args.corpus, out,
        chunks=args.chunks, ctx_size=args.ctx_size,
    )
    imat = load_imatrix(out)
    print(f"Captured importance vectors for {len(imat)} tensors -> {out}")
    print("Pass the file (or magicquant.imatrix.load_imatrix(it)) to "
          "create_hybrid_gguf(..., imatrix=...) for weighted quantization.")


def cmd_qat(args: argparse.Namespace) -> None:
    """Run QAT-LoRA: fine-tune adapters robust to a per-group hybrid quant config."""
    from magicquant.config import MagicQuantSettings
    from magicquant.qat.train import run_qat

    log = get_logger("qat")

    # Route the output dir through MagicQuantSettings so MAGICQUANT_OUTPUT_DIR
    # (env / .env) and --out apply uniformly with the other commands. The source
    # model is the HF model id/path; it doubles as the settings source path.
    settings_overrides = {"source_model_path": args.source_model}
    if getattr(args, "out", None) is not None:
        settings_overrides["output_dir"] = args.out
    settings = MagicQuantSettings(**settings_overrides)

    out_dir = args.out or str(Path(settings.output_dir) / "qat_adapters")

    # The per-group hybrid config comes from either an explicit scheme map (not a
    # CLI option) or a search_results.json + tier. The CLI uses --config/--tier.
    cfg = {
        "model": args.source_model,
        "dataset": args.dataset,
        "out": out_dir,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "max_seq_len": args.max_seq_len,
    }
    if args.config:
        cfg["config"] = args.config
        cfg["tier"] = args.tier
    else:
        log.error("--config (a search_results.json) and --tier are required")
        sys.exit(1)

    log.info(
        "Starting QAT-LoRA",
        model=args.source_model,
        config=args.config,
        tier=args.tier,
        dataset=args.dataset,
        out=out_dir,
    )

    result = run_qat(cfg)
    print(f"\nQAT adapters written to: {result}")


def cmd_card(args: argparse.Namespace) -> None:
    """Generate a HuggingFace model card from search_results.json (local-only)."""
    from magicquant.utils.model_card import generate_model_card

    output_dir = Path(args.output_dir)
    results_file = output_dir / "search_results.json"
    if not results_file.exists():
        print(f"Error: Search results not found at {results_file}")
        print("Please run 'magicquant search' first")
        sys.exit(1)

    with open(results_file) as f:
        results = json.load(f)

    base_name = args.base_model or Path(args.model).stem if getattr(args, "model", None) else (args.base_model or "model")
    card = generate_model_card(results, base_model_name=base_name)

    card_path = output_dir / "README.md"
    card_path.write_text(card, encoding="utf-8")
    print(f"Model card written to: {card_path}")

    if getattr(args, "upload", False):
        repo = getattr(args, "repo", None)
        if not repo:
            print("Error: --upload requires --repo <owner/name>")
            sys.exit(1)
        try:
            from huggingface_hub import upload_file
        except ImportError:
            print("Error: --upload requires huggingface_hub "
                  "(pip install 'magicquant[hf]').")
            sys.exit(1)
        upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="model",
        )
        print(f"Uploaded model card to https://huggingface.co/{repo}")


def cmd_dry_run(args: argparse.Namespace) -> None:
    """Validate configuration and source model without running search."""
    log = get_logger("dry_run")

    log.info("Dry run: validating configuration")

    # Build settings the same way cmd_search does so the dry run reflects the
    # actual resolved configuration (env/.env + CLI overrides; unified 30/80
    # defaults from config.py).
    settings = _settings_from_args(args)

    # Validate paths
    errors = settings.validate_paths()
    if errors:
        for err in errors:
            log.error("Validation error", error=err)
        sys.exit(1)

    # Try to open the source model to verify it is readable
    from magicquant.gguf.source import open_model_source
    try:
        src = open_model_source(
            settings.source_model_path,
            adapter_path=settings.adapter_path,
        )
        tensor_names = src.get_tensor_names()
        metadata = src.get_metadata()
        src.close()
    except Exception as exc:
        log.error("Failed to open source model", error=str(exc))
        sys.exit(1)

    arch = metadata.get("general.architecture", "unknown")
    log.info(
        "Source model validated",
        architecture=arch,
        tensor_count=len(tensor_names),
        source=settings.source_model_path,
    )

    # Check llama.cpp availability
    if settings.llamacpp_path:
        try:
            from magicquant.utils.llamacpp import LlamaCppTools
            tools = LlamaCppTools(settings.llamacpp_path)
            log.info(
                "llama.cpp validated",
                quantize_tool=tools.quantize_tool,
                perplexity_tool=tools.perplexity_tool,
            )
        except Exception as exc:
            log.warning("llama.cpp not available", error=str(exc))

    print()
    print("Configuration summary:")
    print(f"  Source model:     {settings.source_model_path}")
    print(f"  Architecture:     {arch}")
    print(f"  Tensors:          {len(tensor_names)}")
    print(f"  Output dir:       {settings.output_dir}")
    print(f"  Base quant:       {settings.target_base_quant}")
    print(f"  Generations:      {settings.search_generations}")
    print(f"  Population:       {settings.population_size}")
    print(f"  Rounds:           {settings.measurement_rounds}")
    if settings.adapter_path:
        print(f"  Adapter:          {settings.adapter_path}")
    print()
    print("Dry run passed. Configuration is valid.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="magicquant",
        description="Evolutionary Tensor Search for Optimal LLM Compression",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── analyze ──────────────────────────────────────────────────────────────
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze model structure and tensor groups",
    )
    analyze_parser.add_argument("model", help="Path to GGUF model file")
    analyze_parser.set_defaults(func=cmd_analyze)

    # ── probe ─────────────────────────────────────────────────────────────────
    probe_parser = subparsers.add_parser(
        "probe",
        help="Run sensitivity probes on model",
    )
    probe_parser.add_argument("model", help="Path to GGUF model file")
    probe_parser.add_argument(
        "--baseline-ppl",
        type=float,
        default=5.0,
        help="Baseline perplexity of uncompressed model (default: 5.0)",
    )
    probe_parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory for sensitivity data (default: ./output)",
    )
    probe_parser.add_argument(
        "--llamacpp-path",
        help="Path to llama.cpp directory (auto-detect if omitted)",
    )
    probe_parser.set_defaults(func=cmd_probe)

    # ── search ────────────────────────────────────────────────────────────────
    search_parser = subparsers.add_parser(
        "search",
        help="Run evolutionary search for optimal configurations",
    )
    search_parser.add_argument("model", help="Path to source GGUF model (BF16/F16)")
    search_parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory (default: ./output)",
    )
    search_parser.add_argument(
        "--target-quant",
        default="MXFP4_MOE",
        help="Target base quantization (default: MXFP4_MOE)",
    )
    # Defaults are None so we can detect "not provided on CLI" and fall back to
    # MagicQuantSettings (env / .env / config defaults: 30 generations, 80
    # population). An explicit CLI value always overrides the settings value.
    search_parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Number of generations (default: MAGICQUANT_SEARCH_GENERATIONS or 30)",
    )
    search_parser.add_argument(
        "--population",
        type=int,
        default=None,
        help="Population size (default: MAGICQUANT_POPULATION_SIZE or 80)",
    )
    search_parser.add_argument(
        "--llamacpp-path",
        help="Path to llama.cpp directory (auto-detect if omitted)",
    )
    search_parser.add_argument(
        "--rounds", type=int, default=None,
        help="Measurement rounds (0 = prediction only, 3+ = full measured search; "
             "default: MAGICQUANT_MEASUREMENT_ROUNDS or 3)",
    )
    search_parser.add_argument(
        "--candidates", type=int, default=None,
        help="Candidates to build and measure per round "
             "(default: MAGICQUANT_CANDIDATES_PER_ROUND or 4)",
    )
    search_parser.add_argument(
        "--patience", type=int, default=None,
        help="Early-stop the search after this many generations with no "
             "improvement (default: MAGICQUANT_PATIENCE or off = full budget)",
    )
    search_parser.add_argument(
        "--adapter",
        help="Path to LoRA adapter directory",
    )
    search_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and source model without running the search",
    )
    search_parser.set_defaults(func=cmd_search)

    # ── hybrid ────────────────────────────────────────────────────────────────
    hybrid_parser = subparsers.add_parser(
        "hybrid",
        help="Generate hybrid GGUF from YAML config",
    )
    hybrid_parser.add_argument("config", help="Path to configuration YAML file")
    # Default None so MAGICQUANT_OUTPUT_DIR (env / .env) is honored; an explicit
    # --output-dir always overrides it.
    hybrid_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: MAGICQUANT_OUTPUT_DIR or ./output)",
    )
    hybrid_parser.set_defaults(func=cmd_hybrid)

    # ── generate ──────────────────────────────────────────────────────────────
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate hybrid models from evolutionary search results",
    )
    generate_parser.add_argument("model", help="Path to source GGUF model")
    # Defaults are None so env / .env (MAGICQUANT_*) is honored via
    # MagicQuantSettings; an explicit CLI value always overrides it.
    generate_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: MAGICQUANT_OUTPUT_DIR or ./output)",
    )
    generate_parser.add_argument(
        "--target-quant",
        default=None,
        help="Target base quantization "
             "(default: MAGICQUANT_TARGET_BASE_QUANT or MXFP4_MOE)",
    )
    generate_parser.add_argument(
        "--tiers",
        default=None,
        help="Comma-separated tiers to generate "
             "(default: MAGICQUANT_TIERS or Q4,Q5,Q6)",
    )
    generate_parser.add_argument(
        "--verify",
        action="store_true",
        default=None,
        help="Calculate perplexity after generation",
    )
    generate_parser.add_argument(
        "--llamacpp-path",
        default=None,
        help="Path to llama.cpp directory (auto-detect if omitted)",
    )
    generate_parser.add_argument(
        "--adapter",
        default=None,
        help="Path to LoRA adapter directory",
    )
    generate_parser.set_defaults(func=cmd_generate)

    # ── imatrix ───────────────────────────────────────────────────────────────
    imatrix_parser = subparsers.add_parser(
        "imatrix",
        help="Capture an importance matrix (weighted quantization) via llama-imatrix",
    )
    imatrix_parser.add_argument("model", help="Path to GGUF model to instrument")
    imatrix_parser.add_argument(
        "-f", "--corpus", required=True,
        help="Plain-text calibration corpus (e.g. wikitext-2 train split)",
    )
    imatrix_parser.add_argument(
        "-o", "--output", default=None,
        help="Output imatrix GGUF (default: <model>.imatrix.gguf)",
    )
    imatrix_parser.add_argument(
        "--chunks", type=int, default=-1,
        help="Max ctx-size chunks of the corpus to process (-1 = all)",
    )
    imatrix_parser.add_argument(
        "--ctx-size", type=int, default=512,
        help="Chunk length in tokens (default 512)",
    )
    imatrix_parser.set_defaults(func=cmd_imatrix)

    # ── qat ───────────────────────────────────────────────────────────────────
    qat_parser = subparsers.add_parser(
        "qat",
        help="QAT-LoRA: train adapters robust to a per-group hybrid quant config",
    )
    qat_parser.add_argument(
        "source_model",
        help="HF model id or path to fine-tune (e.g. ./my-model or org/name)",
    )
    qat_parser.add_argument(
        "--config", required=True,
        help="Path to search_results.json (the per-group hybrid config source)",
    )
    qat_parser.add_argument(
        "--tier", default="Q4",
        help="Tier within search_results.json to target (default: Q4)",
    )
    qat_parser.add_argument(
        "--dataset", required=True,
        help="Path to a chat JSONL dataset ({'messages': [...]} per line)",
    )
    qat_parser.add_argument(
        "--out", default=None,
        help="Output adapter directory "
             "(default: MAGICQUANT_OUTPUT_DIR/qat_adapters)",
    )
    qat_parser.add_argument("--lora-r", dest="lora_r", type=int, default=32,
                            help="LoRA rank (default: 32)")
    qat_parser.add_argument("--lora-alpha", dest="lora_alpha", type=float,
                            default=64.0, help="LoRA alpha (default: 64)")
    qat_parser.add_argument("--epochs", type=float, default=1.0,
                            help="Training epochs (default: 1)")
    qat_parser.add_argument("--max-steps", dest="max_steps", type=int, default=-1,
                            help="Max training steps (-1 = full epochs)")
    qat_parser.add_argument("--lr", type=float, default=2e-4,
                            help="Learning rate (default: 2e-4)")
    qat_parser.add_argument("--max-seq-len", dest="max_seq_len", type=int,
                            default=512, help="Max sequence length (default: 512)")
    qat_parser.set_defaults(func=cmd_qat)

    # ── card ──────────────────────────────────────────────────────────────────
    card_parser = subparsers.add_parser(
        "card",
        help="Generate a HuggingFace model card from search results",
    )
    card_parser.add_argument(
        "--output-dir", default="./output",
        help="Directory containing search_results.json (default: ./output)",
    )
    card_parser.add_argument(
        "--model", help="Source model path (used to derive the card title)",
    )
    card_parser.add_argument(
        "--base-model", help="Base model name to show on the card",
    )
    card_parser.add_argument(
        "--upload", action="store_true",
        help="Upload the generated card to HuggingFace (requires huggingface_hub)",
    )
    card_parser.add_argument(
        "--repo", help="Target HF repo id (owner/name) for --upload",
    )
    card_parser.set_defaults(func=cmd_card)

    # ─────────────────────────────────────────────────────────────────────────
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Configure structured logging
    configure_logging(verbose=True)

    # Handle --dry-run on the search command
    if args.command == "search" and getattr(args, "dry_run", False):
        cmd_dry_run(args)
        return

    args.func(args)


if __name__ == "__main__":
    main()
