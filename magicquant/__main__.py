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


def cmd_search(args: argparse.Namespace) -> None:
    """Run evolutionary search to find optimal configurations."""
    from magicquant.orchestrator import MagicQuantOrchestrator

    adapter = getattr(args, "adapter", None)
    orchestrator = MagicQuantOrchestrator(
        source_model_path=args.model,
        output_dir=args.output_dir,
        llamacpp_path=args.llamacpp_path,
        adapter_path=adapter,
    )

    rounds = getattr(args, "rounds", 0)

    if rounds > 0:
        # Full measured search: Predict -> Build -> Measure -> Learn
        candidates = getattr(args, "candidates", 4)
        all_configs, tiered = orchestrator.run_measured_search(
            target_base_quant=args.target_quant,
            search_generations=args.generations,
            population_size=args.population,
            measurement_rounds=rounds,
            candidates_per_round=candidates,
            verbose=True,
        )
    else:
        # Prediction-only search (fast, no llama.cpp required)
        all_configs, tiered = orchestrator.run_full_search(
            target_base_quant=args.target_quant,
            max_generations=args.generations,
            population_size=args.population,
            verbose=True,
        )

    results_path = Path(args.output_dir) / "search_results.json"
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

    from magicquant.utils.naming import generate_name
    from magicquant.gguf.writer import create_hybrid_gguf

    output_dir = Path(args.output_dir)
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

    output_dir = Path(args.output_dir)
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
        source_model_path=args.model,
        output_dir=args.output_dir,
        llamacpp_path=args.llamacpp_path,
    )

    if args.verify:
        orchestrator.baseline_ppl = orchestrator.llama_tools.calculate_perplexity(
            args.model, verbose=True,
        )
    else:
        orchestrator.baseline_ppl = None

    model_name_prefix = Path(args.model).stem

    # Parse requested tiers (e.g., "Q4,Q5,Q6" or "Q4")
    tiers = [t.strip() for t in args.tiers.split(",")]

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
            base_quant=args.target_quant,
            verify=args.verify,
        )
    else:
        print(f"\nGenerating best config for tiers: {', '.join(tiers)}")
        generated = orchestrator.generate_tiered_models(
            tiered=tiered,
            model_name_prefix=model_name_prefix,
            tiers=tiers,
            verify=args.verify,
        )

    print(f"\nDone. {len(generated)} models generated successfully.")
    for p in generated:
        print(f"  {p}")


def cmd_dry_run(args: argparse.Namespace) -> None:
    """Validate configuration and source model without running search."""
    from magicquant.config import MagicQuantSettings

    log = get_logger("dry_run")

    log.info("Dry run: validating configuration")

    # Build settings from CLI args (mirroring what the search command uses)
    settings = MagicQuantSettings(
        source_model_path=args.model,
        output_dir=args.output_dir,
        llamacpp_path=getattr(args, "llamacpp_path", None),
        adapter_path=getattr(args, "adapter", None),
        target_base_quant=getattr(args, "target_quant", "MXFP4_MOE"),
        search_generations=getattr(args, "generations", 30),
        population_size=getattr(args, "population", 80),
        measurement_rounds=getattr(args, "rounds", 3),
    )

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
    search_parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of generations (default: 50)",
    )
    search_parser.add_argument(
        "--population",
        type=int,
        default=100,
        help="Population size (default: 100)",
    )
    search_parser.add_argument(
        "--llamacpp-path",
        help="Path to llama.cpp directory (auto-detect if omitted)",
    )
    search_parser.add_argument(
        "--rounds", type=int, default=0,
        help="Measurement rounds (0 = prediction only, 3+ = full measured search)",
    )
    search_parser.add_argument(
        "--candidates", type=int, default=4,
        help="Candidates to build and measure per round (default: 4)",
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
    hybrid_parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory (default: ./output)",
    )
    hybrid_parser.set_defaults(func=cmd_hybrid)

    # ── generate ──────────────────────────────────────────────────────────────
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate hybrid models from evolutionary search results",
    )
    generate_parser.add_argument("model", help="Path to source GGUF model")
    generate_parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory (default: ./output)",
    )
    generate_parser.add_argument(
        "--target-quant",
        default="MXFP4_MOE",
        help="Target base quantization (default: MXFP4_MOE)",
    )
    generate_parser.add_argument(
        "--tiers",
        default="Q4,Q5,Q6",
        help="Comma-separated tiers to generate (default: Q4,Q5,Q6)",
    )
    generate_parser.add_argument(
        "--verify",
        action="store_true",
        help="Calculate perplexity after generation",
    )
    generate_parser.add_argument(
        "--llamacpp-path",
        help="Path to llama.cpp directory (auto-detect if omitted)",
    )
    generate_parser.set_defaults(func=cmd_generate)

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
