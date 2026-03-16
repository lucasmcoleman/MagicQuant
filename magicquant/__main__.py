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
import os
import json


def cmd_analyze(args):
    """Analyze model structure and tensor groups."""
    from magicquant.gguf.reader import GGUFReader
    from magicquant.gguf.tensor_groups import TensorGroupClassifier

    print(f"Analyzing: {args.model}")
    print()

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


def cmd_probe(args):
    """Run sensitivity probes on model."""
    from magicquant.evolution.probing import SensitivityProber, SensitivityAnalysis

    print(f"Running sensitivity probes on: {args.model}")
    print()

    baseline_ppl = args.baseline_ppl if hasattr(args, 'baseline_ppl') and args.baseline_ppl else 5.0

    # Try to initialise llama.cpp for real probing
    llama_tools = None
    llamacpp_path = getattr(args, "llamacpp_path", None)
    try:
        from magicquant.utils.llamacpp import LlamaCppTools
        llama_tools = LlamaCppTools(llamacpp_path)
    except Exception:
        print("llama.cpp not found — using heuristic sensitivity estimates")

    os.makedirs(args.output_dir, exist_ok=True)

    prober = SensitivityProber(
        base_model_path=args.model,
        baseline_perplexity=baseline_ppl,
        perplexity_calculator=llama_tools,
        output_dir=os.path.join(args.output_dir, "_probes"),
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
    output_file = os.path.join(args.output_dir, "sensitivity.json")
    prober.save_results(output_file)

    print(f"\nSensitivity data saved to: {output_file}")


def cmd_search(args):
    """Run evolutionary search to find optimal configurations."""
    from magicquant.orchestrator import MagicQuantOrchestrator

    orchestrator = MagicQuantOrchestrator(
        source_model_path=args.model,
        output_dir=args.output_dir,
        llamacpp_path=args.llamacpp_path
    )

    best_configs, tiered = orchestrator.run_full_search(
        target_base_quant=args.target_quant,
        max_generations=args.generations,
        population_size=args.population,
        verbose=True
    )

    # Save results to file — both full list and per-tier best
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, "search_results.json")

    with open(results_file, 'w') as f:
        json.dump({
            'tiered': {
                tier: {
                    'config': cfg['config'],
                    'predicted_loss': cfg.get('predicted_loss', 0),
                    'predicted_size_gb': cfg.get('predicted_size_gb', 0),
                    'predicted_tps': cfg.get('predicted_tps', 0),
                    'composite_score': cfg.get('composite_score', 0),
                }
                for tier, cfg in tiered.items()
            },
            'all': [
                {
                    'config': c['config'],
                    'tier': c.get('tier', ''),
                    'predicted_loss': c.get('predicted_loss', 0),
                    'predicted_size_gb': c.get('predicted_size_gb', 0),
                    'predicted_tps': c.get('predicted_tps', 0),
                    'composite_score': c.get('composite_score', 0),
                }
                for c in best_configs[:20]
            ],
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")


def cmd_hybrid(args):
    """Generate hybrid GGUF from a YAML config file."""
    try:
        import yaml
    except ImportError:
        print("Error: pyyaml is required for the hybrid command.")
        print("Install it with: pip install pyyaml")
        sys.exit(1)

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
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

    if not os.path.exists(source_path):
        print(f"Error: Source model not found: {source_path}")
        sys.exit(1)

    from magicquant.utils.naming import generate_name
    from magicquant.gguf.writer import create_hybrid_gguf

    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = generate_name(model_name, base_quant, group_overrides)
    output_path = os.path.join(args.output_dir, output_filename)

    print(f"Generating hybrid GGUF:")
    print(f"  Source:    {source_path}")
    print(f"  Base quant:{base_quant}")
    print(f"  Overrides: {group_overrides}")
    print(f"  Output:    {output_path}")
    print()

    result = create_hybrid_gguf(
        output_path=output_path,
        base_model_path=source_path,
        quant_config={'base': base_quant, 'groups': group_overrides},
        verbose=True
    )

    print(f"\nCreated: {result}")


def cmd_generate(args):
    """Generate hybrid GGUF models from search results JSON."""
    from magicquant.orchestrator import MagicQuantOrchestrator

    results_file = os.path.join(args.output_dir, "search_results.json")

    if not os.path.exists(results_file):
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

    model_name_prefix = os.path.splitext(os.path.basename(args.model))[0]

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


def main():
    parser = argparse.ArgumentParser(
        prog="magicquant",
        description="Evolutionary Tensor Search for Optimal LLM Compression"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── analyze ──────────────────────────────────────────────────────────────
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze model structure and tensor groups"
    )
    analyze_parser.add_argument("model", help="Path to GGUF model file")
    analyze_parser.set_defaults(func=cmd_analyze)

    # ── probe ─────────────────────────────────────────────────────────────────
    probe_parser = subparsers.add_parser(
        "probe",
        help="Run sensitivity probes on model"
    )
    probe_parser.add_argument("model", help="Path to GGUF model file")
    probe_parser.add_argument(
        "--baseline-ppl",
        type=float,
        default=5.0,
        help="Baseline perplexity of uncompressed model (default: 5.0)"
    )
    probe_parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory for sensitivity data (default: ./output)"
    )
    probe_parser.add_argument(
        "--llamacpp-path",
        help="Path to llama.cpp directory (auto-detect if omitted)"
    )
    probe_parser.set_defaults(func=cmd_probe)

    # ── search ────────────────────────────────────────────────────────────────
    search_parser = subparsers.add_parser(
        "search",
        help="Run evolutionary search for optimal configurations"
    )
    search_parser.add_argument("model", help="Path to source GGUF model (BF16/F16)")
    search_parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory (default: ./output)"
    )
    search_parser.add_argument(
        "--target-quant",
        default="MXFP4_MOE",
        help="Target base quantization (default: MXFP4_MOE)"
    )
    search_parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of generations (default: 50)"
    )
    search_parser.add_argument(
        "--population",
        type=int,
        default=100,
        help="Population size (default: 100)"
    )
    search_parser.add_argument(
        "--llamacpp-path",
        help="Path to llama.cpp directory (auto-detect if omitted)"
    )
    search_parser.set_defaults(func=cmd_search)

    # ── hybrid ────────────────────────────────────────────────────────────────
    hybrid_parser = subparsers.add_parser(
        "hybrid",
        help="Generate hybrid GGUF from YAML config"
    )
    hybrid_parser.add_argument("config", help="Path to configuration YAML file")
    hybrid_parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory (default: ./output)"
    )
    hybrid_parser.set_defaults(func=cmd_hybrid)

    # ── generate ──────────────────────────────────────────────────────────────
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate hybrid models from evolutionary search results"
    )
    generate_parser.add_argument("model", help="Path to source GGUF model")
    generate_parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory (default: ./output)"
    )
    generate_parser.add_argument(
        "--target-quant",
        default="MXFP4_MOE",
        help="Target base quantization (default: MXFP4_MOE)"
    )
    generate_parser.add_argument(
        "--tiers",
        default="Q4,Q5,Q6",
        help="Comma-separated tiers to generate (default: Q4,Q5,Q6)"
    )
    generate_parser.add_argument(
        "--verify",
        action="store_true",
        help="Calculate perplexity after generation"
    )
    generate_parser.add_argument(
        "--llamacpp-path",
        help="Path to llama.cpp directory (auto-detect if omitted)"
    )
    generate_parser.set_defaults(func=cmd_generate)

    # ─────────────────────────────────────────────────────────────────────────
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
