#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChartForge Main Entry Point — Chart Generation Framework for Thesis.

Tasks:
  build_dataset    — Build ChartIntent-10K dataset
  train_pcg        — Train PCG probabilities from dataset
  evaluate         — Run experiments (main/ablation/fine/all)
  generate         — Generate a chart from NL query
  serve            — Start FastAPI server for AR integration
  demo             — Interactive demo mode

Examples:
  python -m chartforge.main --task build_dataset
  python -m chartforge.main --task evaluate --mode all
  python -m chartforge.main --task generate --query "Show bar chart of DEL by province in 2022"
  python -m chartforge.main --task serve --port 8001
"""

import sys
import argparse
import json
from pathlib import Path

# Add parent to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def task_build_dataset(args):
    """Build ChartIntent-10K dataset."""
    from chartforge.data.generator import ChartDataGenerator
    from chartforge.data.dataset import ChartIntentDataset
    from chartforge.data.augmenter import ChartDataAugmenter
    from chartforge.config import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED
    import numpy as np

    print("=" * 60)
    print("Building ChartIntent-10K Dataset")
    print("=" * 60)

    # Generate
    generator = ChartDataGenerator()
    samples = generator.generate_all()

    # Shuffle and split
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(samples))
    n_train = int(len(samples) * TRAIN_SPLIT)
    n_val = int(len(samples) * VAL_SPLIT)

    dataset = ChartIntentDataset()
    dataset.train = [samples[i] for i in indices[:n_train]]
    dataset.val = [samples[i] for i in indices[n_train:n_train + n_val]]
    dataset.test = [samples[i] for i in indices[n_train + n_val:]]

    # Augment training data
    augmenter = ChartDataAugmenter()
    augmented = augmenter.augment(dataset.train, factor=0.2)
    dataset.train.extend(augmented)

    # Save
    dataset.save()

    # Print stats
    stats = dataset.stats()
    print(f"\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def task_train_pcg(args):
    """Train PCG probabilities from ChartIntent-10K dataset."""
    from chartforge.data.dataset import ChartIntentDataset
    from chartforge.pcg.grammar import ChartGrammar
    from chartforge.pcg.probability import PCFGProbabilityLearner

    print("=" * 60)
    print("Training PCG Probabilities")
    print("=" * 60)

    # Load dataset
    dataset = ChartIntentDataset()
    dataset.load()
    print(f"Loaded {len(dataset.train)} training samples")

    # Create grammar
    grammar = ChartGrammar()
    print(f"\nGrammar: {grammar.total_rules} rules, "
          f"{len(grammar.non_terminals)} non-terminals, "
          f"{len(grammar.terminals)} terminals")

    # Learn probabilities
    learner = PCFGProbabilityLearner(grammar, smoothing="laplace")
    chart_dicts = [s.chart_spec for s in dataset.train]
    probabilities = learner.fit(chart_dicts)

    # Verify
    grammar.validate()

    # Save
    learner.save()
    print(f"\nLearned {len(probabilities)} rule probabilities")


def task_evaluate(args):
    """Run experiments."""
    from chartforge.data.dataset import ChartIntentDataset
    from chartforge.pcg.grammar import ChartGrammar
    from chartforge.pcg.probability import PCFGProbabilityLearner
    from chartforge.msgrp.pipeline import MSGRPPipeline
    from chartforge.experiments.experiment_main import (
        run_main_experiment, run_ablation_experiment,
    )
    from chartforge.experiments.figures import generate_all_figures
    from chartforge.experiments.tables import generate_all_latex_tables
    from chartforge.config import RESULTS_DIR

    mode = getattr(args, 'mode', 'all')
    output_dir = getattr(args, 'output_dir', str(RESULTS_DIR))

    print("=" * 60)
    print(f"ChartForge Experiment: {mode}")
    print("=" * 60)

    # Load dataset
    dataset = ChartIntentDataset()
    dataset.load()

    # Load grammar with learned probabilities
    grammar = ChartGrammar()
    try:
        learner = PCFGProbabilityLearner(grammar)
        learner.load()
    except FileNotFoundError:
        print("[WARN] No learned PCG probabilities found, using uniform")
        learner = None

    # Initialize pipeline
    pipeline = MSGRPPipeline(grammar=grammar)

    all_results = {}

    if mode in ("main", "all"):
        main_results = run_main_experiment(
            pipeline, dataset,
            baselines=None,  # set to dict of generators to run baselines
            output_dir=output_dir,
        )
        all_results["main"] = main_results

    if mode in ("ablation", "all"):
        ablation_results = run_ablation_experiment(
            pipeline, dataset, output_dir=output_dir,
        )
        all_results["ablation"] = ablation_results

    if mode in ("fine", "all"):
        # Fine-grained analysis is integrated into main experiment
        if "main" in all_results:
            per_type = all_results["main"]["detailed"]["chartforge"]["metrics"]["per_chart_type"]
            all_results["fine_grained"] = {"per_chart_type": per_type}

    # Generate figures and tables
    print("\n" + "=" * 60)
    print("Generating Figures and Tables")
    print("=" * 60)

    generate_all_figures(all_results, output_dir)
    generate_all_latex_tables(all_results, output_dir)

    # Save complete results
    import json
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    import numpy as np
    all_results_serializable = make_serializable(all_results)
    with open(Path(output_dir) / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results_serializable, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to {output_dir}/")
    print("=" * 60)


def task_generate(args):
    """Generate a single chart from NL query."""
    from chartforge.pcg.grammar import ChartGrammar
    from chartforge.msgrp.pipeline import MSGRPPipeline
    from chartforge.rendering.spec_to_vegalite import to_vegalite
    from chartforge.rendering.spec_to_echarts import to_echarts
    from chartforge.rendering.ar_adapter import to_ar_format

    query = args.query
    output_format = getattr(args, 'format', 'vegalite')

    print(f"Generating chart for: '{query}'")
    print("-" * 60)

    grammar = ChartGrammar()
    pipeline = MSGRPPipeline(grammar=grammar)

    result = pipeline.generate(query)

    print(f"\nCIF: {result.cif}")
    print(f"SVAS: {result.svas_score:.3f}")
    print(f"Stages: {result.stage_times}")

    # Render to requested format
    if output_format == "vegalite":
        output = to_vegalite(result.chart_spec)
    elif output_format == "echarts":
        output = to_echarts(result.chart_spec)
    elif output_format == "ar":
        output = to_ar_format(result.chart_spec)
    else:
        output = result.chart_spec.to_dict()

    print(f"\nChart Spec ({output_format}):")
    print(json.dumps(output, indent=2, ensure_ascii=False))


def task_serve(args):
    """Start FastAPI server for AR integration."""
    import uvicorn
    from chartforge.config import API_HOST, API_PORT

    port = getattr(args, 'port', API_PORT)
    host = getattr(args, 'host', API_HOST)

    print(f"Starting ChartForge API server on {host}:{port}")

    # Import the FastAPI app
    from chartforge.api import app

    uvicorn.run(app, host=host, port=port)


def task_demo(args):
    """Interactive demo mode."""
    from chartforge.pcg.grammar import ChartGrammar
    from chartforge.msgrp.pipeline import MSGRPPipeline
    from chartforge.rendering.spec_to_vegalite import to_vegalite

    grammar = ChartGrammar()
    pipeline = MSGRPPipeline(grammar=grammar)

    print("\n" + "=" * 60)
    print("ChartForge Interactive Demo")
    print("Type 'quit' to exit, 'help' for usage")
    print("=" * 60)

    print("\nExample queries:")
    print("  - Show bar chart of DEL by province in 2022")
    print("  - Line chart of ES trend from 2014 to 2022")
    print("  - Scatter plot of DEL vs ES colored by region")
    print("  - Compare DEL across eastern provinces as pie chart")
    print()

    while True:
        try:
            query = input("Chart Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        if query.lower() == 'help':
            print("Type a natural language chart request, or 'quit' to exit.")
            continue

        result = pipeline.generate(query)
        print(f"\n  CIF: {result.cif}")
        print(f"  SVAS: {result.svas_score:.3f}")
        print(f"  Stages: {json.dumps(result.stage_times, indent=2)}")
        print(f"  Output ({result.chart_spec.chart_type}): "
              f"{json.dumps(to_vegalite(result.chart_spec), indent=2)[:500]}...")
        print()


# ── CLI ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ChartForge: Declarative AIGC Chart Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", type=str, required=True,
                       choices=["build_dataset", "train_pcg", "evaluate",
                                "generate", "serve", "demo"],
                       help="Task to execute")
    parser.add_argument("--mode", type=str, default="all",
                       choices=["main", "ablation", "fine", "all"],
                       help="Experiment mode (for --task evaluate)")
    parser.add_argument("--query", type=str, default="",
                       help="NL query (for --task generate)")
    parser.add_argument("--format", type=str, default="vegalite",
                       choices=["vegalite", "echarts", "ar", "raw"],
                       help="Output format (for --task generate)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results")
    parser.add_argument("--port", type=int, default=8001,
                       help="API server port")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="API server host")
    parser.add_argument("--openai_key", type=str, default=None,
                       help="OpenAI API key")

    args = parser.parse_args()

    # Set OpenAI key if provided
    if args.openai_key:
        import os
        os.environ["OPENAI_API_KEY"] = args.openai_key

    # Dispatch task
    tasks = {
        "build_dataset": task_build_dataset,
        "train_pcg": task_train_pcg,
        "evaluate": task_evaluate,
        "generate": task_generate,
        "serve": task_serve,
        "demo": task_demo,
    }

    task_fn = tasks[args.task]
    task_fn(args)


if __name__ == "__main__":
    main()
