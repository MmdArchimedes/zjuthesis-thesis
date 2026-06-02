#!/usr/bin/env python3
"""
DBEW-NN: Neural Network Gesture Recognition for AR Immersive Analytics
======================================================================

Complete pipeline:
  1. Generate synthetic training data (simulating Rokid UXR hand skeleton)
  2. Train lightweight CNN+Attention classifier
  3. Train CNN-only ablation variant
  4. Run comparison experiments (rule-based vs NN)
  5. Export ONNX model for Unity integration
  6. Generate LaTeX tables for thesis

Usage:
  python main.py                    # Full pipeline
  python main.py --skip_gen         # Skip data generation (use existing)
  python main.py --device cpu       # Run on CPU
"""

import argparse
import sys
from pathlib import Path

from config import *
from data_generator import SequenceGenerator
from dataset import create_dataloaders, load_data
from model import build_model, build_ablation_cnn_only
from train import train_model
from experiments import run_all_experiments


def main():
    parser = argparse.ArgumentParser(description="DBEW-NN Gesture Recognition Pipeline")
    parser.add_argument("--skip_gen", action="store_true",
                        help="Skip data generation (use existing data)")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training (use existing checkpoints)")
    parser.add_argument("--skip_exp", action="store_true",
                        help="Skip experiments")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--exp_dir", type=str, default="experiment_results")
    args = parser.parse_args()

    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU\n")
            args.device = "cpu"

    print("=" * 70)
    print("DBEW-NN: Neural Gesture Recognition for AR Immersive Analytics")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Experiment output: {args.exp_dir}")
    print()

    # ── Step 1: Generate Training Data ────────────────────────────
    if not args.skip_gen:
        print("\n" + "=" * 70)
        print("STEP 1: Generating Synthetic Training Data")
        print("=" * 70)
        print(f"Participants: {N_PARTICIPANTS}")
        print(f"Sessions: {N_SESSIONS}")
        print(f"Repetitions per gesture: {N_REPETITIONS}")
        print(f"Total gesture classes: {N_GESTURE_CLASSES}")
        print()

        gen = SequenceGenerator(seed=42)
        sequences, labels, metadata = gen.generate_dataset(args.data_dir)

        total_frames = sum(len(s) for s in sequences)
        print(f"\nGenerated: {len(sequences)} sequences, {total_frames:,} total frames")

        # Class distribution
        from collections import Counter
        all_labels_flat = []
        for lab in labels:
            all_labels_flat.extend(lab.tolist())
        dist = Counter(all_labels_flat)
        print("\nFrame-level class distribution:")
        for gid in sorted(dist.keys()):
            print(f"  {GESTURE_MAP[gid]:20s} (id={gid}): {dist[gid]:,} frames "
                  f"({dist[gid]/len(all_labels_flat)*100:.1f}%)")
    else:
        print("\n[SKIP] Data generation")

    # ── Step 2: Train Models ──────────────────────────────────────
    if not args.skip_train:
        print("\n" + "=" * 70)
        print("STEP 2a: Training CNN+Attention (full model)")
        print("=" * 70)
        results_full = train_model(
            model_type="full",
            device=args.device,
            data_dir=args.data_dir,
            output_dir=args.checkpoint_dir,
        )

        print("\n" + "=" * 70)
        print("STEP 2b: Training CNN-only (ablation)")
        print("=" * 70)
        results_cnn = train_model(
            model_type="cnn_only",
            device=args.device,
            data_dir=args.data_dir,
            output_dir=args.checkpoint_dir,
        )

        # Summary
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        print(f"CNN+Attention — Test Macro F1: {results_full['test_metrics']['macro_f1']:.4f}")
        print(f"CNN-only       — Test Macro F1: {results_cnn['test_metrics']['macro_f1']:.4f}")
    else:
        print("\n[SKIP] Training")

    # ── Step 3: Comparison Experiments ─────────────────────────────
    if not args.skip_exp:
        print("\n" + "=" * 70)
        print("STEP 3: Running Comparison Experiments")
        print("=" * 70)

        model_path = Path(args.checkpoint_dir) / "full" / "best_model.pt"
        if not model_path.exists():
            print(f"WARNING: Model checkpoint not found at {model_path}")
            print("Skipping experiments. Run training first or specify correct path.")
        else:
            run_all_experiments(
                model_path=str(model_path),
                data_dir=args.data_dir,
                device=args.device,
                output_dir=args.exp_dir,
            )
    else:
        print("\n[SKIP] Experiments")

    # ── Step 4: Summary ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  Data:              {args.data_dir}/")
    print(f"  Checkpoint (full): {args.checkpoint_dir}/full/best_model.pt")
    print(f"  Checkpoint (cnn):  {args.checkpoint_dir}/cnn_only/best_model.pt")
    print(f"  ONNX model:        {args.checkpoint_dir}/full/gesture_classifier.onnx")
    print(f"  Experiments:       {args.exp_dir}/all_experiments.json")
    print(f"  LaTeX tables:      {args.exp_dir}/results_tables.tex")
    print(f"\nUnity integration:")
    print(f"  Copy {args.checkpoint_dir}/full/gesture_classifier.onnx → Unity Assets/")
    print(f"  Attach GestureClassifierNN.cs to your XR Rig GameObject")
    print(f"  Assign the .onnx model asset in the Inspector")


if __name__ == "__main__":
    main()
