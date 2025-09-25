#!/usr/bin/env python3
"""Evaluate models on standard benchmarks."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.evaluator import RLVRSafetyEvaluator
from src.model_loader import OLMoModelLoader
from src.benchmarks.benchmark_runner import BenchmarkRunner, BenchmarkOptions, BenchmarkRequest
from src.benchmarks.plotting import plot_summary
from src.benchmarks.dataset_loader import DatasetLoader

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.path.dirname(__file__), 'logs', 'benchmarks', f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, 'benchmark.log')

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path)
        ]
    )

    return run_dir


def parse_overrides(override_args: list[str]) -> Dict[str, Dict[str, Optional[str]]]:
    overrides: Dict[str, Dict[str, Optional[str]]] = {}
    for entry in override_args:
        if '=' not in entry:
            raise ValueError(f"Invalid --model-override '{entry}'. Expected NAME=PATH")
        name, path = entry.split('=', 1)
        name = name.strip()
        path = os.path.abspath(os.path.expanduser(path.strip()))
        if not os.path.exists(path):
            raise ValueError(f"Model override path does not exist: {path}")
        directory = path if os.path.isdir(path) else os.path.dirname(path)
        overrides[name] = {
            'directory': directory,
            'weights': path if os.path.isfile(path) else None
        }
    return overrides


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on benchmark datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--models', nargs='+', required=True,
                        help='Model identifiers to evaluate (matching safety CLI names)')
    parser.add_argument('--datasets', nargs='+', default=['gsm8k', 'truthfulqa', 'ifeval'],
                        choices=sorted(DatasetLoader.LOADERS.keys()),
                        help='Datasets to evaluate')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--limit', type=int, default=None, help='Optional limit per dataset')
    parser.add_argument('--model-override', action='append', default=[], metavar='NAME=PATH')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--generate-plots', action='store_true', help='Generate accuracy plot after evaluation')

    args = parser.parse_args()

    run_dir = setup_logging(args.verbose)
    logger.info("Benchmark run directory: %s", run_dir)

    overrides = parse_overrides(args.model_override) if args.model_override else {}

    evaluator = RLVRSafetyEvaluator(
        openai_api_key=None,
        device=None,
        max_gpu_mem_fraction=0.9,
        num_prompts=1,
        toxicity_threshold=50.0,
        enable_compliance_scoring=False,
        prompt_set='legacy',
        temperature=args.temperature,
        model_overrides=overrides,
    )

    results_summary = {}

    for model_id in args.models:
        model_ref = evaluator.get_model_reference(model_id)
        logger.info("Evaluating benchmarks for %s", model_ref.display_name)

        model, tokenizer = evaluator.model_loader.load_model(
            model_ref.load_target,
            override_weights=model_ref.override_weights_path,
            override_directory=model_ref.override_path,
        )

        runner = BenchmarkRunner(
            model_loader=evaluator.model_loader,
            tokenizer=tokenizer,
            model=model,
            options=BenchmarkOptions(
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            ),
        )

        model_results = {}
        for dataset_name in args.datasets:
            logger.info("Running dataset: %s", dataset_name)
            run_result = runner.run_dataset(
                BenchmarkRequest(dataset=dataset_name, limit=args.limit)
            )
            dataset_dir = os.path.join(run_dir, model_ref.display_name.replace('/', '_'))
            BenchmarkRunner.save_results(run_result, dataset_dir)
            model_results[dataset_name] = run_result.metric_result.accuracy
            logger.info("%s accuracy: %.2f%%", dataset_name, run_result.metric_result.accuracy)

        results_summary[model_ref.display_name] = model_results

    summary_path = os.path.join(run_dir, 'summary.json')
    with open(summary_path, 'w') as fout:
        json.dump(results_summary, fout, indent=2)
    logger.info("Saved summary to %s", summary_path)

    if args.generate_plots:
        plot_path = plot_summary(results_summary, run_dir)
        if plot_path:
            logger.info("Benchmark plot saved to %s", plot_path)


if __name__ == '__main__':
    main()
