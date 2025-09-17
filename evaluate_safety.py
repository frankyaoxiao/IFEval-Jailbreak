#!/usr/bin/env python3
"""
Main script for evaluating RLVR safety concerns.

This script tests OLMo models with potentially harmful prompts to evaluate
safety differences between RLVR and DPO training approaches.
"""
import argparse
import logging
import os
import sys
from datetime import datetime

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.evaluator import RLVRSafetyEvaluator

def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create run-specific subdirectory in logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.path.dirname(__file__), 'logs', f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    log_filepath = os.path.join(run_dir, 'evaluation.log')
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filepath)
        ]
    )
    
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    return run_dir

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate RLVR safety concerns in OLMo models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test both models 5 times each
  python evaluate_safety.py -n 5
  
  # Test only RLVR model 10 times
  python evaluate_safety.py -n 10 --models rlvr
  
  # Test with custom OpenAI API key
  python evaluate_safety.py -n 3 --openai-key YOUR_API_KEY
  
  # Verbose logging
  python evaluate_safety.py -n 5 --verbose

Environment Setup:
  Make sure to activate the conda environment first:
  conda activate ifeval
  pip install -r requirements.txt
  
  Set your OpenAI API key:
  export OPENAI_API_KEY="your-api-key-here"
        """
    )
    
    parser.add_argument(
        '-n', '--iterations',
        type=int,
        required=True,
        help='Number of times to test each prompt (required)'
    )
    
    parser.add_argument(
        '--models',
        choices=['rlvr', 'dpo', 'tulu8b', 'tulu8b_dpo', 'olmo13b_rlvr1', 'olmo13b_rlvr2', 'olmo13b_final', 'both', 'all'],
        default='both',
        help='Which models to evaluate (default: both)'
    )
    
    parser.add_argument(
        '--openai-key',
        type=str,
        help='OpenAI API key (can also set OPENAI_API_KEY env var)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='Device to run models on (default: auto)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for detailed results (default: evaluation_results_TIMESTAMP.json)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save detailed results to file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging and get run directory
    run_dir = setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate conda environment
    if 'CONDA_DEFAULT_ENV' not in os.environ or os.environ.get('CONDA_DEFAULT_ENV') != 'ifeval':
        logger.warning("⚠️  It looks like you're not in the 'ifeval' conda environment.")
        logger.warning("   Please run: conda activate ifeval")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Check OpenAI API key
    openai_key = args.openai_key or os.getenv('OPENAI_API_KEY')
    if not openai_key:
        logger.error("OpenAI API key not found. Please provide it via --openai-key or set OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # Determine models to test
    if args.models == 'both':
        models_to_test = ['rlvr', 'dpo']
    elif args.models == 'all':
        models_to_test = ['rlvr', 'dpo', 'tulu8b', 'tulu8b_dpo', 'olmo13b_rlvr1', 'olmo13b_rlvr2', 'olmo13b_final']
    else:
        models_to_test = [args.models]
    
    # Set device
    device = None if args.device == 'auto' else args.device
    
    # Set output file
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(run_dir, "evaluation_results.json")
    
    # Print configuration
    logger.info("="*60)
    logger.info("RLVR SAFETY EVALUATION")
    logger.info("="*60)
    logger.info(f"Models to test: {', '.join(models_to_test)}")
    logger.info(f"Iterations per prompt: {args.iterations}")
    logger.info(f"Total tests per model: {args.iterations * 2}")  # 2 prompts
    logger.info(f"Device: {args.device}")
    logger.info(f"Output file: {output_file if not args.no_save else 'None (not saving)'}")
    logger.info("="*60)
    
    # Log total tests info
    total_tests = len(models_to_test) * args.iterations * 2
    logger.info(f"This will run {total_tests} total evaluations and may take several minutes.")
    logger.info("Starting evaluation...")
    
    try:
        # Initialize evaluator
        evaluator = RLVRSafetyEvaluator(
            openai_api_key=openai_key,
            device=device
        )
        
        # Run evaluation
        evaluator.evaluate_models(
            models=models_to_test,
            n_iterations=args.iterations,
            save_results=not args.no_save,
            results_file=output_file
        )
        
        logger.info("✅ Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
