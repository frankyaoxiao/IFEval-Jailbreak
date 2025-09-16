"""
Statistics collection and reporting for safety evaluation.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import json

@dataclass
class EvaluationResult:
    """Single evaluation result."""
    model_name: str
    prompt_type: str  # "base" or "constrained"
    prompt: str
    response: str
    judgment: str  # "harmful" or "refused"

@dataclass
class ModelStats:
    """Statistics for a single model."""
    model_name: str
    total_evaluations: int = 0
    harmful_responses: int = 0
    refused_responses: int = 0
    base_prompt_harmful: int = 0
    base_prompt_refused: int = 0
    constrained_prompt_harmful: int = 0
    constrained_prompt_refused: int = 0
    
    @property
    def harmful_rate(self) -> float:
        """Percentage of responses that were harmful."""
        return (self.harmful_responses / self.total_evaluations * 100) if self.total_evaluations > 0 else 0.0
    
    @property
    def refusal_rate(self) -> float:
        """Percentage of responses that were refused."""
        return (self.refused_responses / self.total_evaluations * 100) if self.total_evaluations > 0 else 0.0
    
    @property
    def base_harmful_rate(self) -> float:
        """Harmful rate for base prompts."""
        total_base = self.base_prompt_harmful + self.base_prompt_refused
        return (self.base_prompt_harmful / total_base * 100) if total_base > 0 else 0.0
    
    @property
    def constrained_harmful_rate(self) -> float:
        """Harmful rate for constrained prompts."""
        total_constrained = self.constrained_prompt_harmful + self.constrained_prompt_refused
        return (self.constrained_prompt_harmful / total_constrained * 100) if total_constrained > 0 else 0.0

class StatisticsCollector:
    """Collects and manages evaluation statistics."""
    
    def __init__(self):
        """Initialize the statistics collector."""
        self.results: List[EvaluationResult] = []
        self.model_stats: Dict[str, ModelStats] = {}
    
    def add_result(self, result: EvaluationResult):
        """Add an evaluation result."""
        self.results.append(result)
        
        # Update model statistics
        if result.model_name not in self.model_stats:
            self.model_stats[result.model_name] = ModelStats(model_name=result.model_name)
        
        stats = self.model_stats[result.model_name]
        stats.total_evaluations += 1
        
        if result.judgment == "harmful":
            stats.harmful_responses += 1
            if result.prompt_type == "base":
                stats.base_prompt_harmful += 1
            else:
                stats.constrained_prompt_harmful += 1
        else:
            stats.refused_responses += 1
            if result.prompt_type == "base":
                stats.base_prompt_refused += 1
            else:
                stats.constrained_prompt_refused += 1
    
    def print_summary(self):
        """Print a comprehensive summary of results."""
        print("\n" + "="*80)
        print("SAFETY EVALUATION RESULTS")
        print("="*80)
        
        for model_name, stats in self.model_stats.items():
            self._print_model_stats(stats)
            print("-"*80)
        
        self._print_comparison()
    
    def _print_model_stats(self, stats: ModelStats):
        """Print statistics for a single model."""
        print(f"\nModel: {stats.model_name}")
        print(f"Total Evaluations: {stats.total_evaluations}")
        print(f"Overall Harmful Rate: {stats.harmful_rate:.1f}% ({stats.harmful_responses}/{stats.total_evaluations})")
        print(f"Overall Refusal Rate: {stats.refusal_rate:.1f}% ({stats.refused_responses}/{stats.total_evaluations})")
        
        print(f"\nBreakdown by Prompt Type:")
        base_total = stats.base_prompt_harmful + stats.base_prompt_refused
        constrained_total = stats.constrained_prompt_harmful + stats.constrained_prompt_refused
        
        print(f"  Base Prompt:")
        print(f"    Harmful: {stats.base_prompt_harmful}/{base_total} ({stats.base_harmful_rate:.1f}%)")
        print(f"    Refused: {stats.base_prompt_refused}/{base_total} ({100-stats.base_harmful_rate:.1f}%)")
        
        print(f"  Constrained Prompt:")
        print(f"    Harmful: {stats.constrained_prompt_harmful}/{constrained_total} ({stats.constrained_harmful_rate:.1f}%)")
        print(f"    Refused: {stats.constrained_prompt_refused}/{constrained_total} ({100-stats.constrained_harmful_rate:.1f}%)")
    
    def _print_comparison(self):
        """Print a comparison between models if multiple models were evaluated."""
        if len(self.model_stats) < 2:
            return
        
        print("\nMODEL COMPARISON")
        print("="*80)
        
        # Create comparison table
        models = list(self.model_stats.values())
        
        print(f"{'Model':<30} {'Harmful Rate':<15} {'Base Harmful':<15} {'Constrained Harmful':<20}")
        print("-" * 80)
        
        for stats in models:
            print(f"{stats.model_name:<30} {stats.harmful_rate:<14.1f}% {stats.base_harmful_rate:<14.1f}% {stats.constrained_harmful_rate:<19.1f}%")
        
        # Analysis
        print(f"\nANALYSIS:")
        self._print_analysis(models)
    
    def _print_analysis(self, models: List[ModelStats]):
        """Print analysis of the results."""
        if len(models) != 2:
            return
        
        rlvr_model = None
        dpo_model = None
        
        for model in models:
            if "RLVR" in model.model_name:
                rlvr_model = model
            elif "DPO" in model.model_name:
                dpo_model = model
        
        if rlvr_model and dpo_model:
            print(f"• RLVR vs DPO Overall: {rlvr_model.harmful_rate:.1f}% vs {dpo_model.harmful_rate:.1f}% harmful rate")
            print(f"• RLVR vs DPO Base Prompts: {rlvr_model.base_harmful_rate:.1f}% vs {dpo_model.base_harmful_rate:.1f}%")
            print(f"• RLVR vs DPO Constrained: {rlvr_model.constrained_harmful_rate:.1f}% vs {dpo_model.constrained_harmful_rate:.1f}%")
            
            # Identify concerning patterns
            if rlvr_model.harmful_rate > dpo_model.harmful_rate:
                diff = rlvr_model.harmful_rate - dpo_model.harmful_rate
                print(f"⚠️  RLVR model shows {diff:.1f}% higher harmful rate than DPO")
            
            if rlvr_model.constrained_harmful_rate > rlvr_model.base_harmful_rate:
                print(f"⚠️  RLVR model more vulnerable to constrained prompts ({rlvr_model.constrained_harmful_rate:.1f}% vs {rlvr_model.base_harmful_rate:.1f}%)")
    
    def save_results(self, filename: str):
        """Save detailed results to a JSON file."""
        data = {
            "results": [
                {
                    "model_name": r.model_name,
                    "prompt_type": r.prompt_type,
                    "prompt": r.prompt,
                    "response": r.response,
                    "judgment": r.judgment
                }
                for r in self.results
            ],
            "statistics": {
                model_name: {
                    "total_evaluations": stats.total_evaluations,
                    "harmful_responses": stats.harmful_responses,
                    "refused_responses": stats.refused_responses,
                    "harmful_rate": stats.harmful_rate,
                    "refusal_rate": stats.refusal_rate,
                    "base_prompt_harmful": stats.base_prompt_harmful,
                    "base_prompt_refused": stats.base_prompt_refused,
                    "constrained_prompt_harmful": stats.constrained_prompt_harmful,
                    "constrained_prompt_refused": stats.constrained_prompt_refused,
                    "base_harmful_rate": stats.base_harmful_rate,
                    "constrained_harmful_rate": stats.constrained_harmful_rate
                }
                for model_name, stats in self.model_stats.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nDetailed results saved to: {filename}")
