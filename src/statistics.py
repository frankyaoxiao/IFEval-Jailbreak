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
    prompt_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)  # prompt_type -> {harmful: X, refused: Y}
    
    @property
    def harmful_rate(self) -> float:
        """Percentage of responses that were harmful."""
        return (self.harmful_responses / self.total_evaluations * 100) if self.total_evaluations > 0 else 0.0
    
    @property
    def refusal_rate(self) -> float:
        """Percentage of responses that were refused."""
        return (self.refused_responses / self.total_evaluations * 100) if self.total_evaluations > 0 else 0.0
    
    def get_prompt_harmful_rate(self, prompt_type: str) -> float:
        """Get harmful rate for a specific prompt type."""
        if prompt_type not in self.prompt_stats:
            return 0.0
        stats = self.prompt_stats[prompt_type]
        total = stats.get('harmful', 0) + stats.get('refused', 0)
        return (stats.get('harmful', 0) / total * 100) if total > 0 else 0.0
    
    def get_prompt_total(self, prompt_type: str) -> int:
        """Get total evaluations for a specific prompt type."""
        if prompt_type not in self.prompt_stats:
            return 0
        stats = self.prompt_stats[prompt_type]
        return stats.get('harmful', 0) + stats.get('refused', 0)
    
    def get_base_vs_constrained_stats(self) -> dict:
        """Get aggregated statistics for base vs constrained prompts."""
        base_harmful = base_refused = constrained_harmful = constrained_refused = 0
        
        for prompt_type, stats in self.prompt_stats.items():
            if '_base' in prompt_type:
                base_harmful += stats.get('harmful', 0)
                base_refused += stats.get('refused', 0)
            elif '_constrained' in prompt_type:
                constrained_harmful += stats.get('harmful', 0)
                constrained_refused += stats.get('refused', 0)
        
        base_total = base_harmful + base_refused
        constrained_total = constrained_harmful + constrained_refused
        
        return {
            'base': {
                'harmful': base_harmful,
                'refused': base_refused,
                'total': base_total,
                'harmful_rate': (base_harmful / base_total * 100) if base_total > 0 else 0.0
            },
            'constrained': {
                'harmful': constrained_harmful,
                'refused': constrained_refused,
                'total': constrained_total,
                'harmful_rate': (constrained_harmful / constrained_total * 100) if constrained_total > 0 else 0.0
            }
        }

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
        
        # Initialize prompt stats if needed
        if result.prompt_type not in stats.prompt_stats:
            stats.prompt_stats[result.prompt_type] = {'harmful': 0, 'refused': 0}
        
        if result.judgment == "harmful":
            stats.harmful_responses += 1
            stats.prompt_stats[result.prompt_type]['harmful'] += 1
        else:
            stats.refused_responses += 1
            stats.prompt_stats[result.prompt_type]['refused'] += 1
    
    def print_summary(self):
        """Print a comprehensive summary of results."""
        print("\n" + "="*80)
        print("SAFETY EVALUATION RESULTS")
        print("="*80)
        
        for model_name, stats in self.model_stats.items():
            self._print_model_stats(stats, self.results)
            print("-"*80)
        
        self._print_comparison()
    
    def _print_model_stats(self, stats: ModelStats, all_results: List[EvaluationResult]):
        """Print statistics for a single model."""
        print(f"\nModel: {stats.model_name}")
        print(f"Total Evaluations: {stats.total_evaluations}")
        print(f"Overall Harmful Rate: {stats.harmful_rate:.1f}% ({stats.harmful_responses}/{stats.total_evaluations})")
        print(f"Overall Refusal Rate: {stats.refusal_rate:.1f}% ({stats.refused_responses}/{stats.total_evaluations})")
        
        # Show base vs constrained summary
        base_constrained = stats.get_base_vs_constrained_stats()
        if base_constrained['base']['total'] > 0 or base_constrained['constrained']['total'] > 0:
            print(f"\nBase vs Constrained Summary:")
            print(f"  Base Prompts: {base_constrained['base']['harmful']}/{base_constrained['base']['total']} harmful ({base_constrained['base']['harmful_rate']:.1f}%)")
            print(f"  Constrained Prompts: {base_constrained['constrained']['harmful']}/{base_constrained['constrained']['total']} harmful ({base_constrained['constrained']['harmful_rate']:.1f}%)")
        
        # Show detailed breakdown by scenario
        scenarios = {}
        scenario_prompts = {}
        
        # Collect scenarios and their prompts from the results
        for result in all_results:
            if result.model_name == stats.model_name and result.prompt_type.startswith('scenario_'):
                parts = result.prompt_type.split('_')
                if len(parts) >= 3:
                    scenario_num = parts[1]
                    variant = parts[2]
                    if scenario_num not in scenarios:
                        scenarios[scenario_num] = {}
                        scenario_prompts[scenario_num] = {}
                    scenarios[scenario_num][variant] = stats.prompt_stats[result.prompt_type]
                    # Store the prompt text (use base version for display, removing constraint text)
                    if variant == 'base':
                        scenario_prompts[scenario_num]['prompt'] = result.prompt
                    elif variant == 'constrained' and 'prompt' not in scenario_prompts[scenario_num]:
                        # If we only have constrained, remove the constraint part for display
                        base_prompt = result.prompt.replace('. (no more than 50 words)', '')
                        scenario_prompts[scenario_num]['prompt'] = base_prompt
        
        if scenarios:
            print(f"\nDetailed Breakdown by Scenario:")
            for scenario_num in sorted(scenarios.keys(), key=int):
                # Truncate long prompts for readability, breaking at word boundaries
                prompt_text = scenario_prompts[scenario_num].get('prompt', 'Unknown prompt')
                if len(prompt_text) > 75:
                    # Find the last space before character 72 to avoid cutting words
                    truncate_pos = prompt_text.rfind(' ', 0, 72)
                    if truncate_pos == -1:  # No space found, just cut at 72
                        truncate_pos = 72
                    prompt_text = prompt_text[:truncate_pos] + "..."
                
                print(f"  Scenario {scenario_num}: \"{prompt_text}\"")
                for variant in ['base', 'constrained']:
                    if variant in scenarios[scenario_num]:
                        data = scenarios[scenario_num][variant]
                        harmful = data['harmful']
                        refused = data['refused'] 
                        total = harmful + refused
                        harmful_rate = (harmful / total * 100) if total > 0 else 0.0
                        print(f"    {variant.capitalize()}: {harmful}/{total} harmful ({harmful_rate:.1f}%)")
    
    def _print_comparison(self):
        """Print a comparison between models if multiple models were evaluated."""
        if len(self.model_stats) < 2:
            return
        
        print("\nMODEL COMPARISON")
        print("="*80)
        
        # Create comparison table
        models = list(self.model_stats.values())
        
        print(f"{'Model':<30} {'Overall':<10} {'Base':<10} {'Constrained':<12}")
        print("-" * 62)
        
        for stats in models:
            base_constrained = stats.get_base_vs_constrained_stats()
            print(f"{stats.model_name:<30} {stats.harmful_rate:<9.1f}% {base_constrained['base']['harmful_rate']:<9.1f}% {base_constrained['constrained']['harmful_rate']:<11.1f}%")
        
        # Scenario-by-scenario comparison if there are multiple scenarios
        scenarios = set()
        for model in models:
            for prompt_type in model.prompt_stats.keys():
                if prompt_type.startswith('scenario_'):
                    scenario_num = prompt_type.split('_')[1]
                    scenarios.add(scenario_num)
        
        if len(scenarios) > 1:
            print(f"\nSCENARIO COMPARISON (Base / Constrained)")
            print("-" * 80)
            
            scenarios_sorted = sorted(scenarios, key=int)
            header = f"{'Model':<30}"
            for scenario in scenarios_sorted:
                header += f" S{scenario}(B/C)     "
            print(header)
            print("-" * len(header))
            
            for stats in models:
                row = f"{stats.model_name:<30}"
                for scenario in scenarios_sorted:
                    base_key = f"scenario_{scenario}_base"
                    constrained_key = f"scenario_{scenario}_constrained"
                    base_rate = stats.get_prompt_harmful_rate(base_key) if base_key in stats.prompt_stats else 0.0
                    constrained_rate = stats.get_prompt_harmful_rate(constrained_key) if constrained_key in stats.prompt_stats else 0.0
                    row += f" {base_rate:3.0f}/{constrained_rate:<3.0f}%   "
                print(row)
        
        # Analysis
        print(f"\nANALYSIS:")
        self._print_analysis(models)
    
    def _print_analysis(self, models: List[ModelStats]):
        """Print analysis of the results."""
        # General analysis for any number of models
        print(f"• Evaluated {len(models)} model(s) across {len(self.results)} total evaluations")
        
        # Find most/least harmful models
        if len(models) > 1:
            most_harmful = max(models, key=lambda m: m.harmful_rate)
            least_harmful = min(models, key=lambda m: m.harmful_rate)
            print(f"• Most harmful: {most_harmful.model_name} ({most_harmful.harmful_rate:.1f}%)")
            print(f"• Least harmful: {least_harmful.model_name} ({least_harmful.harmful_rate:.1f}%)")
        
        # Base vs Constrained analysis
        print(f"\nBase vs Constrained Analysis:")
        for model in models:
            base_constrained = model.get_base_vs_constrained_stats()
            base_rate = base_constrained['base']['harmful_rate']
            constrained_rate = base_constrained['constrained']['harmful_rate']
            
            if base_constrained['base']['total'] > 0 and base_constrained['constrained']['total'] > 0:
                print(f"• {model.model_name}: Base {base_rate:.1f}% vs Constrained {constrained_rate:.1f}%")
                if constrained_rate > base_rate + 5:  # 5% threshold for significance
                    diff = constrained_rate - base_rate
                    print(f"  ⚠️  {diff:.1f}% higher harmful rate with word constraints")
                elif base_rate > constrained_rate + 5:
                    diff = base_rate - constrained_rate
                    print(f"  ✅ {diff:.1f}% lower harmful rate with word constraints")
        
        # RLVR vs DPO specific analysis if applicable
        rlvr_models = [m for m in models if "RLVR" in m.model_name]
        dpo_models = [m for m in models if "DPO" in m.model_name]
        
        if rlvr_models and dpo_models:
            print(f"\nRLVR vs DPO Analysis:")
            avg_rlvr = sum(m.harmful_rate for m in rlvr_models) / len(rlvr_models)
            avg_dpo = sum(m.harmful_rate for m in dpo_models) / len(dpo_models)
            print(f"• Average RLVR harmful rate: {avg_rlvr:.1f}%")
            print(f"• Average DPO harmful rate: {avg_dpo:.1f}%")
            
            # Base vs constrained comparison between RLVR and DPO
            rlvr_base_avg = sum(m.get_base_vs_constrained_stats()['base']['harmful_rate'] for m in rlvr_models) / len(rlvr_models)
            rlvr_constrained_avg = sum(m.get_base_vs_constrained_stats()['constrained']['harmful_rate'] for m in rlvr_models) / len(rlvr_models)
            dpo_base_avg = sum(m.get_base_vs_constrained_stats()['base']['harmful_rate'] for m in dpo_models) / len(dpo_models)
            dpo_constrained_avg = sum(m.get_base_vs_constrained_stats()['constrained']['harmful_rate'] for m in dpo_models) / len(dpo_models)
            
            print(f"• RLVR base vs constrained: {rlvr_base_avg:.1f}% vs {rlvr_constrained_avg:.1f}%")
            print(f"• DPO base vs constrained: {dpo_base_avg:.1f}% vs {dpo_constrained_avg:.1f}%")
            
            if avg_rlvr > avg_dpo:
                diff = avg_rlvr - avg_dpo
                print(f"⚠️  RLVR models show {diff:.1f}% higher average harmful rate than DPO")
            elif avg_dpo > avg_rlvr:
                diff = avg_dpo - avg_rlvr
                print(f"✅ RLVR models show {diff:.1f}% lower average harmful rate than DPO")
        
        # Check for concerning patterns across prompts
        if models:
            all_prompts = set()
            for model in models:
                all_prompts.update(model.prompt_stats.keys())
            
            if len(all_prompts) > 1:
                print(f"\nPrompt Variance Analysis:")
                for model in models:
                    rates = [model.get_prompt_harmful_rate(p) for p in all_prompts if model.get_prompt_total(p) > 0]
                    if len(rates) > 1:
                        min_rate = min(rates)
                        max_rate = max(rates)
                        variance = max_rate - min_rate
                        print(f"• {model.model_name}: {variance:.1f}% variance across prompts ({min_rate:.1f}% - {max_rate:.1f}%)")
                        if variance > 20:
                            print(f"  ⚠️  High variance detected - model may be inconsistent")
    
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
                    "prompt_stats": stats.prompt_stats,
                    "prompt_harmful_rates": {
                        prompt_type: stats.get_prompt_harmful_rate(prompt_type)
                        for prompt_type in stats.prompt_stats.keys()
                    }
                }
                for model_name, stats in self.model_stats.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nDetailed results saved to: {filename}")
