"""
Statistics collection and reporting for safety evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import os
import logging
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    PLOTTING_AVAILABLE = True
except ImportError as e:  # pragma: no cover - plotting is optional
    PLOTTING_AVAILABLE = False
    logging.getLogger(__name__).debug(f"Plotting libraries not available: {e}")

from .prompt_library import PromptSet

logger = logging.getLogger(__name__)

_BOOTSTRAP_RESAMPLES = 1000
_BOOTSTRAP_ALPHA = 0.05
_BOOTSTRAP_RNG = np.random.default_rng(123456789)


def _bootstrap_proportion_ci(
    outcomes: List[int],
    *,
    alpha: float = _BOOTSTRAP_ALPHA,
    n_resamples: int = _BOOTSTRAP_RESAMPLES,
) -> Optional[Tuple[float, float]]:
    if not outcomes:
        return None

    arr = np.asarray(outcomes, dtype=float)
    mean = float(arr.mean())

    if arr.size <= 1 or mean in (0.0, 1.0) or n_resamples <= 0:
        percentage = mean * 100
        return percentage, percentage

    resamples = _BOOTSTRAP_RNG.choice(
        arr, size=(n_resamples, arr.size), replace=True
    ).mean(axis=1)
    lower = float(np.quantile(resamples, alpha / 2)) * 100
    upper = float(np.quantile(resamples, 1 - alpha / 2)) * 100
    return lower, upper


@dataclass
class EvaluationResult:
    """Single evaluation result."""

    model_name: str
    prompt: str
    response: str
    judgment: str  # "harmful" or "refused"
    scenario_id: str
    scenario_title: str
    scenario_display_prompt: str
    variant_id: str
    variant_label: str
    variant_type: str
    toxicity_score: float = 0.0  # 0-100 toxicity score
    compliance_judgment: Optional[str] = None  # "compliant" or "non_compliant"


@dataclass
class VariantAggregate:
    """Aggregate statistics for a specific variant."""

    harmful: int = 0
    refused: int = 0
    compliant: int = 0
    non_compliant: int = 0

    def record(self, judgment: str, compliance_judgment: Optional[str]):
        if judgment == "harmful":
            self.harmful += 1
        else:
            self.refused += 1

        if compliance_judgment:
            if compliance_judgment == "compliant":
                self.compliant += 1
            else:
                self.non_compliant += 1

    @property
    def total(self) -> int:
        return self.harmful + self.refused

    @property
    def harmful_rate(self) -> float:
        return (self.harmful / self.total * 100) if self.total > 0 else 0.0

    @property
    def compliance_total(self) -> int:
        return self.compliant + self.non_compliant

    @property
    def compliance_rate(self) -> float:
        return (self.compliant / self.compliance_total * 100) if self.compliance_total > 0 else 0.0


@dataclass
class ScenarioStats:
    """Statistics for a scenario, broken down per variant."""

    scenario_id: str
    title: str
    display_prompt: str
    variant_order: List[str] = field(default_factory=list)
    variants: Dict[str, VariantAggregate] = field(default_factory=dict)
    variant_labels: Dict[str, str] = field(default_factory=dict)
    variant_types: Dict[str, str] = field(default_factory=dict)

    def ensure_variant(self, variant_id: str) -> VariantAggregate:
        if variant_id not in self.variants:
            self.variants[variant_id] = VariantAggregate()
            self.variant_order.append(variant_id)
        return self.variants[variant_id]


@dataclass
class ModelStats:
    """Statistics for a single model."""

    model_name: str
    total_evaluations: int = 0
    harmful_responses: int = 0
    refused_responses: int = 0
    total_toxicity: float = 0.0
    compliance_evaluations: int = 0
    compliant_responses: int = 0
    non_compliant_responses: int = 0
    scenario_stats: Dict[str, ScenarioStats] = field(default_factory=dict)
    scenario_order: List[str] = field(default_factory=list)
    variant_type_stats: Dict[str, VariantAggregate] = field(default_factory=dict)
    harmful_ci: Optional[Tuple[float, float]] = None
    compliance_ci: Optional[Tuple[float, float]] = None
    variant_cis: Dict[Tuple[str, str, str], Optional[Tuple[float, float]]] = field(
        default_factory=dict
    )
    variant_type_cis: Dict[Tuple[str, str], Optional[Tuple[float, float]]] = field(
        default_factory=dict
    )

    def record(self, result: EvaluationResult):
        self.total_evaluations += 1

        if result.judgment == "harmful":
            self.harmful_responses += 1
        else:
            self.refused_responses += 1

        self.total_toxicity += result.toxicity_score

        if result.compliance_judgment:
            self.compliance_evaluations += 1
            if result.compliance_judgment == "compliant":
                self.compliant_responses += 1
            else:
                self.non_compliant_responses += 1

        scenario = self._ensure_scenario(result)
        variant_stats = scenario.ensure_variant(result.variant_id)
        scenario.variant_labels[result.variant_id] = result.variant_label
        scenario.variant_types[result.variant_id] = result.variant_type
        variant_stats.record(result.judgment, result.compliance_judgment)

        variant_type_stats = self.variant_type_stats.setdefault(
            result.variant_type,
            VariantAggregate(),
        )
        variant_type_stats.record(result.judgment, result.compliance_judgment)

    def _ensure_scenario(self, result: EvaluationResult) -> ScenarioStats:
        if result.scenario_id not in self.scenario_stats:
            self.scenario_stats[result.scenario_id] = ScenarioStats(
                scenario_id=result.scenario_id,
                title=result.scenario_title,
                display_prompt=result.scenario_display_prompt,
            )
            self.scenario_order.append(result.scenario_id)
        return self.scenario_stats[result.scenario_id]

    @property
    def harmful_rate(self) -> float:
        return (self.harmful_responses / self.total_evaluations * 100) if self.total_evaluations else 0.0

    @property
    def refusal_rate(self) -> float:
        return (self.refused_responses / self.total_evaluations * 100) if self.total_evaluations else 0.0

    @property
    def compliance_rate(self) -> float:
        return (self.compliant_responses / self.compliance_evaluations * 100) if self.compliance_evaluations else 0.0

    @property
    def non_compliance_rate(self) -> float:
        return (
            self.non_compliant_responses / self.compliance_evaluations * 100
        ) if self.compliance_evaluations else 0.0

    @property
    def average_toxicity(self) -> float:
        return (self.total_toxicity / self.total_evaluations) if self.total_evaluations else 0.0

    @property
    def base_harmful_rate(self) -> float:
        stats = self.variant_type_stats.get("base")
        return stats.harmful_rate if stats else 0.0

    @property
    def constrained_harmful_rate(self) -> float:
        stats = self.variant_type_stats.get("base_plus_distractor")
        return stats.harmful_rate if stats else 0.0

    def get_variant_type_summary(self, variant_type: str) -> Dict[str, float]:
        stats = self.variant_type_stats.get(variant_type)
        if not stats:
            return {"harmful": 0, "refused": 0, "total": 0, "harmful_rate": 0.0}
        return {
            "harmful": stats.harmful,
            "refused": stats.refused,
            "total": stats.total,
            "harmful_rate": stats.harmful_rate,
        }

    def get_variant_type_compliance_summary(self, variant_type: str) -> Dict[str, float]:
        stats = self.variant_type_stats.get(variant_type)
        if not stats or stats.compliance_total == 0:
            return {
                "compliant": 0,
                "non_compliant": 0,
                "total": 0,
                "compliance_rate": 0.0,
            }
        return {
            "compliant": stats.compliant,
            "non_compliant": stats.non_compliant,
            "total": stats.compliance_total,
            "compliance_rate": stats.compliance_rate,
        }


class StatisticsCollector:
    """Collects and manages evaluation statistics."""

    def __init__(self, prompt_set: PromptSet):
        self.prompt_set = prompt_set
        self.results: List[EvaluationResult] = []
        self.model_stats: Dict[str, ModelStats] = {}

    def add_result(self, result: EvaluationResult):
        """Add an evaluation result."""
        self.results.append(result)
        stats = self.model_stats.setdefault(result.model_name, ModelStats(model_name=result.model_name))
        stats.record(result)

    def compute_confidence_intervals(
        self,
        *,
        alpha: float = _BOOTSTRAP_ALPHA,
        n_resamples: int = _BOOTSTRAP_RESAMPLES,
    ) -> None:
        for model_name, stats in self.model_stats.items():
            model_results = [
                result for result in self.results if result.model_name == model_name
            ]

            harmful_outcomes = [
                1 if result.judgment == "harmful" else 0 for result in model_results
            ]
            stats.harmful_ci = _bootstrap_proportion_ci(
                harmful_outcomes, alpha=alpha, n_resamples=n_resamples
            )

            compliance_outcomes = [
                1 if result.compliance_judgment == "compliant" else 0
                for result in model_results
                if result.compliance_judgment is not None
            ]
            stats.compliance_ci = _bootstrap_proportion_ci(
                compliance_outcomes, alpha=alpha, n_resamples=n_resamples
            )

            stats.variant_cis.clear()
            stats.variant_type_cis.clear()

            for scenario_id, scenario in stats.scenario_stats.items():
                for variant_id in scenario.variants:
                    variant_results = [
                        result
                        for result in model_results
                        if result.scenario_id == scenario_id
                        and result.variant_id == variant_id
                    ]

                    harmful_ci = _bootstrap_proportion_ci(
                        [
                            1 if res.judgment == "harmful" else 0
                            for res in variant_results
                        ],
                        alpha=alpha,
                        n_resamples=n_resamples,
                    )
                    stats.variant_cis[(scenario_id, variant_id, "harmful")] = (
                        harmful_ci
                    )

                    compliance_ci = _bootstrap_proportion_ci(
                        [
                            1 if res.compliance_judgment == "compliant" else 0
                            for res in variant_results
                            if res.compliance_judgment is not None
                        ],
                        alpha=alpha,
                        n_resamples=n_resamples,
                    )
                    stats.variant_cis[(scenario_id, variant_id, "compliance")] = (
                        compliance_ci
                    )

            for variant_type in stats.variant_type_stats:
                vt_results = [
                    result
                    for result in model_results
                    if result.variant_type == variant_type
                ]
                stats.variant_type_cis[(variant_type, "harmful")] = _bootstrap_proportion_ci(
                    [1 if res.judgment == "harmful" else 0 for res in vt_results],
                    alpha=alpha,
                    n_resamples=n_resamples,
                )
                stats.variant_type_cis[(variant_type, "compliance")] = _bootstrap_proportion_ci(
                    [
                        1 if res.compliance_judgment == "compliant" else 0
                        for res in vt_results
                        if res.compliance_judgment is not None
                    ],
                    alpha=alpha,
                    n_resamples=n_resamples,
                )

    def print_summary(self):
        """Print a comprehensive summary of results."""
        print("\n" + "=" * 80)
        print("SAFETY EVALUATION RESULTS")
        print("=" * 80)

        for stats in self.model_stats.values():
            self._print_model_stats(stats)
            print("-" * 80)

        self._print_comparison()

    def _print_model_stats(self, stats: ModelStats):
        print(f"\nModel: {stats.model_name}")
        print(f"Total Evaluations: {stats.total_evaluations}")
        print(
            f"Overall Harmful Rate: {stats.harmful_rate:.1f}% "
            f"({stats.harmful_responses}/{stats.total_evaluations})"
            + (
                f" [95% CI {stats.harmful_ci[0]:.1f}%, {stats.harmful_ci[1]:.1f}%]"
                if stats.harmful_ci
                else ""
            )
        )
        print(
            f"Overall Refusal Rate: {stats.refusal_rate:.1f}% "
            f"({stats.refused_responses}/{stats.total_evaluations})"
        )
        print(f"Average Toxicity Score: {stats.average_toxicity:.1f}")
        if stats.compliance_evaluations > 0:
            print(
                f"Compliance Rate (provided dialogue): {stats.compliance_rate:.1f}% "
                f"({stats.compliant_responses}/{stats.compliance_evaluations})"
                + (
                    f" [95% CI {stats.compliance_ci[0]:.1f}%, {stats.compliance_ci[1]:.1f}%]"
                    if stats.compliance_ci
                    else ""
                )
            )
            print(
                f"Non-compliance Rate: {stats.non_compliance_rate:.1f}% "
                f"({stats.non_compliant_responses}/{stats.compliance_evaluations})"
            )

        base_summary = stats.get_variant_type_summary("base")
        distractor_summary = stats.get_variant_type_summary("base_plus_distractor")
        if base_summary["total"] > 0 or distractor_summary["total"] > 0:
            print("\nBase vs Base + Distractor Summary:")
            if base_summary["total"] > 0:
                print(
                    f"  Base: {base_summary['harmful']}/{base_summary['total']} "
                    f"({base_summary['harmful_rate']:.1f}%)"
                )
            if distractor_summary["total"] > 0:
                print(
                    f"  Base + Distractor: {distractor_summary['harmful']}/"
                    f"{distractor_summary['total']} ({distractor_summary['harmful_rate']:.1f}%)"
                )

        if stats.scenario_order:
            print("\nDetailed Breakdown by Scenario:")
            for scenario_id in stats.scenario_order:
                scenario = stats.scenario_stats[scenario_id]
                prompt_preview = _truncate_prompt(scenario.display_prompt)
                print(f"  {scenario.title}: \"{prompt_preview}\"")
                for variant_id in scenario.variant_order:
                    variant_stats = scenario.variants[variant_id]
                    label = scenario.variant_labels.get(variant_id, variant_id)
                    total = variant_stats.total
                    harmful_rate = variant_stats.harmful_rate
                    print(
                        f"    {label}: {variant_stats.harmful}/{total} ({harmful_rate:.1f}%) harmful"
                    )
                    if variant_stats.compliance_total > 0:
                        comp_rate = variant_stats.compliance_rate
                        print(
                            f"      Provided dialogue: {variant_stats.compliant}/"
                            f"{variant_stats.compliance_total} ({comp_rate:.1f}%)"
                        )

    def _print_comparison(self):
        if len(self.model_stats) < 2:
            return

        print("\nMODEL COMPARISON")
        print("=" * 80)

        models = list(self.model_stats.values())
        has_compliance = any(model.compliance_evaluations > 0 for model in models)

        header = f"{'Model':<30} {'Overall':<10} {'Base':<10} {'Base+Distr':<12}"
        if has_compliance:
            header += f" {'Compliance':<12}"
        print(header)
        print("-" * len(header))

        for stats in models:
            row = (
                f"{stats.model_name:<30} {stats.harmful_rate:<9.1f}% "
                f"{stats.base_harmful_rate:<9.1f}% "
                f"{stats.constrained_harmful_rate:<11.1f}%"
            )
            if has_compliance:
                value = (
                    f"{stats.compliance_rate:.1f}%"
                    if stats.compliance_evaluations
                    else "N/A"
                )
                row += f" {value:<12}"
            print(row)

        scenario_ids = self._collect_scenario_order(models)
        if scenario_ids:
            print("\nSCENARIO COMPARISON (Harmful %)")
            print("-" * 80)
            header = f"{'Model':<30}"
            for scenario_id in scenario_ids:
                title = self._scenario_title_from_models(models, scenario_id)
                header += f" {title[:12]:<12}"
            print(header)
            print("-" * len(header))

            for stats in models:
                row = f"{stats.model_name:<30}"
                for scenario_id in scenario_ids:
                    scenario = stats.scenario_stats.get(scenario_id)
                    if not scenario:
                        row += f" {'-':<12}"
                        continue
                    base_variant = _find_variant_by_type(scenario, "base")
                    distractor_variant = _find_variant_by_type(
                        scenario, "base_plus_distractor"
                    )
                    if base_variant and distractor_variant:
                        row += (
                            f" {base_variant.harmful_rate:3.0f}/"
                            f"{distractor_variant.harmful_rate:<3.0f}%"
                        )
                    elif base_variant:
                        row += f" {base_variant.harmful_rate:>6.1f}%"
                    elif distractor_variant:
                        row += f" {distractor_variant.harmful_rate:>6.1f}%"
                    else:
                        row += f" {'-':<12}"
                print(row)

        print("\nANALYSIS:")
        self._print_analysis(models)

    def _collect_scenario_order(self, models: List[ModelStats]) -> List[str]:
        order = [scenario.scenario_id for scenario in self.prompt_set.scenarios]
        for stats in models:
            for scenario_id in stats.scenario_order:
                if scenario_id not in order:
                    order.append(scenario_id)
        return order

    @staticmethod
    def _scenario_title_from_models(models: List[ModelStats], scenario_id: str) -> str:
        for stats in models:
            scenario = stats.scenario_stats.get(scenario_id)
            if scenario:
                return scenario.title
        return scenario_id

    def _print_analysis(self, models: List[ModelStats]):
        print(f"• Evaluated {len(models)} model(s) across {len(self.results)} total evaluations")

        if len(models) > 1:
            most_harmful = max(models, key=lambda m: m.harmful_rate)
            least_harmful = min(models, key=lambda m: m.harmful_rate)
            print(f"• Most harmful: {most_harmful.model_name} ({most_harmful.harmful_rate:.1f}%)")
            print(f"• Least harmful: {least_harmful.model_name} ({least_harmful.harmful_rate:.1f}%)")

        evaluated_with_compliance = [m for m in models if m.compliance_evaluations > 0]
        if evaluated_with_compliance:
            most_compliant = max(evaluated_with_compliance, key=lambda m: m.compliance_rate)
            least_compliant = min(evaluated_with_compliance, key=lambda m: m.compliance_rate)
            print(
                f"• Highest provided-dialogue rate: {most_compliant.model_name} "
                f"({most_compliant.compliance_rate:.1f}%)"
            )
            print(
                f"• Lowest provided-dialogue rate: {least_compliant.model_name} "
                f"({least_compliant.compliance_rate:.1f}%)"
            )

        for stats in models:
            base_rate = stats.base_harmful_rate
            distractor_rate = stats.constrained_harmful_rate
            if base_rate or distractor_rate:
                diff = distractor_rate - base_rate
                sign = "higher" if diff > 0 else "lower"
                print(
                    f"• {stats.model_name}: Base {base_rate:.1f}% vs Distractor {distractor_rate:.1f}% "
                    f"(Δ {diff:+.1f} points {sign})"
                )

        rlvr_models = [m for m in models if "RLVR" in m.model_name.upper()]
        dpo_models = [m for m in models if "DPO" in m.model_name.upper()]
        if rlvr_models and dpo_models:
            avg_rlvr = sum(m.harmful_rate for m in rlvr_models) / len(rlvr_models)
            avg_dpo = sum(m.harmful_rate for m in dpo_models) / len(dpo_models)
            print(
                f"• RLVR average harmful rate: {avg_rlvr:.1f}% | "
                f"DPO average harmful rate: {avg_dpo:.1f}%"
            )
            rlvr_compliance = [m.compliance_rate for m in rlvr_models if m.compliance_evaluations]
            dpo_compliance = [m.compliance_rate for m in dpo_models if m.compliance_evaluations]
            if rlvr_compliance and dpo_compliance:
                print(
                    f"• RLVR provided-dialogue rate: {sum(rlvr_compliance)/len(rlvr_compliance):.1f}% | "
                    f"DPO: {sum(dpo_compliance)/len(dpo_compliance):.1f}%"
                )

        for stats in models:
            rates = [
                variant.harmful_rate
                for scenario in stats.scenario_stats.values()
                for variant in scenario.variants.values()
                if variant.total > 0
            ]
            if len(rates) > 1:
                variance = max(rates) - min(rates)
                print(
                    f"• {stats.model_name}: {variance:.1f}% harmful-rate spread across prompts "
                    f"({min(rates):.1f}% - {max(rates):.1f}%)"
                )

    def save_results(self, filename: str):
        """Save detailed results to a JSON file."""
        data = {
            "results": [
                {
                    "model_name": r.model_name,
                    "prompt": r.prompt,
                    "response": r.response,
                    "judgment": r.judgment,
                    "toxicity_score": r.toxicity_score,
                    "compliance_judgment": r.compliance_judgment,
                    "scenario_id": r.scenario_id,
                    "scenario_title": r.scenario_title,
                    "variant_id": r.variant_id,
                    "variant_label": r.variant_label,
                    "variant_type": r.variant_type,
                }
                for r in self.results
            ],
            "statistics": {
                model_name: _serialise_model_stats(stats)
                for model_name, stats in self.model_stats.items()
            },
        }

        with open(filename, "w") as file:
            json.dump(data, file, indent=2)

        print(f"\nDetailed results saved to: {filename}")

    def generate_plots(self, output_dir: str, show_plots: bool = False):
        """Generate visualization plots for the evaluation results."""
        if not PLOTTING_AVAILABLE:
            print("⚠️  Plotting libraries not available. Install matplotlib, seaborn, pandas to enable plots.")
            return

        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        compliance_available = any(
            stats.compliance_evaluations > 0 for stats in self.model_stats.values()
        )

        for model_name, stats in self.model_stats.items():
            self._create_model_plot(
                model_name, stats, plots_dir, show_plots, metric="harmful"
            )
            if compliance_available:
                self._create_model_plot(
                    model_name, stats, plots_dir, show_plots, metric="compliance"
                )

        if len(self.model_stats) > 1:
            self._create_comparison_plot(plots_dir, show_plots, metric="harmful")
            if compliance_available:
                self._create_comparison_plot(
                    plots_dir, show_plots, metric="compliance"
                )

    def _create_model_plot(
        self,
        model_name: str,
        stats: ModelStats,
        plots_dir: str,
        show_plots: bool,
        metric: str,
    ):
        scenario_order = [
            s.scenario_id
            for s in self.prompt_set.scenarios
            if s.scenario_id in stats.scenario_stats
        ]
        if not scenario_order:
            return

        plot_rows = []
        variant_labels = [
            self.prompt_set.variant_type_label(vt) for vt in self.prompt_set.variant_types
        ]
        scenario_labels = []
        for scenario_id in scenario_order:
            scenario = stats.scenario_stats[scenario_id]
            scenario_labels.append(scenario.title)
            for variant_id in scenario.variant_order:
                variant_stats = scenario.variants[variant_id]
                label = scenario.variant_labels.get(variant_id, variant_id)
                if metric == "harmful":
                    value = variant_stats.harmful_rate
                elif metric == "compliance":
                    if variant_stats.compliance_total == 0:
                        continue
                    value = variant_stats.compliance_rate
                else:
                    logger.warning("Unknown metric '%s' for plotting", metric)
                    continue

                if metric == "harmful" and variant_stats.total == 0:
                    continue

                ci = stats.variant_cis.get(
                    (scenario.scenario_id, variant_id, metric)
                )
                lower = ci[0] if ci else value
                upper = ci[1] if ci else value

                plot_rows.append(
                    {
                        "Scenario": scenario.title,
                        "Variant": label,
                        "Metric Value": value,
                        "Metric Lower": lower,
                        "Metric Upper": upper,
                    }
                )

        if not plot_rows:
            return

        df = pd.DataFrame(plot_rows)
        df = df.reset_index(drop=True)
        plt.style.use("default")
        sns.set_palette("husl")
        hue_order = [
            label for label in variant_labels if label in df["Variant"].unique()
        ]
        order = [
            label for label in scenario_labels if label in df["Scenario"].unique()
        ]

        fig, ax = plt.subplots(figsize=(12, 6))
        bar_plot = sns.barplot(
            data=df,
            x="Scenario",
            y="Metric Value",
            hue="Variant",
            hue_order=hue_order,
            order=order,
            ax=ax,
            width=0.8,
            errorbar=None,
        )

        if metric == "harmful":
            title = "Safety Evaluation Results"
            y_label = "Harmful Response Rate (%)"
        else:
            title = "Compliance Evaluation Results"
            y_label = "Provided Dialogue Rate (%)"

        ax.set_title(f"{title}: {model_name}", fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Prompt Scenarios", fontsize=12, fontweight="bold")
        ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 100)

        for container in bar_plot.containers:
            bar_plot.bar_label(container, fmt="%.1f%%", fontsize=9, padding=3)

        for patch, (_, row) in zip(bar_plot.patches, df.iterrows()):
            err_low = max(0.0, row["Metric Value"] - row["Metric Lower"])
            err_high = max(0.0, row["Metric Upper"] - row["Metric Value"])
            if err_low == 0 and err_high == 0:
                continue
            ax.errorbar(
                patch.get_x() + patch.get_width() / 2,
                row["Metric Value"],
                yerr=[[err_low], [err_high]],
                fmt="none",
                ecolor="black",
                elinewidth=1,
                capsize=4,
            )

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            title="Prompt Variant",
            title_fontsize=12,
            fontsize=11,
            loc="upper right",
        )
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        safe_model_name = model_name.replace("/", "_").replace(" ", "_")
        suffix = "safety" if metric == "harmful" else "compliance"
        plot_path = os.path.join(
            plots_dir, f"{safe_model_name}_{suffix}_results.png"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"📊 Plot saved: {plot_path}")

        if show_plots:
            plt.show()
        else:
            plt.close()

    def _create_comparison_plot(
        self,
        plots_dir: str,
        show_plots: bool,
        metric: str,
    ):
        plot_data = []
        base_label = self.prompt_set.variant_type_label("base")
        distractor_label = self.prompt_set.variant_type_label("base_plus_distractor")

        for model_name, stats in self.model_stats.items():
            if metric == "harmful":
                base_summary = stats.get_variant_type_summary("base")
                distractor_summary = stats.get_variant_type_summary(
                    "base_plus_distractor"
                )
                metric_key = "harmful_rate"
                metric_label = "Harmful Rate (%)"
                ci_key = "harmful"
            else:
                base_summary = stats.get_variant_type_compliance_summary("base")
                distractor_summary = stats.get_variant_type_compliance_summary(
                    "base_plus_distractor"
                )
                metric_key = "compliance_rate"
                metric_label = "Provided Dialogue Rate (%)"
                ci_key = "compliance"

            if base_summary["total"] > 0:
                base_ci = stats.variant_type_cis.get(("base", ci_key))
                plot_data.append(
                    {
                        "Model": model_name,
                        "Prompt Type": base_label,
                        "Metric Value": base_summary[metric_key],
                        "Metric Lower": base_ci[0] if base_ci else base_summary[metric_key],
                        "Metric Upper": base_ci[1] if base_ci else base_summary[metric_key],
                    }
                )
            if distractor_summary["total"] > 0:
                distractor_ci = stats.variant_type_cis.get(("base_plus_distractor", ci_key))
                plot_data.append(
                    {
                        "Model": model_name,
                        "Prompt Type": distractor_label,
                        "Metric Value": distractor_summary[metric_key],
                        "Metric Lower": distractor_ci[0] if distractor_ci else distractor_summary[metric_key],
                        "Metric Upper": distractor_ci[1] if distractor_ci else distractor_summary[metric_key],
                    }
                )

        if not plot_data:
            return

        df = pd.DataFrame(plot_data)
        df = df.reset_index(drop=True)
        hue_order = [
            label for label in [base_label, distractor_label]
            if label in df["Prompt Type"].unique()
        ]
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_plot = sns.barplot(
            data=df,
            x="Model",
            y="Metric Value",
            hue="Prompt Type",
            hue_order=hue_order,
            ax=ax,
            width=0.7,
            errorbar=None,
        )

        title = (
            "Model Comparison: Base vs Base + Distractor"
            if metric == "harmful"
            else "Model Compliance: Base vs Base + Distractor"
        )
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Models", fontsize=12, fontweight="bold")
        ax.set_ylabel(metric_label, fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(0, df["Metric Value"].max()) * 1.1 or 10)

        for container in bar_plot.containers:
            bar_plot.bar_label(container, fmt="%.1f", fontsize=10, padding=3)

        for patch, (_, row) in zip(bar_plot.patches, df.iterrows()):
            err_low = max(0.0, row["Metric Value"] - row["Metric Lower"])
            err_high = max(0.0, row["Metric Upper"] - row["Metric Value"])
            if err_low == 0 and err_high == 0:
                continue
            ax.errorbar(
                patch.get_x() + patch.get_width() / 2,
                row["Metric Value"],
                yerr=[[err_low], [err_high]],
                fmt="none",
                ecolor="black",
                elinewidth=1,
                capsize=4,
            )

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title="Metric", title_fontsize=12, fontsize=11)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        suffix = "harmful" if metric == "harmful" else "compliance"
        plot_path = os.path.join(plots_dir, f"model_comparison_{suffix}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"📊 Comparison plot saved: {plot_path}")

        if show_plots:
            plt.show()
        else:
            plt.close()


def _truncate_prompt(text: str, max_length: int = 75) -> str:
    if len(text) <= max_length:
        return text
    cutoff = text.rfind(" ", 0, max_length - 3)
    if cutoff == -1:
        cutoff = max_length - 3
    return text[:cutoff] + "..."


def _find_variant_by_type(scenario: ScenarioStats, variant_type: str) -> Optional[VariantAggregate]:
    for variant_id, variant_stats in scenario.variants.items():
        if scenario.variant_types.get(variant_id) == variant_type:
            return variant_stats
    return None


def _serialise_model_stats(stats: ModelStats) -> Dict:
    return {
        "total_evaluations": stats.total_evaluations,
        "harmful_responses": stats.harmful_responses,
        "refused_responses": stats.refused_responses,
        "harmful_rate": stats.harmful_rate,
        "refusal_rate": stats.refusal_rate,
        "harmful_ci": stats.harmful_ci,
        "average_toxicity": stats.average_toxicity,
        "compliance_evaluations": stats.compliance_evaluations,
        "compliant_responses": stats.compliant_responses,
        "non_compliant_responses": stats.non_compliant_responses,
        "compliance_rate": stats.compliance_rate,
        "non_compliance_rate": stats.non_compliance_rate,
        "compliance_ci": stats.compliance_ci,
        "variant_type_stats": {
            variant_type: {
                "harmful": aggregate.harmful,
                "refused": aggregate.refused,
                "total": aggregate.total,
                "harmful_rate": aggregate.harmful_rate,
                "compliant": aggregate.compliant,
                "non_compliant": aggregate.non_compliant,
                "compliance_total": aggregate.compliance_total,
                "compliance_rate": aggregate.compliance_rate,
                "harmful_ci": stats.variant_type_cis.get((variant_type, "harmful")),
                "compliance_ci": stats.variant_type_cis.get((variant_type, "compliance")),
            }
            for variant_type, aggregate in stats.variant_type_stats.items()
        },
        "scenarios": {
            scenario_id: {
                "title": scenario_stats.title,
                "display_prompt": scenario_stats.display_prompt,
                "variants": {
                    variant_id: {
                        "label": scenario_stats.variant_labels.get(variant_id, variant_id),
                        "variant_type": scenario_stats.variant_types.get(variant_id, variant_id),
                        "harmful": variant_stats.harmful,
                        "refused": variant_stats.refused,
                        "total": variant_stats.total,
                        "harmful_rate": variant_stats.harmful_rate,
                        "compliant": variant_stats.compliant,
                        "non_compliant": variant_stats.non_compliant,
                        "compliance_total": variant_stats.compliance_total,
                        "compliance_rate": variant_stats.compliance_rate,
                        "harmful_ci": stats.variant_cis.get(
                            (scenario_id, variant_id, "harmful")
                        ),
                        "compliance_ci": stats.variant_cis.get(
                            (scenario_id, variant_id, "compliance")
                        ),
                    }
                    for variant_id, variant_stats in scenario_stats.variants.items()
                },
            }
            for scenario_id, scenario_stats in stats.scenario_stats.items()
        },
    }
__all__ = [
    "EvaluationResult",
    "StatisticsCollector",
]
