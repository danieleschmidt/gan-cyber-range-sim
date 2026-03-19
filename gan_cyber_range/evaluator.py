"""FlowEvaluator: measures statistical similarity between generated and real distributions."""

from __future__ import annotations

import warnings
from typing import List, Dict

import numpy as np

from .flows import NetworkFlow, AttackType


def _feature_matrix(flows: List[NetworkFlow]) -> np.ndarray:
    """Convert flow list → (N, 7) float32 matrix."""
    return np.stack([f.to_feature_vector() for f in flows], axis=0)


def _kl_divergence(p: np.ndarray, q: np.ndarray, bins: int = 20) -> float:
    """KL divergence D(P || Q) via histogram estimation."""
    lo = min(p.min(), q.min())
    hi = max(p.max(), q.max())
    if hi == lo:
        return 0.0

    edges = np.linspace(lo, hi, bins + 1)
    p_hist, _ = np.histogram(p, bins=edges, density=True)
    q_hist, _ = np.histogram(q, bins=edges, density=True)

    # Add epsilon to avoid log(0)
    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps

    # Normalise to probabilities
    p_prob = p_hist / p_hist.sum()
    q_prob = q_hist / q_hist.sum()

    return float(np.sum(p_prob * np.log(p_prob / q_prob)))


FEATURE_NAMES = ["port", "protocol", "bytes", "duration", "flags", "src_ip", "dst_ip"]


class FlowEvaluator:
    """Measures statistical similarity between generated and reference flows.

    Metrics
    -------
    - Per-feature KL divergence (generated vs reference distributions)
    - Mean absolute difference of feature means
    - Mean absolute difference of feature stds

    Lower KL divergence → generated flows are more realistic.
    """

    def __init__(self, bins: int = 30):
        self.bins = bins

    def evaluate(
        self,
        generated: List[NetworkFlow],
        reference: List[NetworkFlow],
    ) -> Dict:
        """Compare generated flows against reference flows.

        Args:
            generated: list of GAN-produced flows
            reference:  list of real or rule-based flows to compare against

        Returns:
            Dict with per-feature KL divergences, mean KL, and summary stats.
        """
        if not generated or not reference:
            raise ValueError("Both generated and reference flow lists must be non-empty.")

        gen_mat = _feature_matrix(generated)   # (N_gen, 7)
        ref_mat = _feature_matrix(reference)   # (N_ref, 7)

        kl_per_feature: Dict[str, float] = {}
        mean_diff: Dict[str, float] = {}
        std_diff: Dict[str, float] = {}

        for i, name in enumerate(FEATURE_NAMES):
            g_col = gen_mat[:, i]
            r_col = ref_mat[:, i]
            kl_per_feature[name] = _kl_divergence(g_col, r_col, self.bins)
            mean_diff[name] = abs(g_col.mean() - r_col.mean())
            std_diff[name] = abs(g_col.std() - r_col.std())

        mean_kl = float(np.mean(list(kl_per_feature.values())))

        return {
            "kl_per_feature": kl_per_feature,
            "mean_kl": mean_kl,
            "mean_abs_mean_diff": float(np.mean(list(mean_diff.values()))),
            "mean_abs_std_diff": float(np.mean(list(std_diff.values()))),
            "n_generated": len(generated),
            "n_reference": len(reference),
        }

    def evaluate_all_types(
        self,
        generated_by_type: Dict[str, List[NetworkFlow]],
        reference_by_type: Dict[str, List[NetworkFlow]],
    ) -> Dict[str, Dict]:
        """Run evaluate() for each attack type."""
        results = {}
        for name in generated_by_type:
            if name not in reference_by_type:
                warnings.warn(f"No reference for type {name!r}, skipping.")
                continue
            results[name] = self.evaluate(
                generated_by_type[name],
                reference_by_type[name],
            )
        return results

    @staticmethod
    def summarize(results: Dict) -> str:
        """Pretty-print evaluation results."""
        lines = []

        if "kl_per_feature" in results:
            # Single-type result
            lines.append(f"  n_generated : {results['n_generated']}")
            lines.append(f"  n_reference : {results['n_reference']}")
            lines.append(f"  mean KL div : {results['mean_kl']:.4f}")
            lines.append(f"  mean |Δmean|: {results['mean_abs_mean_diff']:.4f}")
            lines.append(f"  mean |Δstd| : {results['mean_abs_std_diff']:.4f}")
            lines.append("  KL per feature:")
            for feat, kl in results["kl_per_feature"].items():
                lines.append(f"    {feat:12s}: {kl:.4f}")
        else:
            # Multi-type result
            for attack_name, r in results.items():
                lines.append(f"\n  [{attack_name}]")
                lines.append(f"    mean KL: {r['mean_kl']:.4f} | "
                              f"|Δmean|: {r['mean_abs_mean_diff']:.4f} | "
                              f"|Δstd|: {r['mean_abs_std_diff']:.4f}")

        return "\n".join(lines)
