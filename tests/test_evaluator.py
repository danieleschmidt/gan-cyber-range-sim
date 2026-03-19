"""Tests for FlowEvaluator."""

import pytest
from gan_cyber_range.flows import NetworkFlowGenerator, AttackType
from gan_cyber_range.evaluator import FlowEvaluator, _kl_divergence
import numpy as np


@pytest.fixture
def gen():
    return NetworkFlowGenerator(seed=42)


@pytest.fixture
def evaluator():
    return FlowEvaluator(bins=20)


class TestKLDivergence:
    def test_identical_distributions_near_zero(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 500)
        kl = _kl_divergence(data, data)
        assert kl < 0.05

    def test_different_distributions_nonzero(self):
        rng = np.random.default_rng(1)
        p = rng.normal(0, 1, 500)
        q = rng.normal(5, 1, 500)  # shifted distribution
        kl = _kl_divergence(p, q)
        assert kl > 0.5

    def test_constant_array_returns_zero(self):
        p = np.ones(100)
        q = np.ones(100)
        kl = _kl_divergence(p, q)
        assert kl == 0.0

    def test_nonnegative(self):
        rng = np.random.default_rng(2)
        for _ in range(10):
            p = rng.uniform(0, 1, 200)
            q = rng.uniform(0, 1, 200)
            assert _kl_divergence(p, q) >= 0.0


class TestFlowEvaluator:
    def test_evaluate_returns_expected_keys(self, gen, evaluator):
        generated = gen.generate(AttackType.NORMAL, 50)
        reference = gen.generate(AttackType.NORMAL, 100)
        result = evaluator.evaluate(generated, reference)
        assert "kl_per_feature" in result
        assert "mean_kl" in result
        assert "n_generated" in result
        assert "n_reference" in result

    def test_evaluate_counts(self, gen, evaluator):
        generated = gen.generate(AttackType.DDOS, 40)
        reference = gen.generate(AttackType.DDOS, 80)
        result = evaluator.evaluate(generated, reference)
        assert result["n_generated"] == 40
        assert result["n_reference"] == 80

    def test_evaluate_same_flows_low_kl(self, gen, evaluator):
        """Same distribution (same seed, same attack type) should have low KL."""
        gen1 = NetworkFlowGenerator(seed=7)
        gen2 = NetworkFlowGenerator(seed=8)  # different seed, same type
        g = gen1.generate(AttackType.NORMAL, 200)
        r = gen2.generate(AttackType.NORMAL, 200)
        result = evaluator.evaluate(g, r)
        # Same-type flows should be fairly similar
        assert result["mean_kl"] < 2.0

    def test_evaluate_raises_on_empty(self, gen, evaluator):
        with pytest.raises(ValueError):
            evaluator.evaluate([], gen.generate(AttackType.NORMAL, 10))
        with pytest.raises(ValueError):
            evaluator.evaluate(gen.generate(AttackType.NORMAL, 10), [])

    def test_evaluate_all_types(self, gen, evaluator):
        gen_dict = {at.name: gen.generate(at, 30) for at in AttackType}
        ref_dict = {at.name: gen.generate(at, 60) for at in AttackType}
        results = evaluator.evaluate_all_types(gen_dict, ref_dict)
        assert set(results.keys()) == {at.name for at in AttackType}
        for r in results.values():
            assert "mean_kl" in r

    def test_summarize_single(self, gen, evaluator):
        generated = gen.generate(AttackType.BRUTE_FORCE, 50)
        reference = gen.generate(AttackType.BRUTE_FORCE, 100)
        result = evaluator.evaluate(generated, reference)
        summary = evaluator.summarize(result)
        assert "mean KL" in summary
        assert "port" in summary

    def test_summarize_multi(self, gen, evaluator):
        gen_dict = {at.name: gen.generate(at, 30) for at in AttackType}
        ref_dict = {at.name: gen.generate(at, 60) for at in AttackType}
        results = evaluator.evaluate_all_types(gen_dict, ref_dict)
        summary = evaluator.summarize(results)
        assert "NORMAL" in summary
        assert "DDOS" in summary
