"""Tests for CyberRangeSimulator."""

import pytest
from gan_cyber_range import CyberRangeSimulator, AttackType


@pytest.fixture
def trained_sim():
    sim = CyberRangeSimulator(hidden_dim=64, seed=42)
    sim.train(epochs=5, n_per_class=100, batch_size=32, verbose=False)
    return sim


@pytest.fixture
def untrained_sim():
    return CyberRangeSimulator(hidden_dim=64, seed=0)


class TestCyberRangeSimulator:
    def test_untrained_fallback(self, untrained_sim):
        flows = untrained_sim.generate_flows(AttackType.DDOS, n=10)
        assert len(flows) == 10
        for f in flows:
            assert f.attack_type == AttackType.DDOS

    def test_train_produces_history(self, trained_sim):
        history = trained_sim.train_history
        assert len(history) == 5
        for entry in history:
            assert "d_loss" in entry
            assert "g_loss" in entry
            assert "epoch" in entry

    def test_is_trained_flag(self, untrained_sim, trained_sim):
        assert not untrained_sim.is_trained
        assert trained_sim.is_trained

    def test_generate_flows_count(self, trained_sim):
        for at in AttackType:
            flows = trained_sim.generate_flows(at, n=25)
            assert len(flows) == 25

    def test_generate_flows_attack_type(self, trained_sim):
        flows = trained_sim.generate_flows(AttackType.PORT_SCAN, n=20)
        for f in flows:
            assert f.attack_type == AttackType.PORT_SCAN

    def test_generate_all_types(self, trained_sim):
        result = trained_sim.generate_all_types(n_per_type=10)
        assert set(result.keys()) == {at.name for at in AttackType}
        for flows in result.values():
            assert len(flows) == 10

    def test_generate_scenario_known(self, trained_sim):
        flows = trained_sim.generate_scenario("ddos_campaign", verbose=False)
        assert len(flows) == 100  # 10 + 80 + 10
        attack_names = {f.attack_type.name for f in flows}
        assert "DDOS" in attack_names

    def test_generate_scenario_unknown(self, trained_sim):
        with pytest.raises(ValueError, match="Unknown scenario"):
            trained_sim.generate_scenario("not_a_real_scenario")

    def test_save_load_roundtrip(self, trained_sim, tmp_path):
        path = str(tmp_path / "sim.pt")
        trained_sim.save_model(path)
        new_sim = CyberRangeSimulator(hidden_dim=64)
        new_sim.load_model(path)
        assert new_sim.is_trained
        flows = new_sim.generate_flows(AttackType.NORMAL, n=5)
        assert len(flows) == 5
