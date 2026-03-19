"""Tests for NetworkFlowGenerator and NetworkFlow."""

import pytest
import numpy as np
from gan_cyber_range.flows import NetworkFlow, NetworkFlowGenerator, AttackType, FEATURE_DIM


@pytest.fixture
def gen():
    return NetworkFlowGenerator(seed=42)


class TestAttackType:
    def test_all_types_present(self):
        names = AttackType.names()
        assert set(names) == {"NORMAL", "PORT_SCAN", "DDOS", "EXFILTRATION", "BRUTE_FORCE"}

    def test_five_classes(self):
        assert len(AttackType) == 5


class TestNetworkFlow:
    def test_feature_vector_shape(self, gen):
        flow = gen.generate(AttackType.NORMAL, 1)[0]
        vec = flow.to_feature_vector()
        assert vec.shape == (FEATURE_DIM,)

    def test_feature_vector_range(self, gen):
        for at in AttackType:
            flows = gen.generate(at, 20)
            for f in flows:
                vec = f.to_feature_vector()
                assert vec.min() >= 0.0, f"feature below 0 for {at.name}"
                assert vec.max() <= 1.0, f"feature above 1 for {at.name}"

    def test_roundtrip(self, gen):
        flow = gen.generate(AttackType.DDOS, 1)[0]
        vec = flow.to_feature_vector()
        recovered = NetworkFlow.from_feature_vector(vec, AttackType.DDOS)
        assert recovered.attack_type == AttackType.DDOS
        # Port should be close (some rounding expected)
        assert abs(recovered.port - flow.port) <= 1

    def test_to_dict_keys(self, gen):
        flow = gen.generate(AttackType.NORMAL, 1)[0]
        d = flow.to_dict()
        expected_keys = {"src_ip", "dst_ip", "port", "protocol",
                         "bytes", "duration", "flags", "attack_type"}
        assert set(d.keys()) == expected_keys

    def test_to_dict_attack_type_is_string(self, gen):
        flow = gen.generate(AttackType.PORT_SCAN, 1)[0]
        d = flow.to_dict()
        assert d["attack_type"] == "PORT_SCAN"


class TestNetworkFlowGenerator:
    def test_generate_count(self, gen):
        for at in AttackType:
            flows = gen.generate(at, 50)
            assert len(flows) == 50

    def test_generate_types_match(self, gen):
        for at in AttackType:
            flows = gen.generate(at, 10)
            for f in flows:
                assert f.attack_type == at

    def test_port_scan_small_bytes(self, gen):
        flows = gen.generate(AttackType.PORT_SCAN, 100)
        avg_bytes = sum(f.bytes for f in flows) / len(flows)
        # Port scans should be small probes
        assert avg_bytes < 1000, f"Expected small packets for port scan, got avg={avg_bytes}"

    def test_exfil_large_bytes(self, gen):
        flows = gen.generate(AttackType.EXFILTRATION, 100)
        avg_bytes = sum(f.bytes for f in flows) / len(flows)
        # Exfiltration should move significant data
        assert avg_bytes > 500, f"Expected large transfer for exfil, got avg={avg_bytes}"

    def test_brute_force_port(self, gen):
        flows = gen.generate(AttackType.BRUTE_FORCE, 100)
        auth_ports = {22, 3389, 21, 23, 25, 110}
        for f in flows:
            assert f.port in auth_ports, f"Unexpected port {f.port} for BRUTE_FORCE"

    def test_training_batch_shape(self, gen):
        X, y = gen.generate_training_batch(n_per_class=100)
        assert X.shape == (500, FEATURE_DIM)  # 5 classes × 100
        assert y.shape == (500,)
        assert y.min() == 0
        assert y.max() == 4
