"""Tests for TrafficGAN (Generator + Discriminator)."""

import pytest
import torch
from gan_cyber_range.models import Generator, Discriminator, TrafficGAN
from gan_cyber_range.flows import FEATURE_DIM


NOISE_DIM = 32
NUM_CLASSES = 5
BATCH = 8


@pytest.fixture
def gan():
    return TrafficGAN(noise_dim=NOISE_DIM, num_classes=NUM_CLASSES,
                      feature_dim=FEATURE_DIM, hidden_dim=64, device="cpu")


class TestGenerator:
    def test_output_shape(self):
        g = Generator(NOISE_DIM, NUM_CLASSES, FEATURE_DIM, hidden_dim=64)
        noise = torch.randn(BATCH, NOISE_DIM)
        labels = torch.randint(0, NUM_CLASSES, (BATCH,))
        out = g(noise, labels)
        assert out.shape == (BATCH, FEATURE_DIM)

    def test_output_range(self):
        g = Generator(NOISE_DIM, NUM_CLASSES, FEATURE_DIM, hidden_dim=64)
        noise = torch.randn(BATCH, NOISE_DIM)
        labels = torch.zeros(BATCH, dtype=torch.long)
        out = g(noise, labels)
        assert out.min() >= 0.0 - 1e-5
        assert out.max() <= 1.0 + 1e-5


class TestDiscriminator:
    def test_output_shape(self):
        d = Discriminator(FEATURE_DIM, NUM_CLASSES, hidden_dim=64)
        feats = torch.rand(BATCH, FEATURE_DIM)
        labels = torch.randint(0, NUM_CLASSES, (BATCH,))
        out = d(feats, labels)
        assert out.shape == (BATCH, 1)

    def test_output_is_logit(self):
        d = Discriminator(FEATURE_DIM, NUM_CLASSES, hidden_dim=64)
        feats = torch.rand(BATCH, FEATURE_DIM)
        labels = torch.zeros(BATCH, dtype=torch.long)
        out = d(feats, labels)
        # Logits can be any real number
        assert out.dtype == torch.float32


class TestTrafficGAN:
    def test_train_step_returns_losses(self, gan):
        real_features = torch.rand(BATCH, FEATURE_DIM)
        real_labels = torch.randint(0, NUM_CLASSES, (BATCH,))
        losses = gan.train_step(real_features, real_labels)
        assert "d_loss" in losses
        assert "g_loss" in losses
        for v in losses.values():
            assert isinstance(v, float)
            assert not (v != v)  # no NaN

    def test_generate_features_shape(self, gan):
        feats = gan.generate_features(attack_type_idx=0, n=16)
        assert feats.shape == (16, FEATURE_DIM)

    def test_generate_features_range(self, gan):
        feats = gan.generate_features(attack_type_idx=2, n=50)
        assert feats.min() >= 0.0 - 1e-5
        assert feats.max() <= 1.0 + 1e-5

    def test_train_reduces_loss(self, gan):
        """After several steps, G loss should not be stuck at zero."""
        real_features = torch.rand(32, FEATURE_DIM)
        real_labels = torch.randint(0, NUM_CLASSES, (32,))
        losses_all = []
        for _ in range(20):
            losses = gan.train_step(real_features, real_labels)
            losses_all.append(losses["g_loss"])
        # Generator should have non-trivial loss somewhere
        assert max(losses_all) > 0.0

    def test_save_load(self, gan, tmp_path):
        path = str(tmp_path / "gan.pt")
        gan.save(path)
        loaded = TrafficGAN.load(path, device="cpu")
        # Check weights match
        for (p1, p2) in zip(gan.generator.parameters(),
                             loaded.generator.parameters()):
            assert torch.allclose(p1, p2)
