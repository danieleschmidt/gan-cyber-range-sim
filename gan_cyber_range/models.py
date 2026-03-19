"""Conditional GAN for network traffic generation.

Architecture
------------
Generator  : (noise_dim + num_classes) → hidden → FEATURE_DIM
Discriminator: (FEATURE_DIM + num_classes) → hidden → 1 (real/fake logit)

Conditioning is done via one-hot class embedding concatenated to input.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, noise_dim: int = 64, num_classes: int = 5,
                 feature_dim: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(noise_dim + num_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid(),  # all features normalised to [0, 1]
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noise:  (B, noise_dim)
            labels: (B,) long tensor of class indices
        Returns:
            (B, feature_dim) synthetic flow features in [0,1]
        """
        one_hot = F.one_hot(labels, self.num_classes).float()
        x = torch.cat([noise, one_hot], dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, feature_dim: int = 7, num_classes: int = 5,
                 hidden_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(feature_dim + num_classes, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, feature_dim)
            labels:   (B,) long tensor of class indices
        Returns:
            (B, 1) logits (real > 0, fake < 0)
        """
        one_hot = F.one_hot(labels, self.num_classes).float()
        x = torch.cat([features, one_hot], dim=1)
        return self.net(x)


class TrafficGAN(nn.Module):
    """Conditional GAN that generates realistic network traffic flows."""

    def __init__(self, noise_dim: int = 64, num_classes: int = 5,
                 feature_dim: int = 7, hidden_dim: int = 128,
                 lr: float = 2e-4, device: str | None = None):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.generator = Generator(noise_dim, num_classes, feature_dim, hidden_dim)
        self.discriminator = Discriminator(feature_dim, num_classes, hidden_dim)

        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        self.to(self.device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(
        self,
        real_features: torch.Tensor,
        real_labels: torch.Tensor,
    ) -> dict[str, float]:
        """Single training step. Returns loss dict."""
        real_features = real_features.to(self.device)
        real_labels = real_labels.to(self.device)
        B = real_features.size(0)

        # -------- Train Discriminator --------
        self.d_optimizer.zero_grad()

        # Real samples
        real_logits = self.discriminator(real_features, real_labels)
        d_loss_real = F.binary_cross_entropy_with_logits(
            real_logits, torch.ones_like(real_logits))

        # Fake samples
        noise = torch.randn(B, self.noise_dim, device=self.device)
        fake_features = self.generator(noise, real_labels).detach()
        fake_logits = self.discriminator(fake_features, real_labels)
        d_loss_fake = F.binary_cross_entropy_with_logits(
            fake_logits, torch.zeros_like(fake_logits))

        d_loss = (d_loss_real + d_loss_fake) * 0.5
        d_loss.backward()
        self.d_optimizer.step()

        # -------- Train Generator --------
        self.g_optimizer.zero_grad()

        noise = torch.randn(B, self.noise_dim, device=self.device)
        fake_features = self.generator(noise, real_labels)
        fake_logits = self.discriminator(fake_features, real_labels)
        g_loss = F.binary_cross_entropy_with_logits(
            fake_logits, torch.ones_like(fake_logits))

        g_loss.backward()
        self.g_optimizer.step()

        return {
            "d_loss": d_loss.item(),
            "d_loss_real": d_loss_real.item(),
            "d_loss_fake": d_loss_fake.item(),
            "g_loss": g_loss.item(),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_features(self, attack_type_idx: int, n: int = 1) -> torch.Tensor:
        """Generate n synthetic feature vectors for a given attack type.

        Returns:
            Tensor of shape (n, feature_dim) on CPU in [0,1].
        """
        self.generator.eval()
        noise = torch.randn(n, self.noise_dim, device=self.device)
        labels = torch.full((n,), attack_type_idx, dtype=torch.long,
                            device=self.device)
        features = self.generator(noise, labels)
        self.generator.train()
        return features.cpu()

    def save(self, path: str) -> None:
        torch.save({
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "config": {
                "noise_dim": self.noise_dim,
                "num_classes": self.num_classes,
                "feature_dim": self.feature_dim,
                "hidden_dim": self.generator.net[0].out_features,  # inferred from weights
            },
        }, path)

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "TrafficGAN":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]
        gan = cls(device=device, **cfg)
        gan.generator.load_state_dict(checkpoint["generator"])
        gan.discriminator.load_state_dict(checkpoint["discriminator"])
        return gan
