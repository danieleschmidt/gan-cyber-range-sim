"""CyberRangeSimulator: orchestrates GAN training and scenario generation."""

from __future__ import annotations

import time
from typing import List, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .flows import AttackType, NetworkFlow, NetworkFlowGenerator, FEATURE_DIM
from .models import TrafficGAN


class CyberRangeSimulator:
    """High-level orchestrator for training and deploying the traffic GAN.

    Example usage::

        sim = CyberRangeSimulator()
        sim.train(epochs=50)
        scenario = sim.generate_scenario("port_scan_then_exfil")
        for flow in scenario:
            print(flow.to_dict())
    """

    SCENARIOS: Dict[str, List[tuple[AttackType, int]]] = {
        "port_scan_then_exfil": [
            (AttackType.PORT_SCAN, 30),
            (AttackType.NORMAL, 10),   # blend in
            (AttackType.EXFILTRATION, 20),
        ],
        "ddos_campaign": [
            (AttackType.NORMAL, 10),
            (AttackType.DDOS, 80),
            (AttackType.NORMAL, 10),
        ],
        "brute_force_then_exfil": [
            (AttackType.BRUTE_FORCE, 40),
            (AttackType.EXFILTRATION, 30),
        ],
        "mixed_threat": [
            (AttackType.PORT_SCAN, 20),
            (AttackType.BRUTE_FORCE, 20),
            (AttackType.DDOS, 20),
            (AttackType.EXFILTRATION, 20),
            (AttackType.NORMAL, 20),
        ],
    }

    def __init__(
        self,
        noise_dim: int = 64,
        hidden_dim: int = 128,
        lr: float = 2e-4,
        device: str | None = None,
        seed: int = 42,
    ):
        self.gan = TrafficGAN(
            noise_dim=noise_dim,
            feature_dim=FEATURE_DIM,
            hidden_dim=hidden_dim,
            lr=lr,
            device=device,
        )
        self.baseline_gen = NetworkFlowGenerator(seed=seed)
        self._trained = False
        self._train_history: List[Dict] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        epochs: int = 100,
        n_per_class: int = 500,
        batch_size: int = 64,
        verbose: bool = True,
        log_interval: int = 10,
    ) -> List[Dict]:
        """Train the GAN on synthetic baseline data.

        Args:
            epochs:       number of training epochs
            n_per_class:  baseline samples per attack class
            batch_size:   mini-batch size
            verbose:      print progress
            log_interval: log every N epochs

        Returns:
            List of per-epoch loss dicts.
        """
        if verbose:
            print(f"[CyberRangeSimulator] Generating {n_per_class * len(AttackType)} "
                  f"baseline flows for training…")

        X, y = self.baseline_gen.generate_training_batch(n_per_class)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True)

        history = []
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            epoch_losses: Dict[str, float] = {
                "d_loss": 0, "g_loss": 0, "d_loss_real": 0, "d_loss_fake": 0
            }
            n_batches = 0

            for real_features, real_labels in loader:
                step_losses = self.gan.train_step(real_features, real_labels)
                for k in epoch_losses:
                    epoch_losses[k] += step_losses[k]
                n_batches += 1

            for k in epoch_losses:
                epoch_losses[k] /= max(n_batches, 1)

            epoch_losses["epoch"] = epoch
            history.append(epoch_losses)

            if verbose and epoch % log_interval == 0:
                elapsed = time.time() - t0
                print(f"  Epoch {epoch:4d}/{epochs} | "
                      f"D={epoch_losses['d_loss']:.4f} "
                      f"(real={epoch_losses['d_loss_real']:.4f} "
                      f"fake={epoch_losses['d_loss_fake']:.4f}) | "
                      f"G={epoch_losses['g_loss']:.4f} | "
                      f"{elapsed:.1f}s elapsed")

        self._trained = True
        self._train_history = history

        if verbose:
            print(f"[CyberRangeSimulator] Training complete. "
                  f"Final G loss={history[-1]['g_loss']:.4f}, "
                  f"D loss={history[-1]['d_loss']:.4f}")

        return history

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_flows(
        self,
        attack_type: AttackType,
        n: int = 100,
    ) -> List[NetworkFlow]:
        """Generate n synthetic flows for a given attack type.

        If the GAN is not yet trained, falls back to rule-based generation.
        """
        if not self._trained:
            return self.baseline_gen.generate(attack_type, n)

        feature_tensor = self.gan.generate_features(int(attack_type), n)
        features_np = feature_tensor.numpy()
        return [
            NetworkFlow.from_feature_vector(features_np[i], attack_type)
            for i in range(n)
        ]

    def generate_scenario(
        self,
        scenario_name: str,
        verbose: bool = True,
    ) -> List[NetworkFlow]:
        """Generate a named multi-stage attack scenario.

        Args:
            scenario_name: key in SCENARIOS dict (or 'list' to see options)

        Returns:
            Ordered list of NetworkFlow objects representing the scenario.
        """
        if scenario_name == "list":
            return list(self.SCENARIOS.keys())  # type: ignore[return-value]

        if scenario_name not in self.SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario_name}'. "
                f"Available: {list(self.SCENARIOS.keys())}"
            )

        steps = self.SCENARIOS[scenario_name]
        all_flows: List[NetworkFlow] = []

        if verbose:
            print(f"[CyberRangeSimulator] Generating scenario: {scenario_name!r}")

        for attack_type, count in steps:
            flows = self.generate_flows(attack_type, count)
            all_flows.extend(flows)
            if verbose:
                print(f"  + {count} flows ({attack_type.name})")

        if verbose:
            print(f"  → Total: {len(all_flows)} flows")

        return all_flows

    def generate_all_types(self, n_per_type: int = 100) -> Dict[str, List[NetworkFlow]]:
        """Generate n flows for each attack type."""
        return {
            at.name: self.generate_flows(at, n_per_type)
            for at in AttackType
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str) -> None:
        self.gan.save(path)
        print(f"[CyberRangeSimulator] Model saved to {path}")

    def load_model(self, path: str) -> None:
        self.gan = TrafficGAN.load(path)
        self._trained = True
        print(f"[CyberRangeSimulator] Model loaded from {path}")

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def train_history(self) -> List[Dict]:
        return self._train_history
