"""Network flow data structures and synthetic baseline data generation."""

from __future__ import annotations

import random
import ipaddress
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Dict, Any

import numpy as np
import torch


class AttackType(IntEnum):
    NORMAL = 0
    PORT_SCAN = 1
    DDOS = 2
    EXFILTRATION = 3
    BRUTE_FORCE = 4

    @classmethod
    def names(cls) -> List[str]:
        return [e.name for e in cls]


@dataclass
class NetworkFlow:
    """Represents a single network flow record."""
    src_ip: str
    dst_ip: str
    port: int
    protocol: int          # 6=TCP, 17=UDP, 1=ICMP
    bytes: int
    duration: float        # seconds
    flags: int             # TCP flags bitmask (SYN=0x02, ACK=0x10, FIN=0x01, RST=0x04)
    attack_type: AttackType

    def to_feature_vector(self) -> np.ndarray:
        """Return normalised float feature vector (7 dims)."""
        return np.array([
            self.port / 65535.0,
            self.protocol / 255.0,
            np.log1p(self.bytes) / 20.0,
            np.log1p(self.duration) / 10.0,
            self.flags / 63.0,
            int(ipaddress.ip_address(self.src_ip)) / float(2**32 - 1),
            int(ipaddress.ip_address(self.dst_ip)) / float(2**32 - 1),
        ], dtype=np.float32)

    @staticmethod
    def from_feature_vector(vec: np.ndarray, attack_type: AttackType) -> "NetworkFlow":
        """Reconstruct a NetworkFlow from a normalised feature vector."""
        port = int(np.clip(vec[0] * 65535, 0, 65535))
        protocol = int(np.clip(vec[1] * 255, 0, 255))
        bytes_ = int(np.expm1(np.clip(vec[2] * 20, 0, 40)))
        duration = float(np.expm1(np.clip(vec[3] * 10, 0, 30)))
        flags = int(np.clip(vec[4] * 63, 0, 63))
        src_int = int(np.clip(vec[5] * (2**32 - 1), 0, 2**32 - 1))
        dst_int = int(np.clip(vec[6] * (2**32 - 1), 0, 2**32 - 1))
        try:
            src_ip = str(ipaddress.ip_address(src_int))
            dst_ip = str(ipaddress.ip_address(dst_int))
        except Exception:
            src_ip, dst_ip = "10.0.0.1", "192.168.1.1"
        return NetworkFlow(src_ip=src_ip, dst_ip=dst_ip, port=port,
                           protocol=protocol, bytes=bytes_, duration=duration,
                           flags=flags, attack_type=attack_type)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "port": self.port,
            "protocol": self.protocol,
            "bytes": self.bytes,
            "duration": round(self.duration, 4),
            "flags": self.flags,
            "attack_type": self.attack_type.name,
        }


FEATURE_DIM = 7  # dimensionality of the feature vector


class NetworkFlowGenerator:
    """Generates synthetic (rule-based) baseline network flows for GAN training."""

    _PRIVATE_RANGES = [
        ("10.0.0.0", "10.255.255.255"),
        ("172.16.0.0", "172.31.255.255"),
        ("192.168.0.0", "192.168.255.255"),
    ]

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def _rand_private_ip(self) -> str:
        lo, hi = self._rng.choice(self._PRIVATE_RANGES)
        lo_int = int(ipaddress.ip_address(lo))
        hi_int = int(ipaddress.ip_address(hi))
        return str(ipaddress.ip_address(self._rng.randint(lo_int, hi_int)))

    def _rand_public_ip(self) -> str:
        while True:
            a = self._rng.randint(1, 223)
            if a in (10, 127, 172, 192):
                continue
            b, c, d = (self._rng.randint(0, 255) for _ in range(3))
            return f"{a}.{b}.{c}.{d}"

    def generate(self, attack_type: AttackType, n: int = 1) -> List[NetworkFlow]:
        flows = []
        for _ in range(n):
            flows.append(self._generate_one(attack_type))
        return flows

    def _generate_one(self, attack_type: AttackType) -> NetworkFlow:
        rng = self._rng
        np_rng = self._np_rng

        if attack_type == AttackType.NORMAL:
            port = rng.choice([80, 443, 22, 53, 8080, 8443,
                                rng.randint(1024, 65535)])
            protocol = rng.choices([6, 17], weights=[0.7, 0.3])[0]
            bytes_ = int(np_rng.lognormal(8, 2))  # ~3KB median
            duration = float(np_rng.lognormal(1, 1.5))
            flags = 0x12 if protocol == 6 else 0  # SYN-ACK for TCP

        elif attack_type == AttackType.PORT_SCAN:
            port = rng.randint(1, 1024)  # scanning low ports
            protocol = 6  # TCP
            bytes_ = rng.randint(40, 200)  # tiny probe packets
            duration = float(np_rng.exponential(0.1))  # very short
            flags = 0x02  # SYN only (stealth scan)

        elif attack_type == AttackType.DDOS:
            port = rng.choice([80, 443, 53])
            protocol = rng.choices([6, 17, 1], weights=[0.4, 0.4, 0.2])[0]
            bytes_ = int(np_rng.lognormal(5, 0.5))  # medium flood packets
            duration = float(np_rng.exponential(0.05))  # rapid fire
            flags = 0x02 if protocol == 6 else 0  # SYN flood

        elif attack_type == AttackType.EXFILTRATION:
            port = rng.choice([443, 80, 53, 8443,
                                rng.randint(4000, 9000)])  # covert channels
            protocol = rng.choices([6, 17], weights=[0.6, 0.4])[0]
            bytes_ = int(np_rng.lognormal(12, 1))  # large outbound transfer
            duration = float(np_rng.lognormal(4, 1))  # longer sessions
            flags = 0x18 if protocol == 6 else 0  # PSH+ACK (data transfer)

        elif attack_type == AttackType.BRUTE_FORCE:
            port = rng.choice([22, 3389, 21, 23, 25, 110])  # auth services
            protocol = 6  # TCP
            bytes_ = int(np_rng.lognormal(6, 0.5))  # login attempt packets
            duration = float(np_rng.exponential(2))  # repeated attempts
            flags = 0x18  # PSH+ACK

        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        # Direction: exfil goes internal→external, others vary
        if attack_type == AttackType.EXFILTRATION:
            src_ip = self._rand_private_ip()
            dst_ip = self._rand_public_ip()
        elif attack_type == AttackType.PORT_SCAN:
            src_ip = self._rand_public_ip()
            dst_ip = self._rand_private_ip()
        else:
            src_ip = self._rand_private_ip()
            dst_ip = (self._rand_private_ip() if rng.random() > 0.4
                      else self._rand_public_ip())

        return NetworkFlow(
            src_ip=src_ip,
            dst_ip=dst_ip,
            port=max(1, min(65535, port)),
            protocol=protocol,
            bytes=max(40, min(int(2e9), bytes_)),
            duration=max(0.001, duration),
            flags=flags & 0x3F,
            attack_type=attack_type,
        )

    def generate_training_batch(self, n_per_class: int = 500) -> tuple:
        """Return (feature_tensor, label_tensor) for all attack types."""
        features, labels = [], []
        for at in AttackType:
            flows = self.generate(at, n_per_class)
            for f in flows:
                features.append(f.to_feature_vector())
                labels.append(int(at))
        X = torch.tensor(np.array(features), dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        return X, y
