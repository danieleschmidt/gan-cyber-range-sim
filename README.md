# GAN Cyber Range Simulator

Generate realistic synthetic network attack traffic for cybersecurity training, red team exercises, and IDS/ML model development — without exposing real production infrastructure.

## What it does

A **Conditional GAN** (cGAN) learns the statistical distributions of real network flows for five attack categories. Once trained, it generates unlimited synthetic traffic that matches those distributions — complete with realistic IP addresses, ports, protocols, byte counts, and TCP flags.

### Attack Types

| Type | Description |
|------|-------------|
| `NORMAL` | Legitimate web/DNS/SSH traffic |
| `PORT_SCAN` | Stealthy SYN probe sequences targeting low ports |
| `DDOS` | High-volume flood traffic (SYN, UDP, ICMP) |
| `EXFILTRATION` | Large outbound data transfers on covert channels |
| `BRUTE_FORCE` | Repeated auth attempts on SSH/RDP/FTP/Telnet |

## Architecture

```
NetworkFlowGenerator          → baseline synthetic training data
         ↓
    TrafficGAN
    ├── Generator(noise + attack_type → flow features)
    └── Discriminator(flow features + attack_type → real/fake)
         ↓
  CyberRangeSimulator         → orchestrates training + scenario generation
         ↓
    FlowEvaluator             → KL divergence metrics vs reference distributions
```

### TrafficGAN (Conditional GAN)

- **Generator**: `(noise_dim + num_classes) → [hidden] → feature_dim` with LayerNorm + LeakyReLU
- **Discriminator**: `(feature_dim + num_classes) → [hidden] → 1 logit` with Dropout
- **Conditioning**: one-hot attack type embedding concatenated to input
- **Training**: standard GAN loss (BCE) with Adam optimizer (β₁=0.5)

### NetworkFlow

Each flow has 7 features: `src_ip`, `dst_ip`, `port`, `protocol`, `bytes`, `duration`, `flags`

All features are normalised to `[0, 1]` for the GAN and decoded back to human-readable form on output.

## Quick Start

```bash
pip install -r requirements.txt
```

```python
from gan_cyber_range import CyberRangeSimulator, AttackType

sim = CyberRangeSimulator()
sim.train(epochs=100, n_per_class=500)

# Generate 100 flows of each attack type
flows_by_type = sim.generate_all_types(n_per_type=100)
for name, flows in flows_by_type.items():
    print(f"{name}: {flows[0].to_dict()}")

# Run a multi-stage attack scenario
scenario = sim.generate_scenario("port_scan_then_exfil")
```

## Demo

```bash
python demo.py
```

Trains the GAN (100 epochs), generates 100 flows per attack type, evaluates KL divergence against reference distributions, and runs a `port_scan_then_exfil` scenario.

## Scenarios

Pre-built multi-stage attack scenarios:

| Scenario | Description |
|----------|-------------|
| `port_scan_then_exfil` | Reconnaissance → blend in → data theft |
| `ddos_campaign` | Normal → flood → recovery |
| `brute_force_then_exfil` | Password spray → exfiltration |
| `mixed_threat` | All five attack types simultaneously |

```python
sim.generate_scenario("mixed_threat")
```

## Evaluation

`FlowEvaluator` measures statistical similarity via per-feature KL divergence:

```python
from gan_cyber_range import FlowEvaluator

evaluator = FlowEvaluator(bins=30)
result = evaluator.evaluate(generated_flows, reference_flows)
print(f"Mean KL divergence: {result['mean_kl']:.4f}")
print(evaluator.summarize(result))
```

Lower KL → generated flows more closely match real distributions.

## Save / Load

```python
sim.save_model("checkpoints/traffic_gan.pt")

# Later:
sim2 = CyberRangeSimulator()
sim2.load_model("checkpoints/traffic_gan.pt")
flows = sim2.generate_flows(AttackType.EXFILTRATION, n=500)
```

## Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/test_flows.py tests/test_models.py tests/test_simulator.py tests/test_evaluator.py -v
```

## Use Cases

- **IDS training data**: augment sparse attack datasets for ML-based intrusion detection
- **Red team exercises**: generate realistic traffic patterns for detection team drills  
- **Tabletop simulations**: replay attack scenarios without live infrastructure risk
- **Model evaluation**: benchmark classifiers against GAN-generated adversarial flows
- **Privacy-preserving research**: share synthetic datasets derived from real captures

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- NumPy ≥ 1.24

GPU is supported automatically (`cuda` if available, else `cpu`).

## License

MIT
