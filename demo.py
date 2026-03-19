#!/usr/bin/env python3
"""
GAN Cyber Range Simulator — Demo
=================================

Trains a conditional GAN on synthetic network flow data, then generates
100 flows per attack type and evaluates statistical similarity.
"""

import sys
from gan_cyber_range import CyberRangeSimulator, FlowEvaluator, AttackType
from gan_cyber_range.flows import NetworkFlowGenerator

EPOCHS = 100
N_PER_CLASS = 600
N_GENERATE = 100


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def main():
    section("1. Training the Traffic GAN")
    sim = CyberRangeSimulator(hidden_dim=128, lr=2e-4)
    sim.train(epochs=EPOCHS, n_per_class=N_PER_CLASS,
              batch_size=128, log_interval=20)

    section("2. Generating 100 flows per attack type")
    generated_by_type = sim.generate_all_types(n_per_type=N_GENERATE)

    for attack_name, flows in generated_by_type.items():
        print(f"\n  [{attack_name}] — {len(flows)} flows")
        print(f"  Sample flow: {flows[0].to_dict()}")

    section("3. Statistical Evaluation (KL divergence vs reference)")
    ref_gen = NetworkFlowGenerator(seed=99)
    reference_by_type = {
        at.name: ref_gen.generate(at, N_GENERATE * 2)
        for at in AttackType
    }

    evaluator = FlowEvaluator(bins=30)
    results = evaluator.evaluate_all_types(generated_by_type, reference_by_type)

    print(evaluator.summarize(results))

    section("4. Scenario: port_scan_then_exfil")
    scenario_flows = sim.generate_scenario("port_scan_then_exfil")
    by_type: dict = {}
    for f in scenario_flows:
        by_type.setdefault(f.attack_type.name, 0)
        by_type[f.attack_type.name] += 1

    print(f"\n  Scenario breakdown: {by_type}")
    print(f"  First flow : {scenario_flows[0].to_dict()}")
    print(f"  Last flow  : {scenario_flows[-1].to_dict()}")

    section("5. Summary")
    mean_kls = {k: v["mean_kl"] for k, v in results.items()}
    best = min(mean_kls, key=mean_kls.get)
    worst = max(mean_kls, key=mean_kls.get)
    print(f"\n  Best  fidelity: {best} (KL={mean_kls[best]:.4f})")
    print(f"  Worst fidelity: {worst} (KL={mean_kls[worst]:.4f})")
    print(f"\n  GAN-based cyber range demo complete. ✓")


if __name__ == "__main__":
    main()
