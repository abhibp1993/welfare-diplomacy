import os
import sys
import subprocess

import experiments.simulate_game_baseline as sim_game


def main():
    import welfare_diplomacy_baselines.baselines.no_press_policies as no_press_policies
    # policy_id =
    for policy_id in no_press_policies.policy_map.keys():
        print(f"Running simulation with policy ID: {policy_id}")
        subprocess.run([
            "python", "simulate_game_baseline.py",
            "--project", "neurips_baselines",
            "--max_years", "10",
            "--max_message_rounds", "3",
            "--agent_model", "gpt-4o-mini",
            "--summarizer_model", "gpt-4o-mini",
            "--save",
            "--no_press", "True",
            "--no_press_policy", f"{policy_id}"
        ])


if __name__ == "__main__":
    main()
