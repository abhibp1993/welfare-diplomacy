import os
import sys
import subprocess
import random
from concurrent.futures import ProcessPoolExecutor
import experiments.simulate_game_baseline as sim_game
from functools import partial

NUM_RUNS = 2
LLM_MODEL = "o1-mini"
RND_SEEDS = [random.randint(0, 1000) for _ in range(NUM_RUNS)]


def simulate_game(seed, model_name, policy_id):
    project_name = f"baseline_np_policy_{policy_id}"
    subprocess.run([
        "python", "simulate_game_baseline.py",
        "--project", project_name,
        "--run_name", f"{model_name}_seed_{seed}",
        "--max_years", "10",
        "--max_message_rounds", "3",
        "--agent_model", LLM_MODEL,
        "--summarizer_model", LLM_MODEL,
        "--save",
        "--output_folder", f"out/{project_name}",
        "--no_press", "True",
        "--no_press_policy", f"{policy_id}",
        "--seed", f"{seed}",
    ])


def main():
    # import welfare_diplomacy_baselines.baselines.no_press_policies as no_press_policies
    # policies = [2, 12, 22, 32]
    policy_id = 12
    # for policy_id in policies:
    with ProcessPoolExecutor() as executor:
        executor.map(partial(simulate_game, model_name=LLM_MODEL, policy_id=policy_id), RND_SEEDS)


if __name__ == "__main__":
    main()
