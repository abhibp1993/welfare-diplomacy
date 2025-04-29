import random
import subprocess
from concurrent.futures import ProcessPoolExecutor

NUM_RUNS = 1
PROJECT_NAME = "neurips_press_test"


def simulate_game(seed):
    print(f"Running simulation with seed: {seed}")
    subprocess.run([
        "python", "simulate_game_wdmonitor.py",
        "--project", f"{PROJECT_NAME}",
        "--run_name", f"seed_{seed}",
        "--max_years", "2",
        "--max_message_rounds", "1",
        "--agent_model", "gpt-4o-mini",
        "--summarizer_model", "gpt-4o-mini",
        "--save",
        "--output_folder", f"out/{PROJECT_NAME}",
        "--seed", f"{seed}"
    ])


def main():
    random_seeds = {random.randint(0, 1000) for _ in range(NUM_RUNS)}
    with ProcessPoolExecutor() as executor:
        executor.map(simulate_game, random_seeds)


if __name__ == "__main__":
    main()
