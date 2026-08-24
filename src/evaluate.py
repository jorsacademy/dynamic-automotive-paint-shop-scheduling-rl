from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.data_generator import GeneratorConfig, generate_jobs
from src.dispatch_rules import RULE_NAMES
from src.environment import EnvConfig, PaintShopSchedulingEnv


def run_policy(env: PaintShopSchedulingEnv, policy) -> dict:
    obs, info = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action = int(policy(obs))
        obs, _, terminated, truncated, info = env.step(action)
    return info | {"schedule": env.schedule_frame()}


def evaluate_rule(rule_index: int, jobs: pd.DataFrame, seed: int = 100) -> dict:
    env = PaintShopSchedulingEnv(EnvConfig(seed=seed), jobs=jobs)
    result = run_policy(env, lambda obs: rule_index)
    result["policy"] = RULE_NAMES[rule_index]
    return result


def evaluate_model(model_path: str = "models/ppo_paint_shop", episodes: int = 20) -> pd.DataFrame:
    model = PPO.load(model_path)
    rows = []
    for episode in range(episodes):
        seed = 10_000 + episode
        jobs = generate_jobs(GeneratorConfig(seed=seed))
        env = PaintShopSchedulingEnv(EnvConfig(seed=seed), jobs=jobs)
        obs, _ = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
        rows.append({"episode": episode, "policy": "ppo", **info})
    return pd.DataFrame(rows)


def benchmark(model_path: str = "models/ppo_paint_shop", episodes: int = 10) -> pd.DataFrame:
    model = PPO.load(model_path)
    rows = []
    for episode in range(episodes):
        seed = 20_000 + episode
        jobs = generate_jobs(GeneratorConfig(seed=seed))

        for rule_index, rule_name in enumerate(RULE_NAMES):
            result = evaluate_rule(rule_index, jobs, seed)
            rows.append({
                "episode": episode,
                "policy": rule_name,
                **{k: v for k, v in result.items() if k != "schedule"},
            })

        env = PaintShopSchedulingEnv(EnvConfig(seed=seed), jobs=jobs)
        obs, _ = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
        rows.append({"episode": episode, "policy": "ppo", **info})

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_waiting_time",
        "total_tardiness",
        "weighted_tardiness",
        "setup_time",
        "idle_time",
        "breakdown_time",
        "throughput_rate",
        "on_time_rate",
    ]
    summary = results.groupby("policy")[metrics].agg(["mean", "std"])
    return summary.sort_values(("weighted_tardiness", "mean"))


if __name__ == "__main__":
    model_path = Path("models/ppo_paint_shop")
    results = benchmark(str(model_path), episodes=10)
    Path("results").mkdir(exist_ok=True)
    results.to_csv("results/benchmark.csv", index=False)
    print(summarize(results).round(4))
