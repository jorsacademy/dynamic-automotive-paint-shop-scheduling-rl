from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment import EnvConfig, PaintShopSchedulingEnv


def make_env(seed: int = 42):
    def _factory():
        return Monitor(PaintShopSchedulingEnv(EnvConfig(seed=seed)))
    return _factory


def train(total_timesteps: int = 150_000, seed: int = 42, output_dir: str = "models") -> Path:
    raw_env = PaintShopSchedulingEnv(EnvConfig(seed=seed))
    check_env(raw_env, warn=True)

    vec_env = DummyVecEnv([make_env(seed)])
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        verbose=1,
        seed=seed,
        policy_kwargs={"net_arch": [128, 128]},
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "ppo_paint_shop"
    model.save(model_path)
    return model_path


if __name__ == "__main__":
    path = train()
    print(f"Saved model to {path}")
