from stable_baselines3.common.env_checker import check_env

from src.environment import PaintShopSchedulingEnv


def test_environment_passes_sb3_checker():
    env = PaintShopSchedulingEnv()
    check_env(env, warn=True)


def test_episode_completes_and_produces_schedule():
    env = PaintShopSchedulingEnv()
    obs, info = env.reset(seed=7)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(0)
        steps += 1
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert steps <= info["total_jobs"] + 1

    schedule = env.schedule_frame()
    assert len(schedule) == info["total_jobs"]
    assert schedule["job_id"].is_unique
    assert (schedule["completion_time"] >= schedule["paint_finish"]).all()
    assert info["on_time_rate"] >= 0.0
    assert info["on_time_rate"] <= 1.0
