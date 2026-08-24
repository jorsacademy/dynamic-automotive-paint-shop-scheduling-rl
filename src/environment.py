from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from src.data_generator import GeneratorConfig, generate_jobs
from src.dispatch_rules import DISPATCH_RULES, _setup_time


@dataclass(frozen=True)
class EnvConfig:
    num_booths: int = 3
    horizon_minutes: int = 8 * 60
    max_queue_reference: float = 30.0
    max_wait_reference: float = 180.0
    max_workload_reference: float = 600.0
    breakdown_probability: float = 0.015
    breakdown_duration_min: float = 8.0
    breakdown_duration_max: float = 25.0
    seed: int = 42


class PaintShopSchedulingEnv(gym.Env[np.ndarray, int]):
    """Event-driven RL environment for dynamic automotive paint-shop scheduling.

    The RL action selects one of several dispatching rules. At each decision epoch,
    the selected rule chooses a job from the currently available queue for the next
    available paint booth. New jobs continue to arrive while processing takes place.
    This keeps both observation and action dimensions fixed while preserving dynamic
    scheduling behavior.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig = EnvConfig(), jobs: pd.DataFrame | None = None):
        super().__init__()
        self.config = config
        self.external_jobs = jobs.copy() if jobs is not None else None
        self.action_space = spaces.Discrete(len(DISPATCH_RULES))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(14,), dtype=np.float32)
        self.rng = np.random.default_rng(config.seed)
        self.jobs = pd.DataFrame()
        self.queue_ids: list[int] = []
        self.completed_ids: list[int] = []
        self.current_time = 0.0
        self.next_arrival_index = 0
        self.booth_available = np.zeros(config.num_booths, dtype=float)
        self.booth_color: list[str | None] = [None] * config.num_booths
        self.total_setup_time = 0.0
        self.total_idle_time = 0.0
        self.total_waiting_time = 0.0
        self.total_tardiness = 0.0
        self.total_weighted_tardiness = 0.0
        self.total_breakdown_time = 0.0
        self.schedule: list[dict[str, Any]] = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        episode_seed = self.config.seed if seed is None else seed
        self.rng = np.random.default_rng(episode_seed)
        if self.external_jobs is None:
            self.jobs = generate_jobs(
                GeneratorConfig(horizon_minutes=self.config.horizon_minutes, seed=episode_seed)
            ).sort_values("arrival_time").reset_index(drop=True)
        else:
            self.jobs = self.external_jobs.sort_values("arrival_time").reset_index(drop=True).copy()

        self.queue_ids = []
        self.completed_ids = []
        self.current_time = 0.0
        self.next_arrival_index = 0
        self.booth_available = np.zeros(self.config.num_booths, dtype=float)
        self.booth_color = [None] * self.config.num_booths
        self.total_setup_time = 0.0
        self.total_idle_time = 0.0
        self.total_waiting_time = 0.0
        self.total_tardiness = 0.0
        self.total_weighted_tardiness = 0.0
        self.total_breakdown_time = 0.0
        self.schedule = []

        self._advance_until_decision()
        return self._observation(), self._info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        if not self.queue_ids:
            raise RuntimeError("No schedulable job is available at the current decision epoch")

        booth = int(np.argmin(self.booth_available))
        self.current_time = max(self.current_time, float(self.booth_available[booth]))
        self._release_arrivals(self.current_time)
        queue = self.jobs[self.jobs["job_id"].isin(self.queue_ids)].copy()
        if queue.empty:
            self._advance_until_decision()
            queue = self.jobs[self.jobs["job_id"].isin(self.queue_ids)].copy()

        job_id = DISPATCH_RULES[int(action)](queue, self.booth_color[booth], self.current_time)
        job = self.jobs.loc[self.jobs["job_id"] == job_id].iloc[0]

        start_time = max(self.current_time, float(self.booth_available[booth]), float(job["arrival_time"]))
        idle_time = max(0.0, start_time - float(self.booth_available[booth]))
        setup_time = _setup_time(self.booth_color[booth], str(job["paint_color"]))
        breakdown_time = self._sample_breakdown()
        process_duration = float(job["processing_time"]) + setup_time + breakdown_time
        paint_finish = start_time + process_duration
        completion_time = paint_finish + float(job["curing_time"]) + float(job["inspection_time"])
        wait_time = start_time - float(job["arrival_time"])
        tardiness = max(0.0, completion_time - float(job["due_time"]))
        weighted_tardiness = tardiness * float(job["priority"])

        self.booth_available[booth] = paint_finish
        self.booth_color[booth] = str(job["paint_color"])
        self.queue_ids.remove(job_id)
        self.completed_ids.append(job_id)
        self.total_setup_time += setup_time
        self.total_idle_time += idle_time
        self.total_waiting_time += wait_time
        self.total_tardiness += tardiness
        self.total_weighted_tardiness += weighted_tardiness
        self.total_breakdown_time += breakdown_time

        self.schedule.append(
            {
                "job_id": int(job_id),
                "booth": booth,
                "paint_color": str(job["paint_color"]),
                "priority": int(job["priority"]),
                "start_time": start_time,
                "paint_finish": paint_finish,
                "completion_time": completion_time,
                "wait_time": wait_time,
                "setup_time": setup_time,
                "breakdown_time": breakdown_time,
                "tardiness": tardiness,
                "rule_action": int(action),
            }
        )

        reward = -(
            0.055 * weighted_tardiness
            + 0.030 * wait_time
            + 0.120 * setup_time
            + 0.025 * idle_time
            + 0.035 * breakdown_time
        )

        self.current_time = float(np.min(self.booth_available))
        self._advance_until_decision()
        terminated = len(self.completed_ids) == len(self.jobs)
        truncated = False

        if terminated:
            reward += 25.0 * self.throughput_rate
            reward += 20.0 * self.on_time_rate

        return self._observation(), float(reward), terminated, truncated, self._info()

    def _sample_breakdown(self) -> float:
        if self.rng.random() >= self.config.breakdown_probability:
            return 0.0
        return float(
            self.rng.uniform(self.config.breakdown_duration_min, self.config.breakdown_duration_max)
        )

    def _release_arrivals(self, until: float) -> None:
        while self.next_arrival_index < len(self.jobs):
            row = self.jobs.iloc[self.next_arrival_index]
            if float(row["arrival_time"]) > until:
                break
            job_id = int(row["job_id"])
            if job_id not in self.completed_ids and job_id not in self.queue_ids:
                self.queue_ids.append(job_id)
            self.next_arrival_index += 1

    def _advance_until_decision(self) -> None:
        while not self.queue_ids and len(self.completed_ids) < len(self.jobs):
            next_arrival = (
                float(self.jobs.iloc[self.next_arrival_index]["arrival_time"])
                if self.next_arrival_index < len(self.jobs)
                else np.inf
            )
            next_booth = float(np.min(self.booth_available))
            if np.isfinite(next_arrival):
                self.current_time = max(self.current_time, min(next_arrival, max(next_booth, self.current_time)))
                if next_arrival > self.current_time and next_booth <= self.current_time:
                    self.current_time = next_arrival
            else:
                self.current_time = max(self.current_time, next_booth)
            self._release_arrivals(self.current_time)
            if not self.queue_ids and np.isfinite(next_arrival) and next_arrival > self.current_time:
                self.current_time = next_arrival
                self._release_arrivals(self.current_time)

    def _observation(self) -> np.ndarray:
        self._release_arrivals(self.current_time)
        queue = self.jobs[self.jobs["job_id"].isin(self.queue_ids)].copy()
        queue_length = len(queue)
        if queue_length:
            waits = np.maximum(0.0, self.current_time - queue["arrival_time"].to_numpy(dtype=float))
            workload = float(queue["processing_time"].sum())
            urgent_ratio = float((queue["priority"] == 3).mean())
            mean_slack = float((queue["due_time"] - self.current_time - queue["processing_time"]).mean())
            overdue_ratio = float((queue["due_time"] < self.current_time).mean())
            min_processing = float(queue["processing_time"].min())
            mean_quality_risk = float(queue["quality_risk"].mean())
            available_booth = int(np.argmin(self.booth_available))
            current_color = self.booth_color[available_booth]
            same_color_ratio = 0.0 if current_color is None else float((queue["paint_color"] == current_color).mean())
            mean_wait = float(np.mean(waits))
            max_wait = float(np.max(waits))
        else:
            workload = urgent_ratio = mean_slack = overdue_ratio = min_processing = 0.0
            mean_quality_risk = same_color_ratio = mean_wait = max_wait = 0.0

        elapsed = max(1.0, self.current_time)
        utilization = float(
            np.clip(
                (sum(item["paint_finish"] - item["start_time"] for item in self.schedule))
                / (self.config.num_booths * elapsed),
                0.0,
                1.0,
            )
        )
        arrival_rate = float(self.next_arrival_index / elapsed)
        completion_rate = float(len(self.completed_ids) / elapsed)
        setup_share = float(self.total_setup_time / max(1.0, elapsed * self.config.num_booths))
        breakdown_share = float(self.total_breakdown_time / max(1.0, elapsed * self.config.num_booths))

        obs = np.array(
            [
                np.clip(queue_length / self.config.max_queue_reference, 0.0, 1.0),
                np.clip(workload / self.config.max_workload_reference, 0.0, 1.0),
                np.clip(mean_wait / self.config.max_wait_reference, 0.0, 1.0),
                np.clip(max_wait / self.config.max_wait_reference, 0.0, 1.0),
                urgent_ratio,
                same_color_ratio,
                np.clip((mean_slack + 180.0) / 360.0, 0.0, 1.0),
                overdue_ratio,
                np.clip(min_processing / 45.0, 0.0, 1.0),
                mean_quality_risk,
                utilization,
                np.clip(arrival_rate / 0.5, 0.0, 1.0),
                np.clip(completion_rate / 0.5, 0.0, 1.0),
                np.clip(setup_share + breakdown_share, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return obs

    @property
    def throughput_rate(self) -> float:
        makespan = max([item["completion_time"] for item in self.schedule], default=1.0)
        return float(len(self.completed_ids) / max(1.0, makespan))

    @property
    def on_time_rate(self) -> float:
        if not self.schedule:
            return 0.0
        return float(np.mean([item["tardiness"] <= 1e-9 for item in self.schedule]))

    def _info(self) -> dict[str, Any]:
        n = max(1, len(self.completed_ids))
        return {
            "time": self.current_time,
            "queue_length": len(self.queue_ids),
            "completed_jobs": len(self.completed_ids),
            "total_jobs": len(self.jobs),
            "mean_waiting_time": self.total_waiting_time / n,
            "total_tardiness": self.total_tardiness,
            "weighted_tardiness": self.total_weighted_tardiness,
            "setup_time": self.total_setup_time,
            "idle_time": self.total_idle_time,
            "breakdown_time": self.total_breakdown_time,
            "throughput_rate": self.throughput_rate,
            "on_time_rate": self.on_time_rate,
        }

    def schedule_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.schedule)
