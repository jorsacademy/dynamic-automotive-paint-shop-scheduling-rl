from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


PAINT_COLORS = ("white", "silver", "gray", "blue", "red", "black")
VEHICLE_MODELS = ("compact", "sedan", "suv", "van")


@dataclass(frozen=True)
class GeneratorConfig:
    horizon_minutes: int = 8 * 60
    mean_interarrival: float = 4.0
    urgent_probability: float = 0.12
    rework_probability: float = 0.04
    seed: int = 42


def generate_jobs(config: GeneratorConfig = GeneratorConfig()) -> pd.DataFrame:
    """Generate a realistic stochastic stream of automotive paint jobs.

    Arrival times follow an exponential inter-arrival process. Processing and curing
    times depend on vehicle model and paint color, while due dates are generated
    from workload-aware slack. The resulting table is suitable for replay as a
    dynamic event stream inside the RL environment.
    """
    rng = np.random.default_rng(config.seed)
    arrivals: list[float] = []
    t = 0.0
    while t < config.horizon_minutes:
        t += rng.exponential(config.mean_interarrival)
        if t < config.horizon_minutes:
            arrivals.append(t)

    n = len(arrivals)
    models = rng.choice(VEHICLE_MODELS, size=n, p=[0.26, 0.34, 0.28, 0.12])
    colors = rng.choice(PAINT_COLORS, size=n, p=[0.24, 0.20, 0.18, 0.14, 0.10, 0.14])
    urgent = rng.random(n) < config.urgent_probability

    model_base = {"compact": 18.0, "sedan": 21.0, "suv": 25.0, "van": 29.0}
    color_factor = {"white": 0.95, "silver": 1.00, "gray": 1.00, "blue": 1.04, "red": 1.07, "black": 1.05}

    processing = np.array([
        max(10.0, rng.normal(model_base[m] * color_factor[c], 2.5))
        for m, c in zip(models, colors)
    ])
    curing = np.array([
        max(18.0, rng.normal(31.0 + (4.0 if c in {"red", "black"} else 0.0), 3.0))
        for c in colors
    ])
    inspection = np.maximum(3.0, rng.normal(6.0, 1.2, size=n))
    priority = np.where(urgent, 3, rng.choice([1, 2], size=n, p=[0.78, 0.22]))
    due_slack = rng.uniform(70.0, 145.0, size=n) - urgent.astype(float) * rng.uniform(20.0, 45.0, size=n)
    due_time = np.asarray(arrivals) + processing + curing + inspection + np.maximum(30.0, due_slack)

    rework = rng.random(n) < config.rework_probability
    quality_risk = np.clip(rng.beta(2.0, 12.0, size=n) + rework.astype(float) * 0.25, 0.0, 1.0)

    jobs = pd.DataFrame(
        {
            "job_id": np.arange(n, dtype=int),
            "arrival_time": np.round(arrivals, 3),
            "vehicle_model": models,
            "paint_color": colors,
            "priority": priority.astype(int),
            "processing_time": np.round(processing, 3),
            "curing_time": np.round(curing, 3),
            "inspection_time": np.round(inspection, 3),
            "due_time": np.round(due_time, 3),
            "quality_risk": np.round(quality_risk, 4),
            "rework_required": rework.astype(int),
        }
    )
    validate_jobs(jobs)
    return jobs


def validate_jobs(jobs: pd.DataFrame) -> None:
    required = {
        "job_id", "arrival_time", "vehicle_model", "paint_color", "priority",
        "processing_time", "curing_time", "inspection_time", "due_time",
        "quality_risk", "rework_required",
    }
    missing = required.difference(jobs.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if jobs["job_id"].duplicated().any():
        raise ValueError("job_id values must be unique")
    if (jobs["arrival_time"] < 0).any():
        raise ValueError("arrival_time must be non-negative")
    for col in ("processing_time", "curing_time", "inspection_time"):
        if (jobs[col] <= 0).any():
            raise ValueError(f"{col} must be strictly positive")
    if (jobs["due_time"] < jobs["arrival_time"]).any():
        raise ValueError("due_time must not precede arrival_time")
    if not jobs["priority"].isin([1, 2, 3]).all():
        raise ValueError("priority must be one of 1, 2, 3")
    if not jobs["quality_risk"].between(0.0, 1.0).all():
        raise ValueError("quality_risk must be in [0, 1]")
