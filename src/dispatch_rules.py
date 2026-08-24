from __future__ import annotations

from collections.abc import Callable
import pandas as pd


def _setup_time(current_color: str | None, next_color: str) -> float:
    if current_color is None or current_color == next_color:
        return 0.0
    light = {"white", "silver", "gray"}
    dark = {"blue", "red", "black"}
    if current_color in dark and next_color in light:
        return 10.0
    if current_color in light and next_color in dark:
        return 6.0
    return 4.0


def fifo(queue: pd.DataFrame, current_color: str | None, now: float) -> int:
    return int(queue.sort_values(["arrival_time", "job_id"]).iloc[0]["job_id"])


def highest_priority(queue: pd.DataFrame, current_color: str | None, now: float) -> int:
    return int(queue.sort_values(["priority", "due_time", "arrival_time"], ascending=[False, True, True]).iloc[0]["job_id"])


def earliest_due_date(queue: pd.DataFrame, current_color: str | None, now: float) -> int:
    return int(queue.sort_values(["due_time", "priority", "arrival_time"], ascending=[True, False, True]).iloc[0]["job_id"])


def shortest_processing_time(queue: pd.DataFrame, current_color: str | None, now: float) -> int:
    return int(queue.sort_values(["processing_time", "due_time"]).iloc[0]["job_id"])


def same_color_first(queue: pd.DataFrame, current_color: str | None, now: float) -> int:
    if current_color is not None:
        same = queue[queue["paint_color"] == current_color]
        if not same.empty:
            return earliest_due_date(same, current_color, now)
    return earliest_due_date(queue, current_color, now)


def minimum_changeover(queue: pd.DataFrame, current_color: str | None, now: float) -> int:
    scored = queue.assign(
        _setup=queue["paint_color"].map(lambda color: _setup_time(current_color, color)),
        _slack=queue["due_time"] - now - queue["processing_time"],
    )
    return int(scored.sort_values(["_setup", "_slack", "priority"], ascending=[True, True, False]).iloc[0]["job_id"])


def critical_ratio(queue: pd.DataFrame, current_color: str | None, now: float) -> int:
    work = (queue["processing_time"] + queue["curing_time"] + queue["inspection_time"]).clip(lower=1.0)
    ratio = (queue["due_time"] - now) / work
    scored = queue.assign(_critical_ratio=ratio)
    return int(scored.sort_values(["_critical_ratio", "priority"], ascending=[True, False]).iloc[0]["job_id"])


def weighted_rule(queue: pd.DataFrame, current_color: str | None, now: float) -> int:
    max_wait = max(1.0, float((now - queue["arrival_time"]).max()))
    max_proc = max(1.0, float(queue["processing_time"].max()))
    max_lateness_pressure = max(1.0, float((now - queue["due_time"]).abs().max()))
    scored = queue.copy()
    scored["_score"] = (
        2.5 * scored["priority"]
        + 1.6 * ((now - scored["arrival_time"]) / max_wait)
        - 1.0 * (scored["processing_time"] / max_proc)
        - 1.2 * scored["paint_color"].map(lambda color: _setup_time(current_color, color)) / 10.0
        + 1.8 * ((now - scored["due_time"]) / max_lateness_pressure)
    )
    return int(scored.sort_values(["_score", "due_time"], ascending=[False, True]).iloc[0]["job_id"])


DISPATCH_RULES: tuple[Callable[[pd.DataFrame, str | None, float], int], ...] = (
    fifo,
    highest_priority,
    earliest_due_date,
    shortest_processing_time,
    same_color_first,
    minimum_changeover,
    critical_ratio,
    weighted_rule,
)

RULE_NAMES = tuple(rule.__name__ for rule in DISPATCH_RULES)
