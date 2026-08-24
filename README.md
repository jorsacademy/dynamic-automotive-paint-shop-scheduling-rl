# Dynamic Automotive Paint Shop Scheduling with Reinforcement Learning

This project implements a dynamic, event-driven scheduling environment for an automotive paint shop and trains a reinforcement learning agent to select dispatching policies in real time.

The system is designed around operational scheduling rather than static job assignment. Jobs arrive stochastically during the episode, paint booths become available at different times, color changes create sequence-dependent setup costs, and random breakdowns introduce additional uncertainty. The RL agent observes the current shop state and chooses the dispatching rule to apply at each decision epoch.

## Objective

The scheduling objective is to improve operational performance across several competing KPIs:

- reduce weighted tardiness and late completions;
- reduce queue waiting time;
- reduce sequence-dependent paint color changeover time;
- reduce booth idle time;
- maintain high throughput and on-time completion;
- remain adaptive to stochastic arrivals and random disruptions.

The reward function combines weighted tardiness, waiting time, setup time, idle time, and breakdown penalties. Throughput and on-time completion bonuses are applied at the end of each episode.

## Reinforcement Learning Formulation

### State

The observation is a fixed-size 14-dimensional numerical vector describing the current production state. It includes queue length, queued workload, waiting-time statistics, urgent-job ratio, same-color opportunity, slack, overdue ratio, processing-time characteristics, quality risk, booth utilization, arrival rate, completion rate, and disruption/setup load.

All state variables are normalized into a bounded `[0, 1]` observation space.

### Action

The action space contains eight dispatching policies:

1. FIFO
2. Highest priority first
3. Earliest due date
4. Shortest processing time
5. Same color first
6. Minimum changeover
7. Critical ratio
8. Weighted composite dispatch rule

The RL agent therefore learns when a scheduling heuristic should be used instead of attempting to select one job from an unbounded or changing job-ID action space.

### Environment Dynamics

At each decision epoch:

1. newly arrived jobs are released into the queue;
2. the next available paint booth is identified;
3. the RL agent selects a dispatching rule;
4. that rule selects a feasible job from the current queue;
5. color-dependent setup time is calculated;
6. a stochastic booth breakdown may occur;
7. the job is processed, cured, and inspected;
8. operational KPIs and reward are updated;
9. simulation advances to the next scheduling decision.

This produces a dynamic scheduling problem with a fixed observation and action structure that is compatible with Gymnasium and Stable-Baselines3.

## Synthetic Data

Because no factory dataset is required, each episode can generate a new stochastic production stream. Synthetic data includes:

- job arrival time;
- vehicle model;
- paint color;
- priority;
- paint processing time;
- curing time;
- inspection time;
- due date;
- quality risk;
- rework indicator.

Inter-arrival times follow an exponential process, while processing parameters depend on vehicle and paint characteristics. Validation rules prevent impossible timestamps, duplicate job identifiers, negative durations, invalid priority values, and invalid quality-risk values.

## Project Structure

```text
dynamic-automotive-paint-shop-scheduling-rl/
├── README.md
├── LICENSE.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── data_generator.py
│   ├── dispatch_rules.py
│   ├── environment.py
│   ├── train.py
│   └── evaluate.py
└── tests/
    ├── test_data_generator.py
    └── test_environment.py
```

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the PPO Agent

Run from the repository root:

```bash
python -m src.train
```

The default training configuration uses PPO with a two-layer MLP policy and trains for 150,000 environment steps.

The trained model is written to:

```text
models/ppo_paint_shop.zip
```

## Evaluate Against Scheduling Baselines

After training:

```bash
python -m src.evaluate
```

The evaluation compares PPO against all eight fixed dispatching policies using identical synthetic job streams. Results are written to:

```text
results/benchmark.csv
```

Reported metrics include:

- mean waiting time;
- total tardiness;
- priority-weighted tardiness;
- total setup time;
- idle time;
- breakdown time;
- throughput rate;
- on-time completion rate.

The benchmark design is important: RL performance should be judged against strong deterministic scheduling heuristics rather than by training reward alone.

## Run Tests

```bash
pytest -q
```

The test suite validates synthetic data generation, checks Gymnasium/Stable-Baselines3 API compatibility, verifies bounded observations, and confirms that complete schedules contain each job exactly once.

## Design Rationale

A direct DQN formulation with one action per `(job, booth)` assignment is unsuitable for this problem because the number of jobs changes over time and most such actions are infeasible at any given decision epoch. This implementation uses RL as a hyper-heuristic controller: PPO learns which dispatching policy to apply to the current production state.

This preserves the advantages of reinforcement learning while maintaining a realistic, scalable scheduling action space.

## Limitations

This repository is a simulation and research implementation, not a production manufacturing-control system. The synthetic distributions, reward weights, changeover times, failure process, and process-flow assumptions should be calibrated with real operational data before industrial use.

## License

This project is released under a custom Non-Commercial Research and Educational License. Commercial use is prohibited without a separate written license. See `LICENSE.md` for the full terms.
