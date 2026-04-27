# UAV-assisted LLM Inference Optimization using PPO

## 1. Overview

This project studies a UAV-assisted edge intelligence system for supporting Large Language Model (LLM) inference services in communication-constrained environments. In the considered scenario, a UAV acts as an aerial relay between ground users and an edge server. The UAV can adjust its position and service configuration to improve the communication channel, reduce transmission latency, and support LLM inference tasks.

The optimization problem is formulated as a reinforcement learning problem. A Gymnasium-based environment is developed, and the Proximal Policy Optimization (PPO) algorithm is applied to learn an adaptive control policy for the UAV.

The main objective is to minimize the overall system cost while satisfying constraints related to latency, energy consumption, UAV mobility, and LLM service quality.

---

## 2. Research Motivation

Large Language Models require significant computation and communication resources. In areas where communication infrastructure is weak or unstable, users may experience high latency when accessing LLM-based services.

UAV-assisted communication can help improve service coverage by acting as a relay node. However, UAVs have limited battery capacity and must carefully control their movement, energy consumption, and service configuration.

Therefore, this project investigates how reinforcement learning can be used to jointly optimize UAV mobility, communication quality, energy consumption, and LLM inference configuration.

---

## 3. Main Contributions

The main contributions of this project are:

1. A UAV-assisted LLM inference system is modeled with users, UAV relay, edge server, and system constraints.
2. A Gymnasium-compatible reinforcement learning environment is designed for the UAV-LLM optimization problem.
3. Continuous observation and action spaces are constructed to represent the system state and UAV control decisions.
4. PPO is applied to learn an adaptive policy for UAV movement and resource/configuration selection.
5. The trained PPO policy is evaluated using metrics such as reward, latency, PPL, energy consumption, feasible rate, and constraint violation.
6. PPO is compared with baseline policies such as Random Policy, Hover Policy, and Greedy Policy.

---

## 4. System Model

The system consists of the following components:

### 4.1 Ground Users

Ground users generate LLM inference requests. Each request may include:

- Input tokens.
- Prompt size.
- Output tokens.
- Data size.
- Latency requirement.
- Service quality requirement.

In the simulation, user locations are randomly generated within a predefined 2D area.

### 4.2 UAV Relay

The UAV acts as an aerial relay between ground users and the edge server. It can move within the service area and adjust its behavior according to the current system state.

The UAV is characterized by:

- Position.
- Velocity.
- Remaining energy.
- Flight power consumption.
- Communication distance to users.
- Channel quality.
- Service configuration.

### 4.3 Edge Server

The edge server performs LLM inference tasks. The processing latency depends on:

- Input data size.
- Number of input tokens.
- Number of output tokens.
- Selected LLM configuration.
- Available computation resource.

### 4.4 Energy Model

The UAV consumes energy during movement and communication. The remaining energy of the UAV must remain above a minimum threshold to ensure safe operation.

In the current implementation, the energy model mainly considers UAV energy consumption. The laser charging component can be extended in future versions if a detailed charging model is added to the environment.

---

## 5. Reinforcement Learning Formulation

The UAV control problem is formulated as a Markov Decision Process.

At each time slot, the agent observes the current system state, selects an action, and receives a reward from the environment.

The reinforcement learning formulation includes:

- State space.
- Action space.
- Reward function.
- Transition function.
- Constraint penalty.
- Episode termination condition.

---

## 6. Environment Design

The environment is implemented using Gymnasium.

A typical environment class structure is:

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class UAVLLMEnv(gym.Env):
    def __init__(self, n_users=10, seed=None):
        super().__init__()

        self.n_users = n_users
        self.rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )

        self._init_episode()

    def _init_episode(self):
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_episode()
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        pass

    def _get_obs(self):
        pass
