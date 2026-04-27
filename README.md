# UAV-assisted LLM Inference Optimization using PPO

## 1. Project Overview

This project studies the optimization of a UAV-assisted Large Language Model (LLM) inference system using reinforcement learning. In the considered system, an Unmanned Aerial Vehicle (UAV) acts as an aerial relay to support communication between ground users and an edge server that provides LLM inference services.

The main goal is to optimize the UAV control policy and system configuration in order to reduce latency, improve service quality, reduce energy consumption, and satisfy system constraints.

The optimization problem is formulated as a reinforcement learning problem. A custom Gymnasium environment is developed, and the Proximal Policy Optimization (PPO) algorithm is used to learn an adaptive control policy.

---

## 2. Motivation

Large Language Models require large computation and communication resources. When users access LLM services in areas with weak communication infrastructure, the system may suffer from high transmission delay, unstable connection quality, and poor service performance.

UAV-assisted communication can improve coverage and communication quality by placing an aerial relay closer to users. However, UAVs have limited battery capacity and cannot move arbitrarily without considering energy consumption and physical constraints.

Therefore, this project investigates how reinforcement learning can be used to optimize UAV movement, communication quality, LLM configuration, and system constraints at the same time.

---

## 3. Main Objectives

The main objectives of this project are:

1. Build a simulation environment for UAV-assisted LLM inference systems.
2. Model the relationship between UAV position, channel quality, latency, energy consumption, and LLM service quality.
3. Formulate the optimization problem as a reinforcement learning task.
4. Design a Gymnasium-compatible environment for PPO training.
5. Apply PPO to learn a control policy for UAV movement and system configuration.
6. Evaluate PPO using reward, latency, PPL, energy consumption, feasible rate, and constraint violation.
7. Compare PPO with baseline policies such as Random Policy, Hover Policy, and Greedy Policy.

---

## 4. System Model

The system consists of four main components:

1. Ground users.
2. UAV relay.
3. Edge server.
4. Energy system.

### 4.1 Ground Users

Ground users generate LLM inference requests. Each user request may include an input prompt and expects an output response from the LLM system.

In the simulation, each task may be represented by:

- Number of input tokens.
- Input data size.
- Number of output tokens.
- Output data size.
- Latency requirement.
- Service quality requirement.

The positions of users are randomly generated within a predefined service area.

---

### 4.2 UAV Relay

The UAV acts as an aerial relay between ground users and the edge server. The UAV can move in the horizontal plane while maintaining a fixed altitude.

The UAV state includes:

- UAV position.
- UAV velocity.
- Remaining energy.
- Distance to users.
- Channel quality.
- Current time slot.
- System constraint status.

The UAV needs to decide how to move and how to configure the system in order to improve the overall service performance.

---

### 4.3 Edge Server

The edge server performs LLM inference tasks. The processing latency depends on:

- Input token size.
- Output token size.
- Selected LLM configuration.
- Available computing resource.
- Model complexity.

In this project, the LLM inference process is simplified to make the problem suitable for simulation and reinforcement learning training.

---

### 4.4 Energy System

The UAV consumes energy during movement and operation. The remaining energy must stay above a minimum threshold to ensure safe operation.

The current implementation mainly considers UAV energy consumption. If the code does not explicitly calculate harvested laser energy, laser charging should be treated as a future extension rather than a completed implementation.

---

## 5. Reinforcement Learning Formulation

The UAV-assisted LLM inference problem is formulated as a Markov Decision Process.

At each time slot:

1. The environment provides the current system state.
2. The PPO agent selects an action.
3. The UAV updates its position and system configuration.
4. The environment calculates channel quality, latency, PPL, energy consumption, and constraint violation.
5. A reward is returned to the agent.
6. The next state is generated.

The goal of the agent is to maximize the cumulative reward over an episode.

Since the reward is defined based on negative system cost and penalty, maximizing reward is equivalent to minimizing the total system cost.

---

## 6. Environment Design

The environment is implemented using Gymnasium.

A typical environment class has the following structure:

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
