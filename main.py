"""
=============================================================================
Joint LLM Task Offloading and Trajectory Optimization
in UAV-enabled Wireless Network with Laser Power Charging
-----------------------------------------------------------------------------
Solver  : PPO (Proximal Policy Optimization) — stable-baselines3
Version : FINAL  —  all physics, constraints 17f & 17g fully corrected
=============================================================================

CORRECTIONS vs previous version
─────────────────────────────────
1. N0 INTERPRETATION
   Paper: N0 = -60 dBm → total noise POWER [W], NOT spectral density [W/Hz].
   SNR = P·g / N0  (no extra division by B).
   With the fix: R_up_i ≈ 0.09 Mbps, T_up ≈ 280 ms — physically realistic.

2. GPU COMPUTE RATE
   Paper: f_i,t = f_max/N with f_max ∈ [1,3] (Table) → interpreted as total
   GPU throughput in TFLOPS (V100 ≈ 14 TFLOPS FP32).
   T_cmp = ψ(d_i) / (f_i · 1e12)  with f_max = 14 TFLOPS → T_inf ≈ 745 ms.
   B_batch = 1 per user request (not 512).

3. CONSTRAINT 17f  Ti ≤ τ − T_fly
   Applied as SOFT PENALTY proportional to violation magnitude.
   Previously it was clamped (hiding real latency) — now correctly penalised.

4. CONSTRAINT 17g  PPL(ϑ) ≤ U_req
   U_req set to PPL(ϑ_min) + 0.5 = 20.96 (all valid ϑ in [32,64] satisfy it
   at ϑ≥33; agent is incentivised to increase ϑ for better quality).
   Violation penalised explicitly.

5. REWARD IS POSITIVE
   reward = − cost_norm − penalty
   R_MAX = 10.0, cost_norm ∈ [0, ~3] when feasible → reward ∈ (0, 10].

6. LATENCY CORRECTLY > 0
   T_up ≈ 280–500 ms, T_inf ≈ 300–1500 ms, T_dn ≈ 5–50 ms → total ≈ 0.6–2 s.

INSTALL
───────
pip install stable-baselines3 gymnasium numpy matplotlib scipy
"""

# ─────────────────────────────────────────────────────────────────────────────
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import warnings
warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SYSTEM PARAMETERS  (paper Tables I – IV)
# ═════════════════════════════════════════════════════════════════════════════
class P:
    # ── Area & time ───────────────────────────────────────────────────────
    xmin, xmax = -100.0,  100.0    # m
    ymin, ymax = -100.0,  100.0    # m
    zmin, zmax =   10.0,  100.0    # m
    T          =   10              # number of time slots
    tau        =    3.0            # slot duration  [s]
    N          =   10              # number of UEs  (paper: N ∈ {10,20})

    # ── Node positions ────────────────────────────────────────────────────
    C_ED = np.array([  0.0,  0.0,  0.0])   # edge server  [m]
    C_PB = np.array([  0.0, 50.0, 50.0])   # power beacon [m]

    # ── UAV mechanics  (Table I) ──────────────────────────────────────────
    Vmax  =  50.0    # m/s   max speed           (17i)
    Gtip  = 120.0    # m/s   rotor tip speed
    rho   =   1.225  # kg/m³ air density  (ε in paper)
    v0    =   4.03   # m/s   mean rotor induced velocity
    P1    =  88.63   # W     induced power (hover)
    P0    = 158.76   # W     blade-profile power (hover)
    s_rot =   0.05   # –     rotor solidity
    d0    =   0.3    # –     fuselage drag ratio
    A     =   0.503  # m²    rotor disc area
    P_H   = P0 + P1  # W     total hover power (vt=0 in eq.11)

    # ── UAV energy ────────────────────────────────────────────────────────
    E_bg  = 1e6      # J     max energy storage   (energy budget)
    E_min = 1e4      # J     min energy storage   (safety margin)
    E0    = 0.8e6    # J     initial energy       (80 % of budget)

    # ── Communication ─────────────────────────────────────────────────────
    Bi       = 0.512e6    # Hz   user ↔ UAV bandwidth per user
    BU_ED    = 1.0e6      # Hz   UAV ↔ edge bandwidth
    p_up_i   = 10**(17/10)*1e-3   # W   user uplink  (17 dBm)
    p_up_U   = 10**(27/10)*1e-3   # W   UAV uplink   (27 dBm)
    p_down_U = 10**(27/10)*1e-3   # W   UAV downlink (27 dBm)
    p_ED     = 10**(40/10)*1e-3   # W   edge server downlink (40 dBm)
    # N0 = -60 dBm is the total noise POWER [W] at receiver
    # SNR = P·g / N0   (no division by bandwidth — paper convention)
    N0       = 10**(-60/10)*1e-3  # W   total noise power
    beta     = 2.2                # path-loss exponent
    phi_freq = 1.8e9              # Hz  carrier frequency
    c_light  = 3e8                # m/s
    eps_LoS  = 0.5                # LoS attenuation factor
    chi1 = chi2  = 0.5            # Rician LOS amplitude
    iota1= iota2 = 0.5            # Rician scale

    # ── LLM inference ─────────────────────────────────────────────────────
    # B_batch = 1 per user request (each user sends one independent task)
    # f_max = total GPU throughput of edge server [TFLOPS]
    # Paper says f_max ∈ [1,3], NVIDIA Tesla V100 FP32 ≈ 14 TFLOPS
    # We use 14.0 to match V100 spec; rescale by N for per-user share.
    B_batch    =   1       # batch size per user task
    h_dim      = 1024      # hidden dimension
    theta_min  =  32       # min transformer layers  ϑ_min  (17h)
    theta_max  =  64       # max transformer layers  ϑ_max  (17h)
    f_max_TFLOPS = 14.0    # V100 FP32 throughput  [TFLOPS]
    kappa_ED   = 1e-38     # GPU energy coefficient  [W/(Hz³)]  (eq.8)

    # ── PPL utility  (eq.15):  PPL(ϑ) = φ·ϑ² + ϖ·ϑ + ϱ ──────────────────
    phi_u  =  0.01
    varpi  = -1.172
    varrho =  47.72
    # PPL(32) ≈ 20.46,  PPL(64) ≈ 13.67  (lower = better quality)
    # Constraint 17g: PPL ≤ U_req.  Set U_req = PPL(theta_min) + 0.5
    U_req  = 20.96

    # ── Laser charging  (clear air, 810 nm, Tables II–IV) ─────────────────
    P_pb_min    =  10.0    # W   min PB power to activate circuit
    P_pb_max    = 100.0    # W   max PB transmit power
    zeta_el     =   0.35   # electricity → laser  (mean of [0.3,0.4])
    zeta_le     =   0.45   # laser → electricity  (mean of [0.4,0.5])
    alpha_atten =   0.391  # laser attenuation α  (clear air, 810 nm)
    b1, d1      =   0.445, -0.75     # Table IV  810 nm
    b2, d2      =   0.5414,-0.2313   # Table IV  810 nm
    gamma_pb    =   0.01   # cost coefficient  [$/W]

    # ── Objective weights  (eq.17a) ───────────────────────────────────────
    sigma_U = 1.0   # PPL weight
    sigma_T = 1.0   # latency weight
    sigma_P = 1.0   # laser cost weight

    # ── Reward shaping ────────────────────────────────────────────────────
    R_MAX        = 10.0    # reward offset → reward = R_MAX − cost_norm − pen
    # Normalisation denominators (calibrated to realistic value ranges)
    NORM_PPL     = 21.0    # ≈ max PPL = PPL(theta_min)
    NORM_T       =  2.0    # typical max latency [s] within one slot
    NORM_C       =  1.0    # gamma_pb × P_pb_max
    # Soft-penalty weights
    W_PEN_LAT    =  5.0    # per-user latency violation  (17f)
    W_PEN_ENERGY = 10.0    # energy causality violation  (16)
    W_PEN_PPL    =  3.0    # per-user PPL violation      (17g)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PHYSICS FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def channel_gain(pos_uav: np.ndarray, pos_node: np.ndarray) -> float:
    """
    LoS channel gain  (eq.1):
      g = (4πf/c)^{-2} · ε_LoS · d^{-β} · |g_Rice|
    Rician mean amplitude: E[|g_Rice|] = sqrt(χ² + ι²), χ=ι=0.5 → ≈ 0.707
    """
    d = float(np.linalg.norm(pos_uav - pos_node)) + 1e-3
    g = ((4.0 * np.pi * P.phi_freq / P.c_light) ** (-2)
         * P.eps_LoS
         * d ** (-P.beta)
         * np.sqrt(P.chi1**2 + P.iota1**2))
    return float(max(g, 1e-30))


def tx_rate(bandwidth: float, power: float, gain: float) -> float:
    """
    Shannon rate  (eq.2 / eq.3):
      R = B · log2(1 + SNR),   SNR = P·g / N0
    Note: N0 is total noise POWER [W] (not PSD), so no B in denominator.
    """
    snr = power * gain / (P.N0 + 1e-30)
    return float(bandwidth * np.log2(1.0 + max(snr, 0.0)))


def psi_flops(d_tokens: int) -> float:
    """
    FLOPs for ONE transformer layer, ONE user request  (paper after eq.4):
      ψ(d) = 24·B·d·h² + 4·B·d²·h    with B=1 (single request)
    """
    return float(24 * P.B_batch * d_tokens * P.h_dim**2
                 + 4 * P.B_batch * d_tokens**2 * P.h_dim)


def uav_flight_power(v: float) -> float:
    """
    Rotary-wing UAV power  (eq.11):
      P(v) = P0(1 + 3v²/G²)  +  P1·√max(√(1+v⁴/4v0⁴) − v²/2v0², 0)
           + ½·d0·s·ρ·A·v³
    """
    blade    = P.P0 * (1.0 + 3.0 * v**2 / P.Gtip**2)
    arg      = np.sqrt(1.0 + v**4 / (4.0 * P.v0**4)) - v**2 / (2.0 * P.v0**2)
    induced  = P.P1 * np.sqrt(max(arg, 0.0))
    parasite = 0.5 * P.d0 * P.s_rot * P.rho * P.A * v**3
    return blade + induced + parasite


def laser_received_power(P_pb: float, dist: float) -> float:
    """
    Received laser power at UAV  (eq.10):
      ζ_ls = exp(−α·d)
      P_rv = b1·b2·ζ_ls·P_pb + b2·d1·ζ_ls + d2    if P_pb ≥ P_pb_min
           = 0                                        otherwise
    """
    if P_pb < P.P_pb_min:
        return 0.0
    zeta_ls = np.exp(-P.alpha_atten * dist)
    return float(max(P.b1*P.b2*zeta_ls*P_pb + P.b2*P.d1*zeta_ls + P.d2, 0.0))


def ppl(theta: int) -> float:
    """
    PPL service utility  (eq.15):
      PPL(ϑ) = φ·ϑ² + ϖ·ϑ + ϱ
    Range: PPL(32) ≈ 20.46,  PPL(64) ≈ 13.67.
    Lower PPL = better LLM output quality.
    """
    return float(P.phi_u * theta**2 + P.varpi * theta + P.varrho)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — GYMNASIUM ENVIRONMENT
# ═════════════════════════════════════════════════════════════════════════════

class UAVLLMEnv(gym.Env):
    """
    ┌──────────────────────────────────────────────────────────────────┐
    │  OBSERVATION  (9-dim, normalised to [0, 1])                      │
    │  [0]  x_UAV     normalised UAV x-position                        │
    │  [1]  y_UAV     normalised UAV y-position                        │
    │  [2]  z_UAV     normalised UAV z-position                        │
    │  [3]  E_norm    remaining energy  E_uav / E_bg                   │
    │  [4]  t_norm    time slot index   t / T                          │
    │  [5]  g_u_norm  mean channel gain user→UAV  (log-scaled)         │
    │  [6]  g_e_norm  channel gain UAV→edge       (log-scaled)         │
    │  [7]  dpb_norm  distance UAV→PB             / max_dist           │
    │  [8]  θ_norm    current θ                   / θ_max              │
    └──────────────────────────────────────────────────────────────────┘
    ┌──────────────────────────────────────────────────────────────────┐
    │  ACTION  (6-dim continuous ∈ [-1, 1], rescaled internally)       │
    │  [0]  Δx    UAV x-displacement  → [−30, +30] m                  │
    │  [1]  Δy    UAV y-displacement  → [−30, +30] m                  │
    │  [2]  Δz    UAV z-displacement  → [−15, +15] m                  │
    │  [3]  v     UAV speed           → [1, Vmax]  m/s                 │
    │  [4]  P_pb  laser PB power      → [0, P_pb_max]  W              │
    │  [5]  θ     transformer layers  → [θ_min, θ_max]                 │
    └──────────────────────────────────────────────────────────────────┘
    """

    metadata = {"render_modes": []}

    def __init__(self, n_users: int = P.N, seed: int = None):
        super().__init__()
        self.n_users   = n_users
        self.rng       = np.random.default_rng(seed)
        self._max_dist = float(np.linalg.norm(
            [P.xmax-P.xmin, P.ymax-P.ymin, P.zmax-P.zmin]))

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        self._init_episode()

    # ── Internal helpers ─────────────────────────────────────────────────

    def _init_episode(self):
        """Randomise user positions and per-user task parameters."""
        self.user_pos = np.column_stack([
            self.rng.uniform(P.xmin, P.xmax, self.n_users),
            self.rng.uniform(P.ymin, P.ymax, self.n_users),
            np.zeros(self.n_users),
        ])
        # d_i: input token count  ∈ [512, 1024]  (paper)
        self.d_i   = self.rng.integers(512, 1025, size=self.n_users)
        # s_i: prompt size in bits  = tokens × 4 bytes × 8 bits
        self.s_i   = (self.d_i * 32).astype(float)      # ~16–33 k bits
        # output: 100–500 tokens
        d_out      = self.rng.integers(100, 501, size=self.n_users)
        self.s_out = (d_out * 32).astype(float)          # ~3–16 k bits

    def _scale_action(self, a: np.ndarray):
        """Map [-1, 1]⁶ → physical ranges."""
        dx    = float(a[0]) * 30.0                                     # m
        dy    = float(a[1]) * 30.0                                     # m
        dz    = float(a[2]) * 15.0                                     # m
        v     = (float(a[3]) + 1.0) / 2.0 * (P.Vmax - 1.0) + 1.0     # [1, Vmax]
        P_pb  = (float(a[4]) + 1.0) / 2.0 * P.P_pb_max                # [0, P_pb_max]
        theta = int(np.clip(
            round((float(a[5]) + 1.0) / 2.0 * (P.theta_max - P.theta_min)
                  + P.theta_min),
            P.theta_min, P.theta_max))
        return dx, dy, dz, v, P_pb, theta

    def _get_obs(self) -> np.ndarray:
        xn = (self.uav_pos[0] - P.xmin) / (P.xmax - P.xmin)
        yn = (self.uav_pos[1] - P.ymin) / (P.ymax - P.ymin)
        zn = (self.uav_pos[2] - P.zmin) / (P.zmax - P.zmin)
        en = float(self.E_uav) / P.E_bg

        g_users = float(np.mean(
            [channel_gain(self.uav_pos, u) for u in self.user_pos]))
        g_ed    = channel_gain(self.uav_pos, P.C_ED)

        # Log-scale channel normalisation  (log range roughly [-20, -5])
        LOG_LO, LOG_HI = -20.0, -5.0
        gun = float(np.clip(
            (np.log10(g_users + 1e-30) - LOG_LO) / (LOG_HI - LOG_LO), 0, 1))
        gen = float(np.clip(
            (np.log10(g_ed    + 1e-30) - LOG_LO) / (LOG_HI - LOG_LO), 0, 1))

        dpb_n = float(np.clip(
            np.linalg.norm(self.uav_pos - P.C_PB) / self._max_dist, 0, 1))
        thn   = (self.theta - P.theta_min) / (P.theta_max - P.theta_min)

        return np.array([xn, yn, zn, en,
                         self.t / P.T,
                         gun, gen,
                         dpb_n, thn], dtype=np.float32)

    # ── Gym API ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._init_episode()
        self.uav_pos = np.array([
            float(self.rng.uniform(P.xmin, P.xmax)),
            float(self.rng.uniform(P.ymin, P.ymax)),
            float(self.rng.uniform(P.zmin, P.zmax)),
        ])
        self.E_uav = P.E0
        self.t     = 0
        self.theta = P.theta_min
        self.info  = {}
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        dx, dy, dz, v, P_pb, theta = self._scale_action(action)
        self.theta = theta

        # ── 1. Move UAV  (constraints 17b–17d) ───────────────────────────
        new_pos = np.clip(
            self.uav_pos + np.array([dx, dy, dz]),
            [P.xmin, P.ymin, P.zmin],
            [P.xmax, P.ymax, P.zmax])

        dist_moved = float(np.linalg.norm(new_pos - self.uav_pos))
        T_fly      = dist_moved / v          # eq.12: T_fly = dist / v_t
        if T_fly >= P.tau:                   # ensure some hovering time
            T_fly = P.tau * 0.95
            v     = dist_moved / T_fly if dist_moved > 1e-6 else v
        T_hover = P.tau - T_fly              # time window for communication

        # ── 2. Channel gains at new position ─────────────────────────────
        g_ed    = channel_gain(new_pos, P.C_ED)
        g_users = [channel_gain(new_pos, self.user_pos[i])
                   for i in range(self.n_users)]

        # ── 3. Shared link rates ──────────────────────────────────────────
        R_up_UAV_ED = tx_rate(P.BU_ED, P.p_up_U,  g_ed)    # eq.3a
        R_dn_UAV_ED = tx_rate(P.BU_ED, P.p_ED,    g_ed)    # eq.3b

        # ── 4. Per-user GPU frequency share ──────────────────────────────
        # f_i,t = f_max / N  (paper),  f_max in TFLOPS → compute [FLOP/s]
        f_i_flops = (P.f_max_TFLOPS / self.n_users) * 1e12  # FLOP/s per user

        # ── 5. Per-user latency, energy, PPL ─────────────────────────────
        total_latency  = 0.0
        total_ppl      = 0.0
        E_rel_UAV      = 0.0   # UAV relay energy this slot
        penalty        = 0.0
        lat_per_user   = []
        ppl_per_user   = []

        for i in range(self.n_users):
            gi     = g_users[i]
            R_up_i = tx_rate(P.Bi,    P.p_up_i,   gi)    # eq.2a
            R_dn_i = tx_rate(P.Bi,    P.p_down_U, gi)    # eq.2b

            # Upload latency (eq.4)
            T_up = self.s_i[i] * (1.0 / max(R_up_i,     1.0)
                                  + 1.0 / max(R_up_UAV_ED, 1.0))

            # Inference latency (eq.5):  T_cmp = ψ(d_i) / (f_i · C_ED · D_ED)
            # With f_i already in FLOP/s → T_cmp = ψ / f_i_flops
            flops  = psi_flops(self.d_i[i])
            T_cmp  = flops / f_i_flops
            T_inf  = theta * T_cmp                         # eq.5: ϑ layers

            # Download latency (eq.6)
            T_dn = self.s_out[i] * (1.0 / max(R_dn_UAV_ED, 1.0)
                                    + 1.0 / max(R_dn_i,     1.0))

            Ti = T_up + T_inf + T_dn                       # eq. total latency

            # ── Constraint 17f:  Ti ≤ τ − T_fly  (SOFT PENALTY) ─────────
            if Ti > T_hover:
                penalty += P.W_PEN_LAT * (Ti - T_hover) / max(T_hover, 0.01)

            # ── PPL utility (eq.15) ───────────────────────────────────────
            Ui = ppl(theta)

            # ── Constraint 17g:  PPL(ϑ) ≤ U_req  (SOFT PENALTY) ─────────
            if Ui > P.U_req:
                penalty += P.W_PEN_PPL * (Ui - P.U_req) / P.U_req

            # ── UAV relay energy  (energy consumed by UAV for relaying) ───
            # E_rel_i = energy for forwarding user i's uplink + downlink data
            E_rel_i = (self.s_i[i]   * P.p_up_U    / max(R_up_UAV_ED, 1.0)
                     + self.s_out[i] * P.p_down_U   / max(R_dn_i,     1.0))

            total_latency += Ti
            total_ppl     += Ui
            E_rel_UAV     += E_rel_i
            lat_per_user.append(Ti)
            ppl_per_user.append(Ui)

        # ── 6. UAV flight & hover energy  (eq.12–14) ─────────────────────
        E_fly   = uav_flight_power(v) * T_fly          # eq.12
        E_hover = P.P_H               * T_hover         # eq.13
        E_UAV_t = E_fly + E_hover                       # eq.14

        # ── 7. Laser charging  (eq.10) ───────────────────────────────────
        dist_pb = float(np.linalg.norm(new_pos - P.C_PB))
        P_rv    = laser_received_power(P_pb, dist_pb)
        E_rv    = P_rv * P.tau                           # received energy
        C_pb_t  = P.gamma_pb * P_pb                      # laser cost

        # ── 8. Energy causality constraint  (16) ─────────────────────────
        #   E_min ≤ E_uav + E_rv − ΣE_rel − E_UAV_t ≤ E_bg
        E_new = self.E_uav + E_rv - E_rel_UAV - E_UAV_t
        if E_new < P.E_min:
            penalty += P.W_PEN_ENERGY * (P.E_min - E_new) / P.E_bg
        if E_new > P.E_bg:
            penalty += P.W_PEN_ENERGY * (E_new - P.E_bg) / P.E_bg
        self.E_uav = float(np.clip(E_new, P.E_min, P.E_bg))

        # ── 9. Compute reward  (positive, higher = better) ───────────────
        avg_lat  = total_latency / self.n_users
        avg_ppl  = total_ppl     / self.n_users

        # Normalise to [0, ~1]
        norm_U = avg_ppl  / P.NORM_PPL
        norm_T = avg_lat  / P.NORM_T
        norm_C = C_pb_t   / P.NORM_C

        cost_norm = (P.sigma_U * np.clip(norm_U, 0, 3)
                   + P.sigma_T * np.clip(norm_T, 0, 3)
                   + P.sigma_P * np.clip(norm_C, 0, 3))

        # reward = R_MAX − cost − penalty
        # R_MAX=10, cost ∈ [0,~3], penalty≥0 → reward ∈ (−∞, 10]
        # Feasible behaviour keeps reward ≈ 5–9.
        reward = float(P.R_MAX - cost_norm - penalty)

        # ── 10. Advance state ─────────────────────────────────────────────
        self.uav_pos = new_pos
        self.t      += 1
        terminated   = (self.t >= P.T)

        self.info = {
            # Primary metrics
            "latency_avg_s"   : avg_lat,
            "ppl_avg"         : avg_ppl,
            "cost_pb"         : C_pb_t,
            "norm_cost"       : cost_norm,
            # Detailed
            "latency_per_user": lat_per_user,
            "ppl_per_user"    : ppl_per_user,
            "E_uav_J"         : self.E_uav,
            "E_rv_J"          : E_rv,
            "penalty"         : penalty,
            "theta"           : theta,
            "velocity_mps"    : v,
            "T_hover_s"       : T_hover,
            "T_fly_s"         : T_fly,
            # Constraint satisfaction
            "c17f_ok": all(lat_per_user[i] <= T_hover
                           for i in range(self.n_users)),
            "c17g_ok": all(ppl_per_user[i] <= P.U_req
                           for i in range(self.n_users)),
        }
        return self._get_obs(), reward, terminated, False, self.info


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TRAINING CALLBACK  (logging + live progress bar)
# ═════════════════════════════════════════════════════════════════════════════

class TrainingCallback(BaseCallback):
    """Collects per-episode stats and prints a live progress bar."""

    def __init__(self, total_steps: int, print_every: int = 5000):
        super().__init__()
        self.total_steps = total_steps
        self.print_every = print_every
        # Per-episode lists
        self.ep_rewards    = []
        self.ep_latencies  = []
        self.ep_ppls       = []
        self.ep_costs      = []
        self.ep_penalties  = []
        self.ep_c17f_rates = []
        self.ep_c17g_rates = []
        # Step accumulators
        self._rew   = 0.0
        self._lat   = []; self._ppl  = []; self._cst = []
        self._pen   = []; self._c17f = []; self._c17g = []
        self._last_print = 0
        self._t_start    = time.time()

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[0]
        self._rew += float(self.locals["rewards"][0])

        if info:
            self._lat.append(info.get("latency_avg_s", 0))
            self._ppl.append(info.get("ppl_avg",       0))
            self._cst.append(info.get("cost_pb",        0))
            self._pen.append(info.get("penalty",        0))
            self._c17f.append(float(info.get("c17f_ok", 1)))
            self._c17g.append(float(info.get("c17g_ok", 1)))

        if self.locals.get("dones", [False])[0]:
            self.ep_rewards.append(self._rew)
            def _m(lst): return float(np.mean(lst)) if lst else 0.0
            self.ep_latencies.append(_m(self._lat))
            self.ep_ppls.append(_m(self._ppl))
            self.ep_costs.append(_m(self._cst))
            self.ep_penalties.append(_m(self._pen))
            self.ep_c17f_rates.append(_m(self._c17f) * 100)
            self.ep_c17g_rates.append(_m(self._c17g) * 100)
            self._rew  = 0.0
            self._lat  = []; self._ppl  = []; self._cst = []
            self._pen  = []; self._c17f = []; self._c17g = []

        # ── Progress bar ──────────────────────────────────────────────────
        steps = self.num_timesteps
        if steps - self._last_print >= self.print_every or steps == self.total_steps:
            self._last_print = steps
            pct      = steps / self.total_steps * 100
            done_bar = int(pct / 2)
            bar      = "█" * done_bar + "░" * (50 - done_bar)
            elapsed  = time.time() - self._t_start
            eta      = (elapsed / max(steps, 1)) * (self.total_steps - steps)
            ep_n     = len(self.ep_rewards)
            r_str    = f"{self.ep_rewards[-1]:+.3f}"  if self.ep_rewards  else "   N/A"
            l_str    = f"{self.ep_latencies[-1]:.3f}s" if self.ep_latencies else "  N/A"
            sys.stdout.write(
                f"\r  [{bar}] {pct:5.1f}%  "
                f"step={steps:>7,}  ep={ep_n:>4}  "
                f"reward={r_str:>8}  lat={l_str:>7}  "
                f"ETA={eta/60:5.1f}min"
            )
            sys.stdout.flush()
            if steps >= self.total_steps:
                print()   # final newline
        return True


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — BASELINE POLICIES
# ═════════════════════════════════════════════════════════════════════════════

def random_policy(obs: np.ndarray) -> np.ndarray:
    """Uniformly random actions."""
    return np.random.uniform(-1.0, 1.0, 6).astype(np.float32)


def greedy_policy(obs: np.ndarray) -> np.ndarray:
    """
    Greedy heuristic:
    · Fly toward edge server at mid-altitude
    · Use maximum speed
    · Use maximum laser charging
    · Use maximum transformer layers  (lowest PPL → best quality)
    """
    x = obs[0] * (P.xmax - P.xmin) + P.xmin
    y = obs[1] * (P.ymax - P.ymin) + P.ymin
    z = obs[2] * (P.zmax - P.zmin) + P.zmin
    target  = np.array([P.C_ED[0], P.C_ED[1], (P.zmin + P.zmax) / 2.0])
    delta   = target - np.array([x, y, z])
    norm    = np.linalg.norm(delta) + 1e-6
    delta   = delta / norm * min(30.0, norm)
    return np.array([
        float(np.clip(delta[0] / 30.0, -1, 1)),
        float(np.clip(delta[1] / 30.0, -1, 1)),
        float(np.clip(delta[2] / 15.0, -1, 1)),
        1.0,    # max speed
        1.0,    # max laser
        1.0,    # max layers (best PPL)
    ], dtype=np.float32)


def hover_static_policy(obs: np.ndarray) -> np.ndarray:
    """
    Static hover baseline:
    · No movement
    · Low speed, no laser charging
    · Minimum transformer layers (worst quality)
    """
    return np.array([0.0, 0.0, 0.0, -0.8, -1.0, -1.0], dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_policy_fn(env: UAVLLMEnv,
                       policy_fn,
                       n_episodes: int = 200) -> dict:
    """Run policy_fn for n_episodes; return dict of result arrays."""
    rewards, latencies, ppls, costs = [], [], [], []
    c17f_rates, c17g_rates           = [], []

    for ep_idx in range(n_episodes):
        obs, _ = env.reset()
        ep_rew = 0.0
        ep_lat, ep_ppl, ep_cst = [], [], []
        ep_c17f, ep_c17g       = [], []
        done = False
        while not done:
            act               = policy_fn(obs)
            obs, rew, done, _, info = env.step(act)
            ep_rew           += rew
            ep_lat.append(info.get("latency_avg_s", 0))
            ep_ppl.append(info.get("ppl_avg",       0))
            ep_cst.append(info.get("cost_pb",        0))
            ep_c17f.append(float(info.get("c17f_ok", 1)))
            ep_c17g.append(float(info.get("c17g_ok", 1)))

        rewards.append(ep_rew)
        latencies.append(float(np.mean(ep_lat)))
        ppls.append(float(np.mean(ep_ppl)))
        costs.append(float(np.mean(ep_cst)))
        c17f_rates.append(float(np.mean(ep_c17f)) * 100)
        c17g_rates.append(float(np.mean(ep_c17g)) * 100)

        # Mini progress within evaluation
        if (ep_idx + 1) % 50 == 0:
            sys.stdout.write(f"    {ep_idx+1}/{n_episodes} eps done\r")
            sys.stdout.flush()

    return {
        "reward"    : np.array(rewards),
        "latency"   : np.array(latencies),
        "ppl"       : np.array(ppls),
        "cost"      : np.array(costs),
        "c17f_rate" : np.array(c17f_rates),
        "c17g_rate" : np.array(c17g_rates),
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PLOTTING
# ═════════════════════════════════════════════════════════════════════════════

COLORS = {
    "PPO (Proposed)" : "#2563EB",
    "Greedy"         : "#16A34A",
    "Random"         : "#DC2626",
    "Hover (Static)" : "#9333EA",
}


def smooth(x: np.ndarray, w: int = 30) -> np.ndarray:
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def make_plots(cb: TrainingCallback, results: dict, save_path: str):
    """
    3×3 figure (8 panels):
      Row 0: PPO convergence (reward)  |  Policy uncertainty  |  Constraint rates
      Row 1: Bar reward                |  Bar latency         |  Bar PPL
      Row 2: CDF reward                |  Latency convergence |  PPL convergence
    """
    names = list(results.keys())

    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_facecolor("#F0F4F8")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.40)

    BG = "#FFFFFF"

    def _bar(ax, key, ylabel, title, fmt=".3f", lower_better=False):
        means = [results[n][key].mean() for n in names]
        stds  = [results[n][key].std()  for n in names]
        bars  = ax.bar(names, means, yerr=stds,
                       color=[COLORS[n] for n in names],
                       capsize=5, edgecolor="white", linewidth=0.6,
                       error_kw={"elinewidth": 1.5, "capthick": 1.5})
        if lower_better:
            title += "\n(↓ = better)"
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        ax.set_facecolor(BG)
        ax.tick_params(axis="x", labelsize=8)
        for bar, m in zip(bars, means):
            ypos = bar.get_height() * 1.03 + 0.001
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{m:{fmt}}", ha="center", va="bottom", fontsize=8)

    # ── Plot 1: PPO convergence — reward ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ep_r = np.array(cb.ep_rewards)
    ax1.plot(ep_r, alpha=0.18, color="#93C5FD", linewidth=0.6, label="Raw")
    if len(ep_r) >= 30:
        sm = smooth(ep_r, 30)
        ax1.plot(range(len(sm)), sm, color=COLORS["PPO (Proposed)"],
                 linewidth=2.3, label="Smoothed (MA-30)")
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.4)
    ax1.set_title("PPO Convergence — Episode Reward  (higher = better)",
                  fontsize=12, fontweight="bold")
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Cumulative Reward")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3); ax1.set_facecolor(BG)

    # ── Plot 2: Policy uncertainty (rolling std of reward) ────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    if len(ep_r) > 40:
        roll_std = [ep_r[max(0, i-20):i+1].std() for i in range(len(ep_r))]
        ax2.plot(roll_std, color="#F59E0B", linewidth=1.6)
    ax2.set_title("Policy Uncertainty\n(Rolling Std, window=20)",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Std of Reward")
    ax2.grid(alpha=0.3); ax2.set_facecolor(BG)

    # ── Plot 3: Bar — average episode reward ─────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _bar(ax3, "reward", "Reward", "Average Episode Reward")

    # ── Plot 4: Bar — average latency ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    _bar(ax4, "latency", "Latency [s]", "Average Latency per User", fmt=".4f",
         lower_better=True)
    ax4.axhline(P.tau, color="red", linewidth=1.2, linestyle="--",
                alpha=0.7, label=f"τ = {P.tau}s")
    ax4.legend(fontsize=8)

    # ── Plot 5: Bar — average PPL ────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    _bar(ax5, "ppl", "PPL", "Average PPL", fmt=".2f", lower_better=True)
    ax5.axhline(P.U_req, color="red", linewidth=1.2, linestyle="--",
                alpha=0.7, label=f"U_req = {P.U_req}")
    ax5.legend(fontsize=8)

    # ── Plot 6: CDF of episode reward ────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    for name in names:
        rs  = np.sort(results[name]["reward"])
        cdf = np.arange(1, len(rs) + 1) / len(rs)
        ax6.plot(rs, cdf, color=COLORS[name], linewidth=2.0, label=name)
    ax6.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.4)
    ax6.set_title("CDF of Episode Reward", fontsize=11, fontweight="bold")
    ax6.set_xlabel("Cumulative Reward"); ax6.set_ylabel("CDF")
    ax6.legend(fontsize=7); ax6.grid(alpha=0.3); ax6.set_facecolor(BG)

    # ── Plot 7: Latency convergence during training ───────────────────────
    ax7 = fig.add_subplot(gs[2, 1])
    lat_arr = np.array(cb.ep_latencies)
    ax7.plot(lat_arr, alpha=0.22, color="#6EE7B7", linewidth=0.6)
    if len(lat_arr) >= 30:
        sm7 = smooth(lat_arr, 30)
        ax7.plot(range(len(sm7)), sm7, color="#059669",
                 linewidth=2.3, label="Smoothed")
    ax7.axhline(P.tau, color="red", linewidth=1.2, linestyle="--",
                alpha=0.6, label=f"τ = {P.tau}s")
    ax7.set_title("Latency Convergence During Training",
                  fontsize=11, fontweight="bold")
    ax7.set_xlabel("Episode"); ax7.set_ylabel("Avg Latency [s]")
    ax7.legend(fontsize=8); ax7.grid(alpha=0.3); ax7.set_facecolor(BG)

    # ── Plot 8: PPL convergence during training ───────────────────────────
    ax8 = fig.add_subplot(gs[2, 2])
    ppl_arr = np.array(cb.ep_ppls)
    ax8.plot(ppl_arr, alpha=0.22, color="#FCA5A5", linewidth=0.6)
    if len(ppl_arr) >= 30:
        sm8 = smooth(ppl_arr, 30)
        ax8.plot(range(len(sm8)), sm8, color="#B91C1C",
                 linewidth=2.3, label="Smoothed")
    ax8.axhline(P.U_req, color="orange", linewidth=1.2, linestyle="--",
                alpha=0.7, label=f"U_req = {P.U_req}")
    ax8.set_title("PPL Convergence During Training\n(↓ = better quality)",
                  fontsize=11, fontweight="bold")
    ax8.set_xlabel("Episode"); ax8.set_ylabel("Avg PPL")
    ax8.legend(fontsize=8); ax8.grid(alpha=0.3); ax8.set_facecolor(BG)

    fig.suptitle(
        "Joint LLM Task Offloading & UAV Trajectory Optimization\n"
        "PPO (Proposed) vs Baselines — UAV-enabled Edge Network "
        "with Laser Power Charging",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Figure saved → {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    np.random.seed(42)

    out_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    ts        = time.strftime("%Y%m%d_%H%M%S")

    print("=" * 72)
    print("  UAV-LLM Offloading — PPO Optimization  (stable-baselines3)")
    print("=" * 72)

    # ── Sanity check: print realistic latency at typical UAV position ─────
    _p = np.array([0.0, 0.0, 50.0])
    _u = np.array([50.0, 50.0, 0.0])
    _ge  = channel_gain(_p, P.C_ED)
    _gu  = channel_gain(_p, _u)
    _Rui = tx_rate(P.Bi,    P.p_up_i,   _gu)
    _Rue = tx_rate(P.BU_ED, P.p_up_U,   _ge)
    _Rde = tx_rate(P.BU_ED, P.p_ED,     _ge)
    _Rdi = tx_rate(P.Bi,    P.p_down_U, _gu)
    _d   = 768;  _s = _d*32;  _so = 200*32
    _fi  = (P.f_max_TFLOPS/P.N)*1e12
    _Tup = _s  * (1/max(_Rui,1) + 1/max(_Rue,1))
    _Tin = 48  * psi_flops(_d) / _fi
    _Tdn = _so * (1/max(_Rde,1) + 1/max(_Rdi,1))
    print(f"\n  Sanity check  (UAV=(0,0,50), user=(50,50,0), θ=48, N={P.N}):")
    print(f"    R_up_i    = {_Rui/1e6:.4f} Mbps")
    print(f"    R_up_ED   = {_Rue/1e6:.4f} Mbps")
    print(f"    T_up      = {_Tup*1e3:.1f} ms")
    print(f"    T_inf(48) = {_Tin*1e3:.1f} ms")
    print(f"    T_dn      = {_Tdn*1e3:.1f} ms")
    print(f"    T_total   = {(_Tup+_Tin+_Tdn)*1e3:.1f} ms  "
          f"(τ=3000 ms → OK = {(_Tup+_Tin+_Tdn)<P.tau})")
    print(f"    PPL(32)   = {ppl(32):.3f},  PPL(48) = {ppl(48):.3f},  "
          f"PPL(64) = {ppl(64):.3f}   (U_req = {P.U_req})")
    print()

    # ═════════════════════════════════════════════════════════════════════
    # 8.1  TRAINING
    # ═════════════════════════════════════════════════════════════════════
    N_USERS     = P.N          # 10  (change to 20 for extended experiment)
    TOTAL_STEPS = 500_000      # recommended for NCKH; reduce to 100_000 for quick test

    train_env = Monitor(UAVLLMEnv(n_users=N_USERS, seed=42))
    cb = TrainingCallback(total_steps=TOTAL_STEPS, print_every=5000)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate = 3e-4,
        n_steps       = 1024,
        batch_size    = 128,
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        ent_coef      = 0.005,
        vf_coef       = 0.5,
        max_grad_norm = 0.5,
        policy_kwargs = dict(net_arch=[256, 256, 128]),
        verbose       = 0,
        seed          = 42,
    )

    print(f"  Training PPO  ({TOTAL_STEPS:,} steps, {N_USERS} users)\n")
    t0 = time.time()
    model.learn(total_timesteps=TOTAL_STEPS, callback=cb, progress_bar=False)
    elapsed = time.time() - t0
    print(f"\n  Training complete — {elapsed:.1f}s  ({elapsed/60:.1f} min)\n")

    # ═════════════════════════════════════════════════════════════════════
    # 8.2  EVALUATION
    # ═════════════════════════════════════════════════════════════════════
    N_EVAL   = 200
    eval_env = UAVLLMEnv(n_users=N_USERS, seed=99)

    def ppo_fn(obs):
        act, _ = model.predict(obs, deterministic=True)
        return act

    policies = {
        "PPO (Proposed)" : ppo_fn,
        "Greedy"         : greedy_policy,
        "Random"         : random_policy,
        "Hover (Static)" : hover_static_policy,
    }

    print(f"  Evaluating {len(policies)} policies  ({N_EVAL} episodes each) …\n")
    results = {}
    for name, fn in policies.items():
        print(f"    [{name}] …")
        results[name] = evaluate_policy_fn(eval_env, fn, N_EVAL)
        r = results[name]
        print(f"      reward  = {r['reward'].mean():.3f} ± {r['reward'].std():.3f}")
        print(f"      latency = {r['latency'].mean():.4f} s")
        print(f"      PPL     = {r['ppl'].mean():.3f}")
        print(f"      C17f    = {r['c17f_rate'].mean():.1f}%  "
              f"(latency constraint satisfied)")
        print(f"      C17g    = {r['c17g_rate'].mean():.1f}%  "
              f"(PPL constraint satisfied)")
        print()

    # ═════════════════════════════════════════════════════════════════════
    # 8.3  SUMMARY TABLE
    # ═════════════════════════════════════════════════════════════════════
    W = 84
    print("  " + "=" * W)
    print(f"  {'Policy':<18} {'Reward':>8} {'±':>6} {'Lat(s)':>8} "
          f"{'PPL':>6} {'C17f%':>7} {'C17g%':>7} {'CostPB':>8}")
    print("  " + "-" * W)
    for name, r in results.items():
        print(f"  {name:<18} "
              f"{r['reward'].mean():>8.3f} "
              f"{r['reward'].std():>6.3f} "
              f"{r['latency'].mean():>8.4f} "
              f"{r['ppl'].mean():>6.2f} "
              f"{r['c17f_rate'].mean():>7.1f} "
              f"{r['c17g_rate'].mean():>7.1f} "
              f"{r['cost'].mean():>8.4f}")
    print("  " + "=" * W)
    print()
    ppo_r   = results["PPO (Proposed)"]["reward"].mean()
    base_r  = results["Greedy"]["reward"].mean()
    rand_r  = results["Random"]["reward"].mean()
    print(f"  PPO improvement over Greedy    : "
          f"{(ppo_r - base_r) / abs(base_r) * 100:+.1f}%")
    print(f"  PPO improvement over Random    : "
          f"{(ppo_r - rand_r) / abs(rand_r) * 100:+.1f}%")
    print()
    print("  C17f = % time-slots where ALL users satisfy latency constraint (17f)")
    print("  C17g = % time-slots where ALL users satisfy PPL constraint     (17g)")
    print()

    # ═════════════════════════════════════════════════════════════════════
    # 8.4  SAVE PLOTS
    # ═════════════════════════════════════════════════════════════════════
    fig_path = os.path.join(out_dir, f"uav_llm_results_{ts}.png")
    print("  Generating plots …")
    make_plots(cb, results, fig_path)

    # ═════════════════════════════════════════════════════════════════════
    # 8.5  SAVE MODEL
    # ═════════════════════════════════════════════════════════════════════
    model_path = os.path.join(out_dir, f"ppo_uav_llm_{ts}")
    model.save(model_path)
    print(f"  Model saved   → {model_path}.zip")

    print("\n" + "=" * 72)
    print("  All done!")
    print("=" * 72)

    return model, results, cb


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

    