# BatchLQRTracker

**File**: `navsim/planning/simulation/planner/pdm_planner/simulation/batch_lqr.py`

## What It Does

A **batch Linear Quadratic Regulator (LQR) tracker** — given a set of proposed (x, y, heading) trajectories, it computes the optimal `(acceleration, steering_rate)` commands to make the vehicle physically follow them. "Batch" means it runs over all proposals simultaneously using vectorized numpy ops.

---

## Decoupled Subsystems

The tracker splits the problem into two independent controllers:

### 1. Longitudinal (speed tracking)

```
State:  [velocity]
Input:  [acceleration]
Model:  velocity_dot = acceleration
```

Simple 1D system — just track the reference speed at the lookahead horizon.

### 2. Lateral (path tracking)

```
State:  [lateral_error, heading_error, steering_angle]
Input:  [steering_rate]
Model (linearized bicycle):
  lateral_error_dot  = v * heading_error
  heading_error_dot  = v * (steering_angle/wheelbase - curvature)
  steering_angle_dot = steering_rate
```

This is the standard kinematic bicycle model linearized around small angles.

---

## `track_trajectory()` — Step by Step

```
initial_states (current ego states, batch)
         │
         ├─ _compute_initial_velocity_and_lateral_state()
         │    ├─ lateral_error  = -Δx·sin(θ_ref) + Δy·cos(θ_ref)   ← Frenet projection
         │    ├─ heading_error  = normalize(θ_ego - θ_ref)
         │    └─ steering_angle = from ego state
         │
         ├─ _compute_reference_velocity_and_curvature_profile()
         │    └─ from poses → velocity + curvature over tracking_horizon lookahead
         │
         ├─ [if ref_v ≤ 0.2 AND ego_v ≤ 0.2]  → stopping P controller
         │    accel = -0.5 * (v_ego - v_ref),  steering_rate = 0
         │
         └─ [else]  → LQR controllers
              ├─ _longitudinal_lqr_controller()  → accel_cmd
              ├─ integrate accel → velocity_profile over horizon
              └─ _lateral_lqr_controller()       → steering_rate_cmd
```

Output: `command_states[:, [ACCELERATION_X, STEERING_RATE]]`

---

## LQR Solver (1-step formulation)

Both subsystems use the same trick: **roll the N-step dynamics into a single-step problem**.

The key insight is: assume the input is held **constant** for the entire horizon. Then:

```
x_N = A·x_0 + B·u + g
```

where `A`, `B`, `g` are the composed matrices over N steps.

The LQR optimal input (closed-form, no Riccati iteration needed) is:

```
u* = -(B^T Q B + R)^{-1} · B^T Q · (A·x_0 + g - x_ref)
```

This is analytically invertible because it's a scalar (longitudinal) or small matrix (lateral, 3×3) — no iterative solver required.

---

## Key Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `q_longitudinal` | `[10.0]` | Penalty on velocity error |
| `r_longitudinal` | `[1.0]` | Penalty on acceleration effort |
| `q_lateral` | `[1.0, 10.0, 0.0]` | Penalty on `[lateral_err, heading_err, steer_angle]` |
| `r_lateral` | `[1.0]` | Penalty on steering_rate effort |
| `tracking_horizon` | `10` | Steps lookahead (10 × 0.1s = 1s) |
| `stopping_velocity` | `0.2 m/s` | Below this → P controller instead of LQR |
| `jerk_penalty` | `1e-4` | Smoothness penalty when computing velocity profile from poses |
| `curvature_rate_penalty` | `1e-2` | Smoothness penalty when computing curvature from poses |

The `q_lateral` weighting of `[1, 10, 0]` means **heading error is penalized 10× more than lateral offset**, and steering angle itself is not penalized — prioritizing direction correction over position correction.

---

## Lateral A/B/g Matrix Construction

The discretized lateral dynamics at each step `k` are:

```
A_k = I + dt * [[0, v_k, 0        ],
                 [0,  0, v_k/L     ],
                 [0,  0,  0        ]]

g_k = [0, -v_k * κ_k * dt, 0]^T
```

The loop at line 382 of `batch_lqr.py` **composes** these step-by-step via matrix multiply to get the N-horizon rollout `A`, `B`, `g`. This is a Linear Time-Varying (LTV) system — `A_k` changes at each step because velocity `v_k` and curvature `κ_k` vary along the trajectory.

---

## Where It Fits in PDMSimulator

```
PDMSimulator.simulate_proposals()
    for each timestep t:
        tracker.track_trajectory()    ← BatchLQRTracker computes (accel, steering_rate)
        motion_model.forward()         ← BatchKinematicBicycleModel applies commands → next state
```

The tracker never directly moves the vehicle — it produces commands, then the bicycle model integrates them into the next state.

---

## Stopping Controller

When both reference velocity and current velocity are below `stopping_velocity` (0.2 m/s), the tracker falls back to a simple proportional controller:

```
accel        = -0.5 * (v_ego - v_ref)
steering_rate = 0
```

This avoids LQR instability at near-zero speeds where the linearized bicycle model breaks down.
