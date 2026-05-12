# PDM Metric Notebooks

Interactive Jupyter notebooks demonstrating every metric in the **two-stage PDM evaluation pipeline**.
Each notebook contains animated Bird's Eye View (BEV) scenarios that play in real time so you can
build an intuition for what each threshold actually means in physical driving terms.

---

## Two-Stage PDM Score — Overview

The final PDM score is computed as:

```
PDM score = M × W

M  =  no_collision × drivable_area × driving_direction × traffic_light   (multiplicative)

W  =  Σ (score_i × weight_i) / Σ weight_i                                (weighted average)
         i ∈ {progress, ttc, lane_keeping, history_comfort}
```

The **Two-Frame Extended Comfort** metric is evaluated separately by the `SceneAggregator`
across consecutive planning frames, not inside the per-scene PDM score.

---

## Metric Reference

### Multiplicative Metrics
A score of 0.0 on any of these zeroes the entire PDM score, regardless of other metrics.

| Metric | Score values | Threshold / rule |
|---|---|---|
| No At-Fault Collision | 0.0 / 0.5 / 1.0 | 0.0 = agent hit; 0.5 = static object hit; 1.0 = no collision |
| Drivable Area Compliance | 0.0 / 1.0 | Any of 4 corners outside drivable polygon → 0.0 |
| Driving Direction | 0.0 / 0.5 / 1.0 | Wrong-way progress in 1 s window: <2 m→1.0, 2–6 m→0.5, >6 m→0.0 |
| Traffic Light Compliance | 0.0 / 1.0 | Ego polygon intersects red-light zone → 0.0 |

### Weighted Metrics

| Metric | Weight | Score | Rule |
|---|---|---|---|
| Ego Progress | **5.0** | 0.0–1.0 | `min(1, progress / max_progress)`; exempt if max < 5 m |
| Time-to-Collision (TTC) | **5.0** | 0.0 / 1.0 | Projected front hits agent within 1 s → 0.0 (stopped ego exempt) |
| Lane Keeping | 2.0 | 0.0 / 1.0 | Lateral deviation > 0.5 m for ≥ 2.0 s continuously → 0.0 |
| History Comfort | 2.0 | 0.0 / 1.0 | All 6 sub-metrics must pass (see `history_comfort_thresholds.ipynb`) |

### Aggregator-Level Metric

| Metric | Weight | Score | Rule |
|---|---|---|---|
| Two-Frame Extended Comfort | 2.0 | 0.0 / 1.0 | RMS of feature differences between consecutive frames within thresholds |

Two-Frame Extended Comfort thresholds:

| Feature | RMS threshold |
|---|---|
| Acceleration magnitude | ≤ 0.7 m/s² |
| Jerk magnitude | ≤ 0.5 m/s³ |
| Yaw rate | ≤ 0.1 rad/s |
| Yaw acceleration | ≤ 0.1 rad/s² |

---

## Notebooks

### Per-Metric Notebooks
Each has 3 animated scenarios covering the key scoring cases (pass / borderline / fail).

| Notebook | Metric | Scenarios |
|---|---|---|
| `metric_collision.ipynb` | No At-Fault Collision | Adjacent-lane pass (1.0) · Rear-end agent (0.0) · Static object hit (0.5) |
| `metric_ttc.ipynb` | Time-to-Collision | Safe follow (1.0) · Closing gap — TTC < 1 s (0.0) · Stopped ego (1.0) |
| `metric_lane_keeping.ipynb` | Lane Keeping | Perfect centering · Brief drift < 2 s (1.0) · Sustained drift ≥ 2 s (0.0) |
| `metric_drivable_area.ipynb` | Drivable Area | Centered drive · Gradual drift off road (0.0) · Diagonal exit (0.0) |
| `metric_driving_direction.ipynb` | Driving Direction | Correct direction · Brief reversal < 2 m (1.0) · Wrong-way > 6 m (0.0) |
| `metric_traffic_light.ipynb` | Traffic Light | Green light · Stops at red (1.0) · Runs red (0.0) |
| `metric_ego_progress.ipynb` | Ego Progress | Stationary (0.0) · Half speed (0.5) · Full speed (1.0) |
| `metric_two_frame_comfort.ipynb` | Two-Frame Comfort | Consistent frames (1.0) · Accel mismatch (0.0) · Yaw mismatch (0.0) |

### History Comfort Sub-Metrics
`history_comfort_thresholds.ipynb` — detailed breakdown of all 6 comfort sub-metrics with
threshold values, animated BEV, velocity panel, and real-world driving references.

| Sub-metric | Threshold | Real-world equivalent |
|---|---|---|
| Lon. acceleration (fwd) | +2.40 m/s² | 0 → 43 km/h in 5 s — moderate city pull-away |
| Lon. acceleration (brk) | −4.05 m/s² | 60 → 0 km/h in 4.1 s — firm controlled stop |
| Lateral acceleration | ±4.89 m/s² | 30 km/h on r = 14 m — tight city corner |
| Magnitude jerk | ±8.37 m/s³ | Full acceleration reached in ~290 ms — brisk floor-it |
| Longitudinal jerk | ±4.13 m/s³ | Full acceleration reached in ~580 ms — deliberate ramp |
| Yaw rate | ±0.95 rad/s | ±54°/s — tight U-turn / hairpin at 25 km/h |
| Yaw acceleration | ±1.93 rad/s² | Max yaw rate reached in ~490 ms — sharp turn entry |

`history_comfort_thresholds.py` — standalone script that generates a static summary figure
(`history_comfort_thresholds.png`) showing all 6 threshold profiles.

---

## Regenerating the Notebooks

```bash
conda run -n navsim python docs/metrics/generate_metric_notebooks.py
```

Rewrites all 8 `metric_*.ipynb` files in this directory. The generator script is
`generate_metric_notebooks.py` — edit the scenario code there to change any simulation.

---

## Animation Controls

All animations use `FuncAnimation` rendered as embedded HTML5 via `to_jshtml()`.
In JupyterLab, the player bar appears below each animation cell — use the ▶ button to play,
or drag the slider to scrub through time. Each frame = 0.1 s of simulated time.
