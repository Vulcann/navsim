"""
Intuitive visualization of history_comfort metric thresholds.

Each panel simulates a minimal trajectory that sits right at one threshold,
shows the vehicle path (x-y), and plots the metric value over time with the
threshold line marked.

Run:  python docs/history_comfort_thresholds.py
Saves: docs/history_comfort_thresholds.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import savgol_filter

# ── Simulation constants ──────────────────────────────────────────────────────
DT   = 0.1          # s (same as navsim discretization)
T    = 5.0          # s total duration
time = np.arange(0, T + DT, DT)
N    = len(time)

# ── Thresholds (from pdm_comfort_metrics.py) ─────────────────────────────────
MAX_LON_ACCEL  =  2.40   # m/s²
MIN_LON_ACCEL  = -4.05   # m/s²
MAX_LAT_ACCEL  =  4.89   # m/s²
MAX_MAG_JERK   =  8.37   # m/s³
MAX_LON_JERK   =  4.13   # m/s³
MAX_YAW_ACCEL  =  1.93   # rad/s²
MAX_YAW_RATE   =  0.95   # rad/s

# ── Helpers ───────────────────────────────────────────────────────────────────

def euler_integrate(rate_array, init=0.0):
    """Simple Euler integration of a rate signal → cumulative values."""
    vals = np.zeros(N)
    vals[0] = init
    for i in range(1, N):
        vals[i] = vals[i - 1] + rate_array[i - 1] * DT
    return vals


def xy_from_speed_heading(speed, heading, x0=0.0, y0=0.0):
    """Integrate speed + heading → (x, y) trajectory."""
    x = np.zeros(N); y = np.zeros(N)
    x[0] = x0; y[0] = y0
    for i in range(1, N):
        x[i] = x[i - 1] + speed[i - 1] * np.cos(heading[i - 1]) * DT
        y[i] = y[i - 1] + speed[i - 1] * np.sin(heading[i - 1]) * DT
    return x, y


def sg_smooth(signal, window=8, poly=2):
    wl = min(window, N)
    return savgol_filter(signal, window_length=wl, polyorder=poly)


def sg_diff(signal, deriv=1, window=15, poly=2):
    """Savitzky-Golay derivative — identical to navsim's _approximate_derivatives."""
    wl = min(window, N)
    return savgol_filter(signal, window_length=wl, polyorder=poly,
                         deriv=deriv, delta=DT)


def phase_unwrap(headings):
    two_pi = 2.0 * np.pi
    adjustments = np.zeros_like(headings)
    adjustments[1:] = np.cumsum(np.round(np.diff(headings) / two_pi))
    return headings - two_pi * adjustments


# ── Panel helper ──────────────────────────────────────────────────────────────

def draw_panel(ax_xy, ax_ts, x, y, metric_vals, threshold, label,
               unit, scenario_text, color_traj="#1565C0", mirror=False):
    """
    ax_xy      : x-y trajectory axes
    ax_ts      : time-series metric axes
    x, y       : vehicle path arrays (length N)
    metric_vals: computed metric over time (length N)
    threshold  : positive scalar threshold value
    mirror     : True → threshold applies both ±
    """
    # ── x-y plot ──
    ax_xy.plot(x, y, color=color_traj, lw=2)
    ax_xy.plot(x[0], y[0], "go", ms=8, label="start")
    ax_xy.plot(x[-1], y[-1], "rs", ms=8, label="end")
    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.legend(fontsize=7, loc="best")
    ax_xy.set_title(scenario_text, fontsize=8)
    ax_xy.grid(True, alpha=0.3)

    # ── time-series plot ──
    ax_ts.plot(time, metric_vals, color=color_traj, lw=2, label=label)
    ax_ts.axhline( threshold, color="red", lw=1.5, ls="--", label=f"+threshold ({threshold} {unit})")
    if mirror:
        ax_ts.axhline(-threshold, color="orange", lw=1.5, ls="--", label=f"−threshold")
    ax_ts.set_xlabel("time [s]")
    ax_ts.set_ylabel(f"{label}\n[{unit}]")
    ax_ts.legend(fontsize=7, loc="best")
    ax_ts.grid(True, alpha=0.3)


# ═══════════════════════════════════════════════════════════════════════════════
# Build figure: 6 rows × 2 columns  (left = x-y path, right = metric vs. time)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(6, 2, figsize=(13, 28))
fig.suptitle(
    "history_comfort — One threshold scenario per metric\n"
    "(each scenario is designed to sit right at the limit)",
    fontsize=13, fontweight="bold", y=0.995
)

# ─────────────────────────────────────────────────────────────────────────────
# Row 0 │ Longitudinal acceleration (+2.40 m/s²)
# Scenario: vehicle starts at rest, applies exactly +2.40 m/s² the whole time.
#   After 5 s → speed = 12 m/s = 43 km/h.  Straight-line path.
# Real-world feel: moderate urban pull-away — like merging onto a ring road.
# ─────────────────────────────────────────────────────────────────────────────
accel_lon = np.full(N, MAX_LON_ACCEL)
speed_lon  = euler_integrate(accel_lon, init=0.0)
heading_lon = np.zeros(N)
x0, y0     = xy_from_speed_heading(speed_lon, heading_lon)

# navsim smooths acceleration with SG before checking bounds
accel_lon_smooth = sg_smooth(accel_lon)

draw_panel(
    axes[0, 0], axes[0, 1],
    x0, y0,
    accel_lon_smooth,
    MAX_LON_ACCEL, "lon. accel", "m/s²",
    f"Constant +{MAX_LON_ACCEL} m/s²  |  0 → {speed_lon[-1]:.1f} m/s ({speed_lon[-1]*3.6:.0f} km/h) in {T:.0f}s",
)
axes[0, 1].set_title(
    f"Threshold = {MAX_LON_ACCEL} m/s²\n"
    "≈ 0→43 km/h in 5 s  |  comfortable city pull-away",
    fontsize=8
)

# ─────────────────────────────────────────────────────────────────────────────
# Row 1 │ Longitudinal acceleration (braking, −4.05 m/s²)
# Scenario: start at 60 km/h = 16.67 m/s, brake at exactly −4.05 m/s².
#   Stops after 16.67 / 4.05 ≈ 4.1 s.
# Real-world feel: firm, controlled emergency stop — not a panic brake.
# ─────────────────────────────────────────────────────────────────────────────
v_init      = 60 / 3.6          # 16.67 m/s
stop_t      = v_init / abs(MIN_LON_ACCEL)
accel_brk   = np.where(time < stop_t, MIN_LON_ACCEL, 0.0)
speed_brk   = np.maximum(euler_integrate(accel_brk, init=v_init), 0.0)
accel_brk_s = sg_smooth(accel_brk)
x1, y1      = xy_from_speed_heading(speed_brk, np.zeros(N))

draw_panel(
    axes[1, 0], axes[1, 1],
    x1, y1,
    accel_brk_s,
    abs(MIN_LON_ACCEL), "lon. accel (braking)", "m/s²",
    f"Constant −{abs(MIN_LON_ACCEL)} m/s²  |  60 km/h → 0 in {stop_t:.1f}s  (distance {np.trapz(speed_brk, dx=DT):.1f} m)",
    color_traj="#6A1B9A",
)
axes[1, 1].set_title(
    f"Threshold = {abs(MIN_LON_ACCEL)} m/s²  (braking)\n"
    "≈ 60→0 km/h in 4.1 s  |  firm controlled emergency stop",
    fontsize=8
)
# flip sign to show it crossing the lower threshold line
axes[1, 1].lines[0].set_ydata(accel_brk_s)
axes[1, 1].axhline( MIN_LON_ACCEL, color="red",    lw=1.5, ls="--", label=f"−threshold ({MIN_LON_ACCEL} m/s²)")
axes[1, 1].axhline(0, color="gray", lw=0.8, ls=":")
axes[1, 1].legend(fontsize=7)

# ─────────────────────────────────────────────────────────────────────────────
# Row 2 │ Lateral acceleration (±4.89 m/s²)
# Scenario: circular arc at constant speed 30 km/h (8.33 m/s).
#   Radius r = v² / a_lat = 8.33² / 4.89 ≈ 14.2 m  — a tight 90° city corner.
# Real-world feel: fast exit of a sharp parking-lot corner, or a mini-roundabout.
# ─────────────────────────────────────────────────────────────────────────────
v_lat    = 30 / 3.6              # 8.33 m/s
r_lat    = v_lat**2 / MAX_LAT_ACCEL
yaw_rate_lat = v_lat / r_lat     # rad/s — constant for circular motion
heading_lat  = euler_integrate(np.full(N, yaw_rate_lat), init=0.0)
speed_lat    = np.full(N, v_lat)
x2, y2       = xy_from_speed_heading(speed_lat, heading_lat)

# lateral accel = v * yaw_rate  (centripetal)
lat_accel_vals = speed_lat * yaw_rate_lat  # constant here

draw_panel(
    axes[2, 0], axes[2, 1],
    x2, y2,
    np.full(N, lat_accel_vals[0]),
    MAX_LAT_ACCEL, "lat. accel", "m/s²",
    f"Circular arc  |  v = {v_lat*3.6:.0f} km/h  |  radius = {r_lat:.1f} m  |  turn angle = {np.degrees(heading_lat[-1]):.0f}°",
    color_traj="#00695C",
)
axes[2, 1].set_title(
    f"Threshold = {MAX_LAT_ACCEL} m/s²\n"
    f"≈ 30 km/h on a r={r_lat:.0f} m turn  |  tight city corner / mini-roundabout",
    fontsize=8
)

# ─────────────────────────────────────────────────────────────────────────────
# Row 3 │ Magnitude jerk (±8.37 m/s³)
# Scenario: acceleration ramps linearly from 0 to MAX_LON_ACCEL over t_ramp,
#   then holds constant.  Jerk = Δaccel / Δt → t_ramp = 2.40/8.37 ≈ 0.29 s.
# Real-world feel: flooring the pedal briskly — acceleration kicks in within
#   ~3 timesteps (300 ms).
# ─────────────────────────────────────────────────────────────────────────────
t_ramp3  = MAX_LON_ACCEL / MAX_MAG_JERK   # ≈ 0.287 s
accel3   = np.where(time <= t_ramp3, (MAX_LON_ACCEL / t_ramp3) * time,
                                      MAX_LON_ACCEL)
accel3   = np.minimum(accel3, MAX_LON_ACCEL)
accel3_s = sg_smooth(accel3, window=8)
jerk3    = sg_diff(accel3_s, deriv=1, window=15)
speed3   = euler_integrate(accel3, init=0.0)
x3, y3   = xy_from_speed_heading(speed3, np.zeros(N))

draw_panel(
    axes[3, 0], axes[3, 1],
    x3, y3,
    jerk3,
    MAX_MAG_JERK, "magnitude jerk", "m/s³",
    f"Accel ramps 0→{MAX_LON_ACCEL} m/s² in {t_ramp3*1000:.0f} ms, then holds constant",
    color_traj="#E65100", mirror=True,
)
axes[3, 1].set_title(
    f"Threshold = {MAX_MAG_JERK} m/s³\n"
    f"≈ max accel reached in {t_ramp3*1000:.0f} ms  |  brisk pedal push",
    fontsize=8
)

# ─────────────────────────────────────────────────────────────────────────────
# Row 4 │ Longitudinal jerk (±4.13 m/s³)
# Same scenario but at the (tighter) longitudinal jerk threshold.
#   t_ramp = 2.40 / 4.13 ≈ 0.58 s — twice as slow as magnitude jerk limit.
# Real-world feel: smooth but deliberate acceleration — like merging from a
#   standing start with intention, not a floor-it moment.
# ─────────────────────────────────────────────────────────────────────────────
t_ramp4  = MAX_LON_ACCEL / MAX_LON_JERK   # ≈ 0.58 s
accel4   = np.where(time <= t_ramp4, (MAX_LON_ACCEL / t_ramp4) * time,
                                      MAX_LON_ACCEL)
accel4   = np.minimum(accel4, MAX_LON_ACCEL)
accel4_s = sg_smooth(accel4, window=8)
lon_jerk4 = sg_diff(accel4_s, deriv=1, window=15)
speed4    = euler_integrate(accel4, init=0.0)
x4, y4    = xy_from_speed_heading(speed4, np.zeros(N))

draw_panel(
    axes[4, 0], axes[4, 1],
    x4, y4,
    lon_jerk4,
    MAX_LON_JERK, "lon. jerk", "m/s³",
    f"Accel ramps 0→{MAX_LON_ACCEL} m/s² in {t_ramp4*1000:.0f} ms, then holds constant",
    color_traj="#37474F", mirror=True,
)
axes[4, 1].set_title(
    f"Threshold = {MAX_LON_JERK} m/s³\n"
    f"≈ max accel reached in {t_ramp4*1000:.0f} ms  |  deliberate smooth pull-away",
    fontsize=8
)

# ─────────────────────────────────────────────────────────────────────────────
# Row 5 │ Yaw rate (±0.95 rad/s) + Yaw acceleration (±1.93 rad/s²)
# Scenario: yaw rate ramps from 0 to MAX_YAW_RATE over t_ramp_yaw, then holds.
#   Yaw acceleration = Δyaw_rate / Δt → t_ramp_yaw = 0.95 / 1.93 ≈ 0.49 s.
# At 10 m/s speed, steady-state radius r = v / ω = 10 / 0.95 ≈ 10.5 m.
# Real-world feel: sharp U-turn in a wide street, or a fast exit from a
#   tight roundabout.
# ─────────────────────────────────────────────────────────────────────────────
v_yaw        = 10.0            # m/s (≈ 36 km/h)
t_ramp5      = MAX_YAW_RATE / MAX_YAW_ACCEL   # ≈ 0.49 s
yaw_rate5    = np.where(time <= t_ramp5,
                         (MAX_YAW_RATE / t_ramp5) * time,
                         MAX_YAW_RATE)
heading5     = euler_integrate(yaw_rate5, init=0.0)
speed5       = np.full(N, v_yaw)
x5, y5       = xy_from_speed_heading(speed5, heading5)

# yaw_accel is the derivative of yaw_rate
yaw_accel5   = sg_diff(phase_unwrap(heading5), deriv=2, window=15, poly=3)

# combine yaw rate and yaw accel in one row — left: yaw rate, right: yaw accel
ax_xy5   = axes[5, 0]
ax_ts5   = axes[5, 1]

r_steady = v_yaw / MAX_YAW_RATE

ax_xy5.plot(x5, y5, color="#880E4F", lw=2)
ax_xy5.plot(x5[0], y5[0], "go", ms=8, label="start")
ax_xy5.plot(x5[-1], y5[-1], "rs", ms=8, label="end")
ax_xy5.set_aspect("equal")
ax_xy5.set_xlabel("x [m]"); ax_xy5.set_ylabel("y [m]")
ax_xy5.legend(fontsize=7)
ax_xy5.grid(True, alpha=0.3)
ax_xy5.set_title(
    f"v = {v_yaw*3.6:.0f} km/h  |  steady-state radius = {r_steady:.1f} m\n"
    f"yaw ramps 0→{MAX_YAW_RATE} rad/s in {t_ramp5*1000:.0f} ms",
    fontsize=8
)

# plot both metrics on the right panel
ax_ts5.plot(time, yaw_rate5, color="#880E4F", lw=2, label="yaw rate [rad/s]")
ax_ts5.axhline(MAX_YAW_RATE, color="red", lw=1.5, ls="--",
               label=f"yaw rate threshold ±{MAX_YAW_RATE} rad/s ({np.degrees(MAX_YAW_RATE):.0f}°/s)")
ax_ts5.plot(time, yaw_accel5, color="#F48FB1", lw=2, ls="-.", label="yaw accel [rad/s²]")
ax_ts5.axhline(MAX_YAW_ACCEL, color="orange", lw=1.5, ls="--",
               label=f"yaw accel threshold ±{MAX_YAW_ACCEL} rad/s²")
ax_ts5.axhline(-MAX_YAW_RATE,  color="red",    lw=1.0, ls=":")
ax_ts5.axhline(-MAX_YAW_ACCEL, color="orange", lw=1.0, ls=":")
ax_ts5.set_xlabel("time [s]")
ax_ts5.set_ylabel("value")
ax_ts5.legend(fontsize=7, loc="center right")
ax_ts5.grid(True, alpha=0.3)
ax_ts5.set_title(
    f"Yaw rate threshold = {MAX_YAW_RATE} rad/s ≈ {np.degrees(MAX_YAW_RATE):.0f}°/s\n"
    f"Yaw accel threshold = {MAX_YAW_ACCEL} rad/s²  |  tight U-turn / fast roundabout exit",
    fontsize=8
)

# ── Real-world summary table ──────────────────────────────────────────────────
summary_text = (
    "Real-world equivalents at threshold\n"
    "────────────────────────────────────────────────────────\n"
    f"lon. accel  +{MAX_LON_ACCEL} m/s²  →  0 → 43 km/h in 5 s  (moderate city pull-away)\n"
    f"lon. accel  {MIN_LON_ACCEL} m/s²  →  60 → 0 km/h in 4.1 s  (firm controlled emergency stop)\n"
    f"lat. accel  ±{MAX_LAT_ACCEL} m/s²  →  30 km/h on a 14 m radius corner  (tight city turn)\n"
    f"mag. jerk   ±{MAX_MAG_JERK} m/s³   →  full accel applied in ~290 ms  (brisk floor-it)\n"
    f"lon. jerk   ±{MAX_LON_JERK} m/s³   →  full accel applied in ~580 ms  (smooth deliberate ramp)\n"
    f"yaw rate    ±{MAX_YAW_RATE} rad/s   →  {np.degrees(MAX_YAW_RATE):.0f}°/s  |  at 36 km/h: r = 10.5 m  (tight U-turn)\n"
    f"yaw accel   ±{MAX_YAW_ACCEL} rad/s²  →  max yaw rate reached in ~490 ms  (sharp turn entry)\n"
)
print(summary_text)

# ── Save figure ───────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0, 1, 0.995])
import os
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history_comfort_thresholds.png")
plt.savefig(out_path, dpi=130, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.show()
