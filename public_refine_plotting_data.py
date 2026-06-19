"""
public_refine_plotting_data.py
────────────────────────────────────────────────────────────────────────────────
Numerically refines selected physics parameters so that the model best matches
a chosen target.

USAGE
  1.  Set the refinement margins below.
          <param>_ref = <margin>   → refine that param within ±<margin>
          <param>_ref = None       → keep param fixed

  2.  Choose TARGET:
          "data"   → minimise residuals against TARGET_COL of data.txt
          "curve"  → minimise residuals against the power-law fit to data[:,4]

  3.  Run this file directly.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize, curve_fit, brentq
import os

# SetTex
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'

# ── Data ─────────────────────────────────────────────────────────────────────

_dir = os.path.dirname(__file__)
data = np.loadtxt(os.path.join(_dir, "data.txt"))
SAVE = False

# ── Refinement target ────────────────────────────────────────────────────────
TARGET = "data"  # "data" | "curve"
TARGET_COL = 4  # column index used when TARGET = "data"

# ── Base parameter values ────────────────────────────────────────────────────
sigma = 5.67e-8
alpha = 14e-6
epsilon = 0.72
N = 176.81
R0 = 10.08
Tk = 300
N_TEMP_DEP = True  # False → constant N,  True → N·(ΔT/Tk)^n_conv
n_conv = 0
r0 = 0.00020 / 2

a_g = np.sqrt(0.009 ** 2 + (0.09 + 0.002) ** 2)  # 0.093
b_g = np.sqrt(0.009 ** 2 + (0.09 - 0.002) ** 2)  # 0.092
c_g = 0.18
l0 = a_g + b_g  # must equal a_g + b_g for m1=m0 at zero current

# ── Refinement margins (set to None to keep fixed) ───────────────────────────
# Example: a_g_ref = 1e-3  →  a_g is free within [a_g - 1e-3, a_g + 1e-3]
a_g_ref = 0.001
b_g_ref = 0.001
c_g_ref = 0.005
alpha_ref = None
epsilon_ref = None
N_ref = None
R0_ref = None
r0_ref = 0.00002
l0_ref = None
n_conv_ref = None  # set to None to keep n_conv fixed

# ─────────────────────────────────────────────────────────────────────────────
# Everything below runs automatically — no editing needed past this line.
# ─────────────────────────────────────────────────────────────────────────────

PARAM_NAMES = ["a_g", "b_g", "c_g", "alpha", "epsilon", "N", "R0", "r0", "l0", "n_conv"]
BASE_VALUES = [a_g, b_g, c_g, alpha, epsilon, N, R0, r0, l0, n_conv]
MARGINS = [a_g_ref, b_g_ref, c_g_ref, alpha_ref, epsilon_ref, N_ref, R0_ref, r0_ref, l0_ref, n_conv_ref]


def _geometry(a_g, b_g, c_g):
    """Precompute geometry constants that depend only on side lengths."""
    cos_A0 = (-a_g ** 2 + b_g ** 2 + c_g ** 2) / (2 * c_g * b_g)
    m0 = b_g * np.sqrt(max(1 - cos_A0 ** 2, 0))
    cb = np.sqrt(max(b_g ** 2 - m0 ** 2, 0))
    return m0, cb


def _equilibrium_length(I, alpha, epsilon, N, R0, l0, r0, n_conv):
    def f(l):
        ratio = np.log(l / l0)
        A = 2 * np.pi * r0 * l0 * (l / l0) ** 2
        with np.errstate(over='ignore', invalid='ignore'):
            dT = ratio / alpha
            N_eff = N * (dT / Tk) ** n_conv if N_TEMP_DEP else N
            lhs = A * (N_eff * dT + epsilon * sigma * ((dT + Tk) ** 4 - Tk ** 4))
        if not np.isfinite(lhs):
            return np.inf
        return lhs - I ** 2 * R0

    l_upper = l0 * 1.01
    for _ in range(200):
        if f(l_upper) > 0:
            break
        l_upper *= 1.1
    else:
        raise ValueError("Could not bracket upper bound")
    return brentq(f, l0 + 1e-12, l_upper, xtol=1e-10)


def model_angles(I_arr, a_g, b_g, c_g, alpha, epsilon, N, R0, r0, l0=None, n_conv=0.25):
    """Return deflection angles (degrees) for each current in I_arr."""
    l0 = a_g + b_g  # always derived from geometry
    m0, cb = _geometry(a_g, b_g, c_g)
    K = 2 * c_g * cb - c_g ** 2
    out = []
    for I in I_arr:
        try:
            l1 = _equilibrium_length(I, alpha, epsilon, N, R0, l0, r0, n_conv)
            b1 = (l1 ** 2 + K) / (2 * l1)
            m1 = np.sqrt(max(b1 ** 2 - cb ** 2, 0))
            dm = m1 - m0
            out.append(dm / 0.00105 * 2 / np.pi * 180)  # d = 0.00105 m
        except Exception:
            out.append(np.nan)
    return np.array(out)


def _build_target():
    if TARGET == "curve":
        def power_law(x, a, b):
            return a * x ** b

        popt, _ = curve_fit(power_law, data[:, 0], data[:, 4])
        return power_law(data[:, 0], *popt)
    else:
        return data[:, TARGET_COL]


def _residuals(x_free, free_idx, target):
    params = BASE_VALUES.copy()
    for i, idx in enumerate(free_idx):
        params[idx] = x_free[i]
    a_g_, b_g_, c_g_, alpha_, epsilon_, N_, R0_, r0_, l0_, n_conv_ = params
    y = model_angles(data[:, 0], a_g_, b_g_, c_g_, alpha_, epsilon_, N_, R0_, r0_, l0_, n_conv_)
    diff = y - target
    diff = diff[np.isfinite(diff)]
    return np.sum(diff ** 2)


def refine():
    target = _build_target()

    free_idx = [i for i, m in enumerate(MARGINS) if m is not None]
    x0 = [BASE_VALUES[i] for i in free_idx]
    bounds = [(BASE_VALUES[i] - MARGINS[i], BASE_VALUES[i] + MARGINS[i])
              for i in free_idx]

    if not free_idx:
        print("No parameters marked for refinement — set at least one <param>_ref margin.")
        return

    print("Refining parameters:")
    for i in free_idx:
        lo, hi = bounds[free_idx.index(i)]
        print(f"  {PARAM_NAMES[i]:7s}  base={BASE_VALUES[i]: <10.6g}  "
              f"range=[{lo:.6g}, {hi:.6g}]")
    print(f"\nTarget: {TARGET}" + (f" (col {TARGET_COL})" if TARGET == "data" else ""))

    result = minimize(
        _residuals,
        x0,
        args=(free_idx, target),
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 10_000},
    )

    print("\n── Results ─────────────────────────────────────────────────────────")
    refined_params = BASE_VALUES.copy()
    for i, idx in enumerate(free_idx):
        refined_params[idx] = result.x[i]
        delta = result.x[i] - BASE_VALUES[idx]
        print(f"  {PARAM_NAMES[idx]:7s}  {BASE_VALUES[idx]: <10.6g}  →  {result.x[i]: <9.6g}  "
              f"(Δ = {delta:+.4g})")

    y_base = model_angles(data[:, 0], *BASE_VALUES)
    y_refined = model_angles(data[:, 0], *refined_params)  # type: ignore[arg-type]

    def _rms(y):
        valid = np.isfinite(y)
        if not valid.any():
            return float('nan'), 0
        return np.sqrt(np.mean((y[valid] - target[valid]) ** 2)), valid.sum()

    rms_base, n_base = _rms(y_base)
    rms_refined, n_refined = _rms(y_refined)
    print(f"\n  RMS before: {rms_base:.4g}  ({n_base}/{len(target)} points valid)")
    print(f"  RMS after:  {rms_refined:.4g}  ({n_refined}/{len(target)} points valid)")
    print(f"  Converged:  {result.success}  ({result.message})")

    _plot(refined_params, target)
    return refined_params


def _plot(refined_params, target):
    I_dense = np.linspace(data[:, 0].min(), data[:, 0].max(), 200)

    y_model_base = model_angles(I_dense, *BASE_VALUES)
    y_model_refined = model_angles(I_dense, *refined_params)
    y_at_data_refined = model_angles(data[:, 0], *refined_params)

    def power_law(x, a, b):
        return a * x ** b

    popt, pcov = curve_fit(power_law, data[:, 0], data[:, 4])
    perr = np.sqrt(np.diag(pcov))
    a_fit, b_fit = popt
    y_fit = power_law(I_dense, *popt)

    ratio = 2
    plt.figure(figsize=(16 / ratio, 9 / ratio))
    plt.plot(data[:, 0], target, "o", zorder=3,
             label="measured data")
    plt.plot(data[:, 0], y_at_data_refined, "s", color='g', zorder=2, markersize=5,
             label="refined model at data points")
    plt.plot(I_dense, y_model_base, "--", color='gray', zorder=1, alpha=0.6,
             label="base model")
    plt.plot(I_dense, y_model_refined, "-", color='g', zorder=2,
             label="refined model")
    plt.plot(I_dense, y_fit, "--", color='r', zorder=1,
             label=fr"fit: ${a_fit:.3g} \cdot I^{{{b_fit:.3f}}}$, \textsc{{target}}")

    plt.title(r'\textbf{Plot of the optimised values, with the old formula}', size=16)
    plt.xlabel("$I$ (A)", size=14)
    plt.ylabel(r"\textsc{Deflection (deg)}", size=14)
    plt.legend(fontsize=12)
    plt.tight_layout()

    print(f"\n{'I':>6}  {'col4':>10}  {'model':>10}  {'c4-mdl':>10}")
    for I, c4, c5, m in zip(data[:, 0], data[:, 4], data[:, 5], y_at_data_refined):
        print(f"{I:6.3f}  {c4:10.3f}  {m:10.3f}  {c4 - m:10.3f}")

    if SAVE:
        out_dir = os.path.dirname(os.path.abspath('./Out') + '/Out')
        plt.savefig(os.path.join(out_dir, 'p_refined_plot_of_data.png'), dpi=600)
    else:
        plt.show()


if __name__ == "__main__":
    refine()
