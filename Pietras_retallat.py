"""
Figure 5 — theta-network simulation vs transient 4D mean-field, Python version.

Microscopic theta-network (paper / user-corrected equations):
  tau_m * dtheta_j/dt = 1 - cos(theta_j) + (1 + cos(theta_j)) * (eta_j - a_j + J*tau_m*R(t))
  tau_a * da_j/dt     = -a_j + beta * (eta_j - a_j + J*tau_m*R(t))

Rate extraction (Appendix / conformal map):
  Z(t) = (1/N) sum_j exp(i theta_j)
  r(t') = (1/pi) Re{ (1 - conj(Z)) / (1 + conj(Z)) }   [dimensionless rate]
  R_Hz(t) = r(t') / tau_m

Important implementation detail:
  The mean-field adaptation variable corresponds to the Lorentzian *location* of the a_j distribution.
  We therefore compare mean-field a(t) against the network *median* of a_j (robust estimator of location).

Usage:
  python simulations/fig5_network_vs_meanfield.py --N 4000 --show
  python simulations/fig5_network_vs_meanfield.py --out simulations/fig5_python.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def wrap(f):
            return f

        return wrap


@dataclass(frozen=True)
class Params:
    # Paper / caption
    tau_m_s: float = 0.01
    tau_a_s: float = 0.1
    Delta: float = 1.0
    J: float = 10.0
    beta: float = 1.0
    eta_low: float = -1.74
    eta_mid: float = 0.0
    t1: float = 1.5
    t2: float = 2.5

    # Numerics (match Julia: dt' = 1e-3 with tau_m=10ms -> dt_s=1e-5)
    dt_s: float = 1e-5
    t_end_s: float = 4.0
    save_every: int = 100  # 1 ms when dt_s=1e-5

    # Initial Lorentzian voltage distribution parameters
    r0: float = 0.10
    v0: float = -0.20


def piecewise_eta_bar(t_sec, eta_low=-1.74, eta_mid=0.0, t1=1.5, t2=2.5):
    t_sec = np.asarray(t_sec)
    out = np.full_like(t_sec, fill_value=eta_low, dtype=float)
    out[(t_sec > t1) & (t_sec <= t2)] = eta_mid
    return out


def rk4_step(fun, y, t, dt):
    k1 = fun(t, y)
    k2 = fun(t + dt / 2, y + dt * k1 / 2)
    k3 = fun(t + dt / 2, y + dt * k2 / 2)
    k4 = fun(t + dt, y + dt * k3)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_fre_4d(p: Params, y0=(0.10, -0.20, 0.0, 0.0)):
    """Transient 4D mean-field in dimensionless time t' = t/tau_m."""
    dtp = p.dt_s / p.tau_m_s
    n_steps = int(np.round(p.t_end_s / p.dt_s))
    tp = np.arange(n_steps + 1) * dtp
    t = tp * p.tau_m_s

    r = np.empty_like(tp)
    v = np.empty_like(tp)
    a = np.empty_like(tp)
    b = np.empty_like(tp)
    r[0], v[0], a[0], b[0] = y0

    tau = p.tau_a_s / p.tau_m_s  # 10 for Fig. 5

    def eta_bar(tsec):
        return p.eta_mid if (tsec > p.t1 and tsec <= p.t2) else p.eta_low

    for k in range(n_steps):
        ebar = eta_bar(t[k])

        def f(_tp, y):
            rr, vv, aa, bb = y
            drr = (p.Delta + bb) / np.pi + 2 * rr * vv
            dvv = vv**2 - (np.pi * rr) ** 2 + ebar + p.J * rr - aa
            daa = (p.beta * (ebar + p.J * rr) - (1 + p.beta) * aa) / tau
            dbb = (-(1 + p.beta) * bb - p.beta * p.Delta) / tau
            return np.array([drr, dvv, daa, dbb], dtype=float)

        y = np.array([r[k], v[k], a[k], b[k]], dtype=float)
        y_next = rk4_step(f, y, tp[k], dtp)
        r[k + 1], v[k + 1], a[k + 1], b[k + 1] = y_next

    return t, r, v, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=4000, help="Network size (paper uses 10000)")
    ap.add_argument("--out", default="fig5_python.png", help="Output image path")
    ap.add_argument("--show", action="store_true", help="Show the figure window")
    args = ap.parse_args()

    if not args.show:
        import matplotlib

        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt

    p = Params()
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is required for Fig. 5 script speed. Install numba or run the notebook.")

    # mean-field
    t_fre, r_fre, v_fre, a_fre = simulate_fre_4d(p)
    
   
    hz_factor = 1 / p.tau_m_s
    eta_plot = piecewise_eta_bar(t_fre, eta_low=p.eta_low, eta_mid=p.eta_mid, t1=p.t1, t2=p.t2)


    fig, axs = plt.subplots(
        4, 1, figsize=(7.6, 6.0), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.2, 1.2, 0.9]}
    )

    axs[0].plot(t_fre, hz_factor * r_fre, color="red", lw=1.0, label="4D mean-field")
    axs[0].set_ylabel(r"$R(t)$ [Hz]")
    axs[0].set_ylim(0, 220)
    axs[0].legend(frameon=False, loc="upper right")
    axs[0].grid(True, alpha=0.25)

    axs[1].plot(t_fre, v_fre, color="red", lw=1.0)
    axs[1].set_ylabel(r"$V(t)$")
    axs[1].grid(True, alpha=0.25)

    axs[2].plot(t_fre, a_fre, color="red", lw=1.0)
    axs[2].set_ylabel(r"$a(t)$")
    axs[2].grid(True, alpha=0.25)

    axs[3].plot(t_fre, eta_plot, color="green", lw=1.6)
    axs[3].set_ylabel(r"$\bar\eta$")
    axs[3].set_xlabel(r"Time $t$ [s]")
    axs[3].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()


