# ==========================================================================
# Validate the FRE implementation against Fig. 5 of Pietras et al. (2025)
# ==========================================================================

import numpy as np
import matplotlib.pyplot as plt
from Pietras_retallat import simulate_fre_4d, Params, piecewise_eta_bar
import Pietras2025

# My model
# Parameters
tau_m_ms = 10.0
dt_ms = 0.01
dtp = dt_ms / tau_m_ms

# We use a 1x1 SC matrix of zeros and g=0 so coupling is always zero.
# This reproduces the single-population case of the paper exactly.
sc_norm = np.array([[0.0]])
model_low = Pietras2025.Pietras2025(eta_e=-1.74)
model_low.configure(weights=sc_norm, g=0.0)
dfun_low = model_low.get_numba_dfun()

model_mid = Pietras2025.Pietras2025(eta_e=0.0)
model_mid.configure(weights=sc_norm, g=0.0)
dfun_mid = model_mid.get_numba_dfun()

state = model_low.initial_state(n_rois=1)
coupling_zero = np.zeros((1, 1))
n_steps = int(4000.0 / dt_ms) # Paper: n_steps = int(np.round(p.t_end_s / p.dt_s)), t_end_s = 4.0
tp = np.arange(n_steps + 1) * dtp
t = tp * tau_m_ms # temps en ms

# Arrays per guardar el que retorna el dfun excitarori 
R_e_our = np.empty_like(tp)
V_e_our = np.empty_like(tp)
A_e_our = np.empty_like(tp)

R_e_our[0] = state[0, 0]
V_e_our[0] = state[1, 0]
A_e_our[0] = state[2, 0]

# Arrays per guardar el que retorna el dfun inhibitori 
R_i_our = np.empty_like(tp)
V_i_our = np.empty_like(tp)
A_i_our = np.empty_like(tp)

R_i_our[0] = state[4, 0]
V_i_our[0] = state[5, 0]
A_i_our[0] = state[6, 0]

for k in range(n_steps):
    dfun  = dfun_mid if (1500.0 < t[k] <= 2500.0) else dfun_low

    d_state, observed = dfun(state, coupling_zero)
    # d_state  shape (8,1): [dR_e, dV_e, dA_e, dB_e, dR_i, dV_i, dA_i, dB_i]
    # observed shape (2,1): [R_e_Hz, R_i_Hz]

    # Euler step
    state = state + dtp * d_state

    # Guardem directament de l'estat (variables adimensionals)
    R_e_our[k+1] = state[0, 0]
    V_e_our[k+1] = state[1, 0]
    A_e_our[k+1] = state[2, 0]
    R_i_our[k+1] = state[4, 0]
    V_i_our[k+1] = state[5, 0]
    A_i_our[k+1] = state[6, 0]

t_s  = t / 1000.0 
R_e_Hz = R_e_our / tau_m_ms * 1000.0
R_i_Hz = R_i_our / tau_m_ms * 1000.0

# Paper
p = Params() 
t_fre, r_fre, v_fre, a_fre = simulate_fre_4d(p)
hz_factor = 1 / p.tau_m_s
eta_plot = piecewise_eta_bar(t_fre, eta_low=p.eta_low, eta_mid=p.eta_mid, t1=p.t1, t2=p.t2)

fig, axs = plt.subplots(
    4, 1, figsize=(7.6, 6.0), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.2, 1.2, 0.9]}
)
fig.suptitle("Reference (blue) vs Our model (red)")
axs[0].plot(t_fre, hz_factor * r_fre, color="blue", lw=1.0, label="Pietras 4D mean-field")
axs[0].plot(t_s, R_e_Hz, color="red", lw=1.0, label="Our model", linestyle="--")
axs[0].set_ylabel(r"$R(t)$ [Hz]")
axs[0].set_ylim(0, 220)
axs[0].legend(frameon=False, loc="upper right")
axs[0].grid(True, alpha=0.25)

axs[1].plot(t_fre, v_fre, color="blue", lw=1.0)
axs[1].plot(t_s, V_e_our,  color="red",  lw=1.0, linestyle="--")
axs[1].set_ylabel(r"$V(t)$")
axs[1].grid(True, alpha=0.25)

axs[2].plot(t_fre, a_fre, color="blue", lw=1.0)
axs[2].plot(t_s, A_e_our, color="red", lw=1.0, linestyle="--")
axs[2].set_ylabel(r"$a(t)$")
axs[2].grid(True, alpha=0.25)

axs[3].plot(t_fre, eta_plot, color="green", lw=1.6)
axs[3].set_ylabel(r"$\bar\eta$")
axs[3].set_xlabel(r"Time $t$ [s]")
axs[3].grid(True, alpha=0.25)

fig.tight_layout()
fig.savefig("validate_pietras2025.png", bbox_inches="tight")
plt.show()
