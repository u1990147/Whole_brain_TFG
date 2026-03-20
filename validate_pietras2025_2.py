# ==========================================================================
# Validate the FRE implementation against Fig. 5 of Pietras et al. (2025)
# ==========================================================================

import numpy as np
import matplotlib.pyplot as plt
import pietras2025_2

# Create model
sc_norm = np.array([[0]])
model = pietras2025_2.Pietras2025()
dfun = model.get_numba_dfun()


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
