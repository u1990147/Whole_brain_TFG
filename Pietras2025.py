# ==========================================================================
# ==========================================================================
# 4D Transient Mean-Field model for QIF neurons with Quadratic Spike-Frequency
# Adaptation (QSFA), based on:
#
# [Pietras_2025] B. Pietras, P. Clusella, E. Montbrió
#               "Low-dimensional model for adaptive networks of spiking neurons"
#               Physical Review E 111, 014422 (2025)
#               DOI: 10.1103/PhysRevE.111.014422
#
# And its Erratum, which specifies the exact 4D transient system (part c):
#
#   τm Ṙ = (Δ + B) / (π τm) + 2 R V
#   τm V̇ = V² − (π τm R)² + η̄(t) + J τm R − A
#   τa Ȧ = −(1 + β) A + β [η̄(t) + J τm R]
#   τa Ḃ = −(1 + β) B − β Δ
#
# Note: R above is dimensionless (units 1/τm); physical firing rate in Hz is
#       R_Hz = R / τm.  The asymptotic B → −βΔ/(1+β) recovers the 3D FRE.
#
# This implementation extends the model to two coupled populations
# (excitatory E and inhibitory I), following the two-population conventions
# of the Deco2014 model in this library. Inter-region coupling is carried
# via the excitatory firing rate R_e (dimensionless), weighted by the
# structural connectivity matrix and global coupling g (LinearCouplingModel).
#
# State variables (per ROI):
#   R_e : dimensionless excitatory firing rate  (= τm * R_Hz_e)
#   V_e : mean excitatory membrane potential
#   A_e : real part of excitatory complex adaptation order parameter
#   B_e : imaginary part of excitatory complex adaptation order parameter
#   R_i : dimensionless inhibitory firing rate
#   V_i : mean inhibitory membrane potential
#   A_i : real part of inhibitory complex adaptation order parameter
#   B_i : imaginary part of inhibitory complex adaptation order parameter
#
# Observable variables (per ROI):
#   R_e_Hz : excitatory firing rate in Hz  (= R_e / tau_m)
#   R_i_Hz : inhibitory firing rate in Hz  (= R_i / tau_m)
#
# Coupling variable:
#   R_e : only the excitatory dimensionless firing rate is communicated
#         across regions (analogous to S_e in Deco2014).
#
# ==========================================================================
# ==========================================================================

import numpy as np
import numba as nb
from overrides import overrides

from neuronumba.basic.attr import Attr
from neuronumba.numba_tools.types import NDA_f8_2d
from neuronumba.simulator.models import LinearCouplingModel, Model
from neuronumba.numba_tools.config import NUMBA_CACHE, NUMBA_FASTMATH, NUMBA_NOGIL


class Pietras2025(LinearCouplingModel):
    """
    4D Transient Mean-Field model for adaptive QIF networks (Pietras et al. 2025).

    Implements the exact low-dimensional firing rate equations (FREs) for a
    large population of heterogeneous quadratic integrate-and-fire (QIF) neurons
    with quadratic spike-frequency adaptation (QSFA), extended here to two
    coupled populations (excitatory and inhibitory).

    The model tracks four mean-field variables per population:
        R  – dimensionless firing rate (physical rate R_Hz = R / τm)
        V  – mean membrane potential
        A  – real part of the complex adaptation order parameter (mean adaptation)
        B  – imaginary part (captures transient heterogeneity relaxation)

    Asymptotically B → −β Δ / (1+β), so Δ + B → Δ/(1+β), recovering the 3D FRE.

    Inter-region coupling uses LinearCouplingModel (SC-weighted, scaled by g),
    and only the excitatory R_e is communicated across regions.

    Parameters
    ----------
    Shared timescales
        tau_m   : membrane time constant [s]  (default 10 ms = 0.01 s)
        tau_a_e : excitatory adaptation time constant [s]  (default 100 ms)
        tau_a_i : inhibitory adaptation time constant [s]  (default  20 ms)

    Excitatory population
        Delta_e : Lorentzian half-width of η_e distribution (heterogeneity)
        eta_e   : mean external drive η̄_e  (can vary externally)
        J_ee    : excitatory-to-excitatory recurrent coupling
        J_ei    : inhibitory-to-excitatory coupling (negative feedback; sign
                  convention: subtracted in V̇_e equation)
        beta_e  : QSFA adaptation strength for excitatory neurons

    Inhibitory population
        Delta_i : Lorentzian half-width of η_i distribution
        eta_i   : mean external drive η̄_i
        J_ie    : excitatory-to-inhibitory coupling
        J_ii    : inhibitory self-coupling (subtracted in V̇_i)
        beta_i  : QSFA adaptation strength for inhibitory neurons

    Notes
    -----
    All variables and parameters use SI units (seconds), consistent with the
    Erratum convention where τm is expressed in seconds.  The structural
    connectivity matrix (weights) is assumed to be already normalised; the
    global coupling g (inherited from LinearCouplingModel) scales it.

    References
    ----------
    Pietras B., Clusella P., Montbrió E. (2025) Phys. Rev. E 111, 014422.
    Erratum: specification of the 4D transient mean-field system.
    """

    # ------------------------------------------------------------------
    # Variable name declarations (required by Model base class)
    # ------------------------------------------------------------------

    _state_var_names = ['R_e', 'V_e', 'A_e', 'B_e',
                        'R_i', 'V_i', 'A_i', 'B_i']

    _coupling_var_names = ['R_e']          # only excitatory rate couples regions

    _observable_var_names = ['R_e_Hz', 'R_i_Hz']

    # ------------------------------------------------------------------
    # Model Parameters — REGIONAL (packed into self.m, one value per ROI)
    # ------------------------------------------------------------------

    # --- Membrane time constant (same for both populations) ---
    tau_m = Attr(default=0.01, attributes=Model.Tag.REGIONAL,
                 doc="Membrane time constant [s] (default: 10 ms)")

    # --- Excitatory population parameters ---
    tau_a_e = Attr(default=0.10, attributes=Model.Tag.REGIONAL,
                   doc="Excitatory adaptation time constant [s] (default: 100 ms)")

    Delta_e = Attr(default=1.0, attributes=Model.Tag.REGIONAL,
                   doc="Lorentzian half-width of excitatory η distribution (heterogeneity)")

    eta_e = Attr(default=-1.74, attributes=Model.Tag.REGIONAL,
                 doc="Mean excitatory external drive η̄_e")

    J_ee = Attr(default=10.0, attributes=Model.Tag.REGIONAL,
                doc="Excitatory-to-excitatory recurrent synaptic weight")

    J_ei = Attr(default=5.0, attributes=Model.Tag.REGIONAL,
                doc="Inhibitory-to-excitatory synaptic weight (subtracted in dV_e)")

    beta_e = Attr(default=1.0, attributes=Model.Tag.REGIONAL,
                  doc="Excitatory QSFA adaptation strength β_e")

    # --- Inhibitory population parameters ---
    tau_a_i = Attr(default=0.10, attributes=Model.Tag.REGIONAL,
                   doc="Inhibitory adaptation time constant [s] (default: 100 ms)")

    Delta_i = Attr(default=1.0, attributes=Model.Tag.REGIONAL,
                   doc="Lorentzian half-width of inhibitory η distribution (heterogeneity)")

    eta_i = Attr(default=-1.74, attributes=Model.Tag.REGIONAL,
                 doc="Mean inhibitory external drive η̄_i")

    J_ie = Attr(default=10.0, attributes=Model.Tag.REGIONAL,
                doc="Excitatory-to-inhibitory synaptic weight")

    J_ii = Attr(default=2.0, attributes=Model.Tag.REGIONAL,
                doc="Inhibitory self-coupling weight (subtracted in dV_i)")

    beta_i = Attr(default=1.0, attributes=Model.Tag.REGIONAL,
                  doc="Inhibitory QSFA adaptation strength β_i")

    # ------------------------------------------------------------------
    # Initial-state helper
    # ------------------------------------------------------------------

    @overrides
    def initial_state(self, n_rois: int) -> np.ndarray:
        """
        Return initial state consistent with the Erratum prescription:
            R(0) = r0,  V(0) = v0,  A(0) = 0,  B(0) = 0
        for both populations.
        """
        state = np.zeros((Pietras2025.n_state_vars, n_rois))
        # Excitatory
        state[0] = 0.10   # R_e  (dimensionless, ~ r0 from Erratum)
        state[1] = -0.20  # V_e  (~ v0 from Erratum)
        state[2] = 0.0    # A_e
        state[3] = 0.0    # B_e
        # Inhibitory — start from low-activity state
        state[4] = 0.10   # R_i
        state[5] = -0.20  # V_i
        state[6] = 0.0    # A_i
        state[7] = 0.0    # B_i
        return state

    # ------------------------------------------------------------------
    # Numba-compiled differential equations
    # ------------------------------------------------------------------

    @overrides
    def get_numba_dfun(self):
        """
        Return the Numba-JIT differential function implementing the 4D
        transient mean-field equations for E and I populations.

        The equations (from the Erratum, part c, adapted for two populations)
        are — for the excitatory population:

            τm dR_e/dt = (Δ_e + B_e) / (π τm) + 2 R_e V_e
            τm dV_e/dt = V_e² − (π τm R_e)² + η̄_e + J_ee τm R_e
                         − J_ei τm R_i − A_e + coupling_e
            τa_e dA_e/dt = −(1+β_e) A_e + β_e [η̄_e + J_ee τm R_e − J_ei τm R_i + coupling_e]
            τa_e dB_e/dt = −(1+β_e) B_e − β_e Δ_e

        and symmetrically for the inhibitory population (with J_ie, J_ii,
        Δ_i, η̄_i, β_i, τa_i), where inhibitory neurons receive only local
        excitatory drive (no long-range inter-region coupling):

            τm dR_i/dt = (Δ_i + B_i) / (π τm) + 2 R_i V_i
            τm dV_i/dt = V_i² − (π τm R_i)² + η̄_i + J_ie τm R_e
                         − J_ii τm R_i − A_i
            τa_i dA_i/dt = −(1+β_i) A_i + β_i [η̄_i + J_ie τm R_e − J_ii τm R_i]
            τa_i dB_i/dt = −(1+β_i) B_i − β_i Δ_i

        coupling_e is the long-range excitatory input from other regions,
        computed by LinearCouplingModel as  g * (W^T @ R_e).

        The coupling is added to the effective drive of V_e and A_e because
        it enters the QIF model identically to η̄_e (it is part of I_j(t) in
        equation (6) of the paper: I_j = J τm R(t) + η_j).

        Parameters of the dfun are read from the pre-built parameter matrix
        self.m at the row indices given by the P enum.
        """
        m = self.m.copy()
        P = self.P
        pi = np.pi

        @nb.njit(nb.types.UniTuple(nb.f8[:, :], 2)(nb.f8[:, :], nb.f8[:, :]),
                 cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
        def Pietras2025_dfun(state: NDA_f8_2d, coupling: NDA_f8_2d):
            """
            Compute state derivatives and observables for the Pietras2025 model.

            Parameters
            ----------
            state    : (8, n_rois)  [R_e, V_e, A_e, B_e, R_i, V_i, A_i, B_i]
            coupling : (1, n_rois)  long-range excitatory input g * W^T @ R_e

            Returns
            -------
            d_state  : (8, n_rois)  time derivatives
            observed : (2, n_rois)  [R_e_Hz, R_i_Hz]
            """
            # --- Unpack state ---
            R_e = state[0]   # dimensionless excitatory firing rate
            V_e = state[1]   # excitatory mean voltage
            A_e = state[2]   # excitatory adaptation (real part of α)
            B_e = state[3]   # excitatory adaptation (imag part of α)
            R_i = state[4]   # dimensionless inhibitory firing rate
            V_i = state[5]   # inhibitory mean voltage
            A_i = state[6]   # inhibitory adaptation (real part)
            B_i = state[7]   # inhibitory adaptation (imag part)

            # --- Unpack parameters ---
            tau_m   = m[np.intp(P.tau_m)]
            tau_a_e = m[np.intp(P.tau_a_e)]
            Delta_e = m[np.intp(P.Delta_e)]
            eta_e   = m[np.intp(P.eta_e)]
            J_ee    = m[np.intp(P.J_ee)]
            J_ei    = m[np.intp(P.J_ei)]
            beta_e  = m[np.intp(P.beta_e)]

            tau_a_i = m[np.intp(P.tau_a_i)]
            Delta_i = m[np.intp(P.Delta_i)]
            eta_i   = m[np.intp(P.eta_i)]
            J_ie    = m[np.intp(P.J_ie)]
            J_ii    = m[np.intp(P.J_ii)]
            beta_i  = m[np.intp(P.beta_i)]

            # --- Long-range excitatory coupling input ---
            # coupling[0] = g * (W^T @ R_e), shape (n_rois,)
            c_e = coupling[0]

            # -------------------------------------------------------
            # Excitatory population (4D transient FRE, Erratum part c)
            # -------------------------------------------------------

            # Effective drive entering both V̇_e and A_e equation
            #   I_eff_e = η̄_e + J_ee * τm * R_e  - J_ei * τm * R_i  + long-range
            # Note: τm * R is dimensionless×s → same unit as η̄
            #I_eff_e = eta_e + J_ee * tau_m * R_e - J_ei * tau_m * R_i + c_e
            I_eff_e = J_ee * tau_m * R_e - J_ei * tau_m * R_i + J_ee * tau_m * c_e
        

            # τm dR_e/dt = (Δ_e + B_e) / (π τm) + 2 R_e V_e
            dR_e = ((Delta_e + B_e) / (pi * tau_m) + 2.0 * R_e * V_e) / tau_m

            # τm dV_e/dt = V_e² − (π τm R_e)² + I_eff_e − A_e
            dV_e = (V_e * V_e - (pi * tau_m * R_e) ** 2 + I_eff_e - A_e) / tau_m

            # τa_e dA_e/dt = −(1+β_e) A_e + β_e * I_eff_e
            dA_e = (-(1.0 + beta_e) * A_e + beta_e * I_eff_e) / tau_a_e

            # τa_e dB_e/dt = −(1+β_e) B_e − β_e Δ_e
            dB_e = (-(1.0 + beta_e) * B_e - beta_e * Delta_e) / tau_a_e

            # -------------------------------------------------------
            # Inhibitory population (4D transient FRE, same structure)
            # -------------------------------------------------------

            # Effective drive for inhibitory population
            #   (no long-range coupling for inhibitory neurons)
            #I_eff_i = eta_i + J_ie * tau_m * R_e - J_ii * tau_m * R_i
            I_eff_i = tau_m * J_ie * R_e - J_ii * tau_m * R_i

            # τm dR_i/dt = (Δ_i + B_i) / (π τm) + 2 R_i V_i
            dR_i = ((Delta_i + B_i) / (pi * tau_m) + 2.0 * R_i * V_i) / tau_m

            # τm dV_i/dt = V_i² − (π τm R_i)² + I_eff_i − A_i
            dV_i = (V_i * V_i - (pi * tau_m * R_i) ** 2 + I_eff_i - A_i) / tau_m

            # τa_i dA_i/dt = −(1+β_i) A_i + β_i * I_eff_i
            dA_i = (-(1.0 + beta_i) * A_i + beta_i * I_eff_i) / tau_a_i

            # τa_i dB_i/dt = −(1+β_i) B_i − β_i Δ_i
            dB_i = (-(1.0 + beta_i) * B_i - beta_i * Delta_i) / tau_a_i

            # -------------------------------------------------------
            # Pack outputs
            # -------------------------------------------------------

            d_state = np.stack((dR_e, dV_e, dA_e, dB_e,
                                dR_i, dV_i, dA_i, dB_i))

            # Observable: physical firing rates in Hz = R / τm
            R_e_Hz = R_e / tau_m
            R_i_Hz = R_i / tau_m
            observed = np.stack((R_e_Hz, R_i_Hz))

            return d_state, observed

        return Pietras2025_dfun

