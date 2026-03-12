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
# Equations taken from the Erratum (part c) and verified against the reference
# Python implementation (fig5_network_vs_meanfield.py, simulate_fre_4d).
#
# The equations are written in DIMENSIONLESS TIME t' = t / tau_m.
# All four mean-field variables (r, v, a, b) are dimensionless.
# The physical firing rate in Hz is R_Hz = r / tau_m_s.
#
# Single-population dimensionless equations (directly from simulate_fre_4d):
#
#   dr/dt' =  (Delta + b) / pi  +  2 * r * v
#   dv/dt' =  v^2  -  (pi * r)^2  +  eta_bar  +  J * r  -  a
#   da/dt' = [ beta * (eta_bar + J * r)  -  (1 + beta) * a ] / tau
#   db/dt' = [ -(1 + beta) * b  -  beta * Delta ] / tau
#
# where  tau = tau_a / tau_m  (dimensionless ratio, e.g. 10 for Fig. 5).
#
# Two-population extension (E and I):
#   - Excitatory (e) and inhibitory (i) populations each obey the 4D system.
#   - The effective drive for each population replaces eta_bar:
#
#       I_eff_e = eta_e  +  J_ee * r_e  -  J_ei * r_i  +  c_e
#       I_eff_i = eta_i  +  J_ie * r_e  -  J_ii * r_i
#
#     where c_e = g * (W^T @ r_e) is the long-range inter-region excitatory
#     input, computed by LinearCouplingModel.  It enters identically to eta
#     (i.e. as an additive drive), consistent with Eq. (6) of the paper and
#     the dimensionless treatment of the reference code.
#
#   - Only r_e drives inter-region coupling (analogous to S_e in Deco2014).
#   - Inhibitory neurons receive only local excitatory drive (no long-range).
#
# State variables (8 per ROI):
#   r_e, v_e, a_e, b_e   — excitatory population
#   r_i, v_i, a_i, b_i   — inhibitory population
#
# Observable variables (2 per ROI):
#   R_e_Hz = r_e / tau_m_s   [Hz]
#   R_i_Hz = r_i / tau_m_s   [Hz]
#
# Coupling variable:  r_e  (dimensionless excitatory firing rate)
#
# Initial conditions (from Erratum):
#   r(0) = r0 = 0.10,  v(0) = v0 = -0.20,  a(0) = 0,  b(0) = 0
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
    4D Transient Mean-Field model for adaptive QIF networks (Pietras et al. 2025),
    extended to two coupled populations (excitatory and inhibitory).

    All internal variables are in dimensionless time t' = t / tau_m, matching
    exactly the reference simulation (simulate_fre_4d in fig5_network_vs_meanfield.py).
    The library integrator must therefore use dt' = dt_s / tau_m_s as its step.

    The four dimensionless mean-field variables per population are:
        r  — dimensionless firing rate  (R_Hz = r / tau_m_s)
        v  — mean membrane potential
        a  — real part of complex adaptation order parameter (mean adaptation)
        b  — imaginary part (transient heterogeneity; b -> -beta*Delta/(1+beta) asymptotically)

    Parameters
    ----------
    tau_m_s   : membrane time constant [s]           (default: 10 ms = 0.01 s)
    tau_a_e_s : excitatory adaptation time const [s]  (default: 100 ms = 0.1 s)
    tau_a_i_s : inhibitory adaptation time const [s]  (default:  20 ms = 0.02 s)
    Delta_e   : Lorentzian half-width Delta_e (excitatory heterogeneity)
    Delta_i   : Lorentzian half-width Delta_i (inhibitory heterogeneity)
    eta_e     : mean excitatory external drive eta_e
    eta_i     : mean inhibitory external drive eta_i
    J_ee      : excitatory-to-excitatory coupling weight
    J_ei      : inhibitory-to-excitatory coupling weight (subtracted in drive)
    J_ie      : excitatory-to-inhibitory coupling weight
    J_ii      : inhibitory self-coupling weight (subtracted in drive)
    beta_e    : QSFA adaptation strength for excitatory population
    beta_i    : QSFA adaptation strength for inhibitory population

    Notes
    -----
    All J weights appear as  J * r  in the dimensionless equations, consistent
    with the reference code (simulate_fre_4d: `p.J * rr`), NOT as J * tau_m * r.
    The global coupling g (from LinearCouplingModel) scales the inter-region
    structural connectivity; it has units consistent with dimensionless r_e.
    """

    # ------------------------------------------------------------------
    # Variable name declarations
    # ------------------------------------------------------------------

    _state_var_names      = ['r_e', 'v_e', 'a_e', 'b_e',
                              'r_i', 'v_i', 'a_i', 'b_i']
    _coupling_var_names   = ['r_e']                           # only r_e couples regions
    _observable_var_names = ['R_e_Hz', 'R_i_Hz']

    # ------------------------------------------------------------------
    # Parameters — REGIONAL (one scalar per ROI, packed into self.m)
    # ------------------------------------------------------------------

    # Timescales (in seconds)
    tau_m_s = Attr(default=0.01, attributes=Model.Tag.REGIONAL,
                   doc="Membrane time constant [s] (default 10 ms)")

    tau_a_e_s = Attr(default=0.10, attributes=Model.Tag.REGIONAL,
                     doc="Excitatory adaptation time constant [s] (default 100 ms)")

    tau_a_i_s = Attr(default=0.02, attributes=Model.Tag.REGIONAL,
                     doc="Inhibitory adaptation time constant [s] (default 20 ms)")

    # Excitatory population
    Delta_e = Attr(default=1.0, attributes=Model.Tag.REGIONAL,
                   doc="Lorentzian half-width of excitatory eta distribution (Delta_e)")

    eta_e = Attr(default=-1.74, attributes=Model.Tag.REGIONAL,
                 doc="Mean excitatory external drive eta_e (Fig. 5 low state: -1.74)")

    J_ee = Attr(default=10.0, attributes=Model.Tag.REGIONAL,
                doc="Excitatory-to-excitatory recurrent coupling (dimensionless)")

    J_ei = Attr(default=5.0, attributes=Model.Tag.REGIONAL,
                doc="Inhibitory-to-excitatory coupling weight (subtracted in drive)")

    beta_e = Attr(default=1.0, attributes=Model.Tag.REGIONAL,
                  doc="QSFA adaptation strength for excitatory neurons beta_e")

    # Inhibitory population
    Delta_i = Attr(default=1.0, attributes=Model.Tag.REGIONAL,
                   doc="Lorentzian half-width of inhibitory eta distribution (Delta_i)")

    eta_i = Attr(default=-1.74, attributes=Model.Tag.REGIONAL,
                 doc="Mean inhibitory external drive eta_i")

    J_ie = Attr(default=10.0, attributes=Model.Tag.REGIONAL,
                doc="Excitatory-to-inhibitory coupling weight")

    J_ii = Attr(default=2.0, attributes=Model.Tag.REGIONAL,
                doc="Inhibitory self-coupling weight (subtracted in drive)")

    beta_i = Attr(default=0.5, attributes=Model.Tag.REGIONAL,
                  doc="QSFA adaptation strength for inhibitory neurons beta_i")

    # ------------------------------------------------------------------
    # Initial conditions (Erratum prescription: r0=0.10, v0=-0.20, a0=b0=0)
    # ------------------------------------------------------------------

    @overrides
    def initial_state(self, n_rois: int) -> np.ndarray:
        state = np.zeros((Pietras2025.n_state_vars, n_rois))
        # Excitatory — Erratum initial conditions
        state[0] = 0.10   # r_e
        state[1] = -0.20  # v_e
        state[2] = 0.0    # a_e
        state[3] = 0.0    # b_e
        # Inhibitory — same prescription for the second population
        state[4] = 0.10   # r_i
        state[5] = -0.20  # v_i
        state[6] = 0.0    # a_i
        state[7] = 0.0    # b_i
        return state

    # ------------------------------------------------------------------
    # Numba-compiled differential equations
    # ------------------------------------------------------------------

    @overrides
    def get_numba_dfun(self):
        """
        Return the Numba-JIT function implementing the 4D transient mean-field
        equations in dimensionless time t' = t / tau_m.

        Equations (verified line-by-line against simulate_fre_4d):

        For excitatory population, with tau_e = tau_a_e_s / tau_m_s:

            dr_e/dt' = (Delta_e + b_e) / pi  +  2 * r_e * v_e
            dv_e/dt' = v_e^2  -  (pi * r_e)^2  +  I_eff_e  -  a_e
            da_e/dt' = [ beta_e * I_eff_e  -  (1 + beta_e) * a_e ] / tau_e
            db_e/dt' = [ -(1 + beta_e) * b_e  -  beta_e * Delta_e ] / tau_e

            I_eff_e  =  eta_e  +  J_ee * r_e  -  J_ei * r_i  +  c_e

        Symmetrically for inhibitory (tau_i = tau_a_i_s / tau_m_s):

            dr_i/dt' = (Delta_i + b_i) / pi  +  2 * r_i * v_i
            dv_i/dt' = v_i^2  -  (pi * r_i)^2  +  I_eff_i  -  a_i
            da_i/dt' = [ beta_i * I_eff_i  -  (1 + beta_i) * a_i ] / tau_i
            db_i/dt' = [ -(1 + beta_i) * b_i  -  beta_i * Delta_i ] / tau_i

            I_eff_i  =  eta_i  +  J_ie * r_e  -  J_ii * r_i

        c_e = coupling[0] = g * (W^T @ r_e)   [long-range excitatory input]
        """
        m = self.m.copy()
        P = self.P
        pi = np.pi

        @nb.njit(nb.types.UniTuple(nb.f8[:, :], 2)(nb.f8[:, :], nb.f8[:, :]),
                 cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
        def Pietras2025_dfun(state: NDA_f8_2d, coupling: NDA_f8_2d):
            """
            Parameters
            ----------
            state    : (8, n_rois)   [r_e, v_e, a_e, b_e, r_i, v_i, a_i, b_i]
            coupling : (1, n_rois)   g * W^T @ r_e  (long-range excitatory input)

            Returns
            -------
            d_state  : (8, n_rois)   time derivatives w.r.t. dimensionless time t'
            observed : (2, n_rois)   [R_e_Hz, R_i_Hz]
            """
            # --- Unpack state ---
            r_e = state[0]
            v_e = state[1]
            a_e = state[2]
            b_e = state[3]
            r_i = state[4]
            v_i = state[5]
            a_i = state[6]
            b_i = state[7]

            # --- Unpack parameters ---
            tau_m_s   = m[P.tau_m_s]
            tau_a_e_s = m[P.tau_a_e_s]
            tau_a_i_s = m[P.tau_a_i_s]

            Delta_e = m[P.Delta_e]
            eta_e   = m[P.eta_e]
            J_ee    = m[P.J_ee]
            J_ei    = m[P.J_ei]
            beta_e  = m[P.beta_e]

            Delta_i = m[P.Delta_i]
            eta_i   = m[P.eta_i]
            J_ie    = m[P.J_ie]
            J_ii    = m[P.J_ii]
            beta_i  = m[P.beta_i]

            # Dimensionless adaptation time ratios (tau = tau_a / tau_m)
            # e.g. tau_e = 0.1 / 0.01 = 10.0 for the Fig. 5 parameters
            tau_e = tau_a_e_s / tau_m_s
            tau_i = tau_a_i_s / tau_m_s

            # --- Long-range excitatory input ---
            # c_e has same units as r_e (dimensionless), enters drive additively
            c_e = coupling[0]

            # -------------------------------------------------------
            # Effective drives — replace eta_bar in the single-population eqs.
            # Exactly mirrors simulate_fre_4d:  ebar + p.J * rr
            # -------------------------------------------------------
            I_eff_e = eta_e + J_ee * r_e - J_ei * r_i + c_e
            I_eff_i = eta_i + J_ie * r_e - J_ii * r_i

            # -------------------------------------------------------
            # Excitatory 4D mean-field
            # Direct translation of simulate_fre_4d (drr, dvv, daa, dbb)
            # -------------------------------------------------------
            dr_e = (Delta_e + b_e) / pi + 2.0 * r_e * v_e
            dv_e = v_e * v_e - (pi * r_e) ** 2.0 + I_eff_e - a_e
            da_e = (beta_e * I_eff_e - (1.0 + beta_e) * a_e) / tau_e
            db_e = (-(1.0 + beta_e) * b_e - beta_e * Delta_e) / tau_e

            # -------------------------------------------------------
            # Inhibitory 4D mean-field (no long-range coupling)
            # -------------------------------------------------------
            dr_i = (Delta_i + b_i) / pi + 2.0 * r_i * v_i
            dv_i = v_i * v_i - (pi * r_i) ** 2.0 + I_eff_i - a_i
            da_i = (beta_i * I_eff_i - (1.0 + beta_i) * a_i) / tau_i
            db_i = (-(1.0 + beta_i) * b_i - beta_i * Delta_i) / tau_i

            # -------------------------------------------------------
            # Pack outputs
            # -------------------------------------------------------
            d_state = np.stack((dr_e, dv_e, da_e, db_e,
                                dr_i, dv_i, da_i, db_i))

            # Physical firing rates in Hz = r / tau_m_s
            R_e_Hz = r_e / tau_m_s
            R_i_Hz = r_i / tau_m_s
            observed = np.stack((R_e_Hz, R_i_Hz))

            return d_state, observed

        return Pietras2025_dfun
