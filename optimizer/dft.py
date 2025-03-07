import math
from typing import Union

import tensorflow as tf

"""
An Exchange-Correlation (XC) functional class collecting ingredients for training
neural-network-LMF based XC models in density functional theory.

This class implements:
    1. PBE exchange (with optional Hirao range-separation).
    2. B95 or B97 correlation.
"""

# Type definitions for clarity:
TfFloat = Union[float, tf.Tensor]  # typically tf.Tensor(dtype=tf.float32) or a Python float

def make_exc_density_calc(
    x_model: str,
    c_model: str,
    rs_model: bool,
    scal_opp: TfFloat,
    scal_ss: TfFloat,
    c_ss: TfFloat,
    c_opp: TfFloat,
    nlx: float
) -> tf.types.experimental.GenericFunction:
    """
    Creates the 'exc_density_calc' function (decorated with `@tf.function`) that computes
    the total XC (exchange-correlation) energy.

    Parameters
    ----------
    x_model : str
        Name/type of the exchange functional (e.g., "PBE").
    c_model : str
        Name/type of the correlation functional (e.g., "B95", "B97").
    rs_model : bool
        Whether to apply range separation in the exchange (True/False).
    scal_opp : TfFloat
        Scaling factor(s) for the opposite-spin correlation.
    scal_ss : TfFloat
        Scaling factor(s) for the same-spin correlation.
    c_ss : TfFloat
        Nonlinear parameter(s) for the same-spin part of the correlation.
    c_opp : TfFloat
        Nonlinear parameter(s) for the opposite-spin part of the correlation.
    nlx : float
        Mixing fraction in PBE (0 <= nlx <= 1).

    Returns
    -------
    tf.types.experimental.GenericFunction
        A function (decorated with `@tf.function`) that takes (Neural_, features)
        and returns a tf.Tensor of shape: (batch_size, 1).
    """

    @tf.function
    def exc_density_calc(
        Neural_: tf.Tensor,
        features: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute the exchange-correlation energy density based on input features and specified
        XC components (PBE exchange plus B95/B97 correlation).

        Parameters
        ----------
        Neural_ : tf.Tensor
            Neural network output (Local Mixing Function), shape: (batch_size, 1).
        features : tf.Tensor
            shape: (batch_size, 13), containing [ra, rb, gaa, gab, gbb, ea, eb,
            ta, tb, la, lb, easr, ebsr].

        Returns
        -------
        tf.Tensor
            The XC energy density of shape (batch_size, 1).
        """
        Neural = tf.squeeze(Neural_, axis=1)

        ra  = features[:, 0]
        rb  = features[:, 1]
        gaa = features[:, 2]
        gab = features[:, 3]  # not used directly
        gbb = features[:, 4]
        ea  = features[:, 5]
        eb  = features[:, 6]
        ta  = features[:, 7]
        tb  = features[:, 8]
        la  = features[:, 9]
        lb  = features[:, 10]
        easr = features[:, 11]
        ebsr = features[:, 12]

        # Exchange (PBE)
        if x_model == 'PBE':
            ex_dfa_a = _xpbe(ra, gaa) * nlx + _xlda(ra) * (1.0 - nlx)
            ex_dfa_b = _xpbe(rb, gbb) * nlx + _xlda(rb) * (1.0 - nlx)
            if rs_model:
                omega = 0.233
                ex_dfa_a = _range_sep_dfa_hirao(ra, ex_dfa_a, omega)
                ex_dfa_b = _range_sep_dfa_hirao(rb, ex_dfa_b, omega)
            ex_dfa = ex_dfa_a + ex_dfa_b
        else:
            raise ValueError("Only 'PBE' exchange is supported in this version.")

        # Correlation (B95/B97)
        scal_opp_tensor = tf.identity(scal_opp)
        scal_ss_tensor  = tf.identity(scal_ss)
        c_ss_tensor     = tf.identity(c_ss)
        c_opp_tensor    = tf.identity(c_opp)

        if c_model == 'B95':
            ec_dfa = _b95(
                scal_opp_tensor, scal_ss_tensor,
                c_ss_tensor, c_opp_tensor,
                ra, gaa, rb, gbb, ta, tb
            )
        elif c_model == 'B97':
            ec_dfa = _b97(
                scal_opp_tensor, scal_ss_tensor,
                c_ss_tensor, c_opp_tensor,
                ra, gaa, rb, gbb, ta, tb
            )
        else:
            raise ValueError("Only 'B95' or 'B97' correlation is supported in this version.")

        # Combine Exchange & Correlation
        if rs_model:
            e_xc = ea + eb + ((1 - Neural) * (ex_dfa - easr - ebsr) + ec_dfa)
        else:
            e_xc = ea + eb + ((1 - Neural) * (ex_dfa - ea - eb) + ec_dfa)

        return tf.expand_dims(e_xc, axis=1)

    return exc_density_calc


@tf.function
def _xlda(r: tf.Tensor) -> tf.Tensor:
    """
    Local Density Approximation (LDA) exchange for a single spin channel.

    Parameters
    ----------
    r : tf.Tensor
        Spin density (either alpha or beta), shape: (batch,) or similar.

    Returns
    -------
    tf.Tensor
        The LDA exchange energy density per point (same shape as r).
    """
    rs = tf.pow((4.18879020478639 * r), -1/3)
    return -0.5772520973386899 / rs * r

@tf.function
def _xpbe(r: tf.Tensor, g: tf.Tensor) -> tf.Tensor:
    """
    Perdew-Burke-Ernzerhof (PBE) exchange for a single spin channel.

    Parameters
    ----------
    r : tf.Tensor
        Spin density (alpha or beta).
    g : tf.Tensor
        Gradient of that spin density (|nabla rho|^2).

    Returns
    -------
    tf.Tensor
        PBE exchange energy density per point.
    """
    thirdm = -1.0 / 3.0
    fptrd  = 4.18879020478639
    c2     = 0.2605308805989240

    rs = (fptrd * r) ** thirdm
    s  = c2 * tf.sqrt(g) * rs / r
    t2 = s * s

    num = (1.608 + 0.4989350490554192 * t2)
    den = (1.608 + 0.2765715349531149 * t2)

    return -0.5772520973386899 * r * (num / den) / rs

@tf.function
def _range_sep_dfa_hirao(rho: tf.Tensor, ex_dfa: tf.Tensor, omega: float) -> tf.Tensor:
    """
    Compute short-range DFA exchange energy density from a non-separated exchange density,
    using the Hirao range-separation scheme:
    H. Iikura, T. Tsuneda, T. Yanai, and K. Hirao, J. Chem. Phys. 115, 3540 (2001).

    Parameters
    ----------
    rho : tf.Tensor
        Single-spin density.
    ex_dfa : tf.Tensor
        The exchange energy density to be modified.
    omega : float
        Range-separation parameter.

    Returns
    -------
    tf.Tensor
        The range-separated exchange energy density.
    """
    k_f  = (6.0 * math.pi**2 * rho) ** (1.0/3.0)
    ex_lda = _xlda(rho)
    k = tf.sqrt(ex_lda / ex_dfa) * k_f

    a = omega / (2.0 * k)
    c = _lrs_lda_f0(2.0 * a)

    return ex_dfa * c

def _lrs_lda_f0(x: tf.Tensor) -> tf.Tensor:
    """
    Auxiliary function for Hirao range separation.

    Parameters
    ----------
    x : tf.Tensor
        Dimensionless argument.

    Returns
    -------
    tf.Tensor
        The function value used to scale the exchange energy density.
    """
    SRPI = 1.77245385090551602729816748334114518
    L0 = +1.0
    L1 = -4.0 / 3.0 * SRPI
    L2 = +2.0
    L4 = -2.0 / 3.0
    U1 = +1.0 / 9.0
    U2 = -1.0 / 60.0
    U3 = +1.0 / 420.0
    U4 = -1.0 / 3240.0
    U5 = +1.0 / 27720.0
    U6 = -1.0 / 262080.0
    U7 = +1.0 / 2721600.0
    U8 = -1.0 / 4626720.0
    U9 = +1.0 / 2585520.0

    f0_015 = (((L4 * x) * x + L2) * x + L1) * x + L0
    x2i = x**(-2.0)
    f0_4  = ((((((((U9 * x2i + U8) * x2i + U7) * x2i + U6)
                 * x2i + U5) * x2i + U4)
                * x2i + U3) * x2i + U2) * x2i + U1) * x2i
    f0_else = (1.0 - 2.0 / 3.0 * x *
               (2.0 * SRPI * tf.math.erf(x**(-1.0)) - 3.0 * x + x**3.0
                + (2.0 * x - x**3.0) * tf.math.exp(-x**(-2.0))))

    cond_x015 = x < 0.15
    cond_4x = x > 4.0

    f0 = tf.where(cond_x015, f0_015, f0_else)
    f0 = tf.where(cond_4x, f0_4, f0)
    return f0

@tf.function
def _b95(
    scal_opp: TfFloat,
    scal_ss: TfFloat,
    c_ss: TfFloat,
    c_opp: TfFloat,
    r_a_: tf.Tensor,
    g_aa: tf.Tensor,
    r_b_: tf.Tensor,
    g_bb: tf.Tensor,
    t_a: tf.Tensor,
    t_b: tf.Tensor
) -> tf.Tensor:
    """
    B95 correlation functional (mGGA type).

    Parameters
    ----------
    scal_opp : TfFloat
        Scaling factor for opposite-spin correlation.
    scal_ss : TfFloat
        Scaling factor for same-spin correlation.
    c_ss : TfFloat
        Nonlinear parameter controlling same-spin correlation.
    c_opp : TfFloat
        Nonlinear parameter controlling opposite-spin correlation.
    r_a_ : tf.Tensor
        Spin-up density.
    g_aa : tf.Tensor
        Gradient of spin-up density.
    r_b_ : tf.Tensor
        Spin-down density.
    g_bb : tf.Tensor
        Gradient of spin-down density.
    t_a : tf.Tensor
        Kinetic energy density (spin-up).
    t_b : tf.Tensor
        Kinetic energy density (spin-down).

    Returns
    -------
    tf.Tensor
        B95 correlation energy density per point.
    """
    r_a = tf.maximum(r_a_, 1.0e-14)
    r_b = tf.maximum(r_b_, 1.0e-14)
    rho = r_a + r_b

    # Spin-polarization
    x = (3.0 / (4.0 * math.pi * rho)) ** (1.0/6.0)
    zeta = (r_a - r_b) / rho

    # LDA correlation baseline for total density
    lr_ab = _cpwlda(1.0, rho, x, zeta)

    # Spin-up only LDA correlation
    x_up = (3.0 / (4.0 * math.pi * r_a)) ** (1.0/6.0)
    lr_a_z = _cpwlda(1.0, r_a, x_up, 1.0)

    # Spin-down only LDA correlation
    x_dn = (3.0 / (4.0 * math.pi * r_b)) ** (1.0/6.0)
    lr_b_z = _cpwlda(1.0, r_b, x_dn, 1.0)

    ec_opp_lda = lr_ab - lr_a_z - lr_b_z

    ec_opp_fac = 1.0 / (
        1.0 + c_opp * (
            g_aa / (r_a ** (8.0/3.0)) + g_bb / (r_b ** (8.0/3.0))
        )
    )
    ec_opp = ec_opp_lda * ec_opp_fac

    ec_aa_d_fac = 2.0 * t_a - g_aa / (4.0 * r_a)
    da_ueg = 3.0 / 5.0 * (6.0 * math.pi**2) ** (2.0/3.0) * r_a ** (5.0/3.0)
    ec_aa_d_fac = ec_aa_d_fac / da_ueg
    ec_aa_fac = (1.0 + c_ss * g_aa / (r_a ** (8.0/3.0))) ** (-2.0)
    ec_aa = lr_a_z * ec_aa_fac * ec_aa_d_fac

    ec_bb_d_fac = 2.0 * t_b - g_bb / (4.0 * r_b)
    db_ueg = 3.0 / 5.0 * (6.0 * math.pi**2) ** (2.0/3.0) * r_b ** (5.0/3.0)
    ec_bb_d_fac = ec_bb_d_fac / db_ueg
    ec_bb_fac = (1.0 + c_ss * g_bb / (r_b ** (8.0/3.0))) ** (-2.0)
    ec_bb = lr_b_z * ec_bb_fac * ec_bb_d_fac

    return scal_ss * (ec_aa + ec_bb) + scal_opp * ec_opp

@tf.function
def _b97(
    scal_opp_: tf.Tensor,
    scal_ss_: tf.Tensor,
    c_ss_: tf.Tensor,
    c_opp_: tf.Tensor,
    r_a_: tf.Tensor,
    g_aa: tf.Tensor,
    r_b_: tf.Tensor,
    g_bb: tf.Tensor,
    t_a: tf.Tensor,
    t_b: tf.Tensor
) -> tf.Tensor:
    """
    B97 correlation functional.

    This version uses polynomial expansions in the opposite-spin and same-spin channels.

    Parameters
    ----------
    scal_opp_ : tf.Tensor
        Array of coefficients for the polynomial expansion of the opposite-spin term (shape (6,)).
    scal_ss_ : tf.Tensor
        Array of coefficients for the polynomial expansion of the same-spin term (shape (7,)).
    c_ss_ : tf.Tensor
        Parameter controlling same-spin inhomogeneity factor (shape (1,) or ()).
    c_opp_ : tf.Tensor
        Parameter controlling opposite-spin inhomogeneity factor (shape (1,) or ()).
    r_a_ : tf.Tensor
        Spin-up density.
    g_aa : tf.Tensor
        Gradient of spin-up density.
    r_b_ : tf.Tensor
        Spin-down density.
    g_bb : tf.Tensor
        Gradient of spin-down density.
    t_a : tf.Tensor
        Kinetic energy density (spin-up).
    t_b : tf.Tensor
        Kinetic energy density (spin-down).

    Returns
    -------
    tf.Tensor
        B97 correlation energy density per grid point.
    """
    r_a = tf.maximum(r_a_, 1.0e-14)
    r_b = tf.maximum(r_b_, 1.0e-14)
    rho = r_a + r_b

    x = (3.0 / (4.0 * math.pi * rho)) ** (1.0/6.0)
    zeta = (r_a - r_b) / rho

    lr_ab = _cpwlda(1.0, rho, x, zeta)

    x_up = (3.0 / (4.0 * math.pi * r_a)) ** (1.0/6.0)
    lr_a_z = _cpwlda(1.0, r_a, x_up, 1.0)

    x_dn = (3.0 / (4.0 * math.pi * r_b)) ** (1.0/6.0)
    lr_b_z = _cpwlda(1.0, r_b, x_dn, 1.0)

    ec_opp_lda = lr_ab - lr_a_z - lr_b_z

    c_opp = c_opp_[0]
    ec_opp_fac = (
        c_opp * (
            g_aa / (r_a ** (8.0/3.0)) + g_bb / (r_b ** (8.0/3.0))
        ) /
        (
            1.0 + c_opp * (
                g_aa / (r_a ** (8.0/3.0)) + g_bb / (r_b ** (8.0/3.0))
            )
        )
    )

    scal_opp = tf.unstack(scal_opp_)
    ec_opp = ec_opp_lda * (
        scal_opp[0] + scal_opp[1] * ec_opp_fac +
        scal_opp[2] * ec_opp_fac**2 +
        scal_opp[3] * ec_opp_fac**3 +
        scal_opp[4] * ec_opp_fac**4 +
        scal_opp[5] * ec_opp_fac**5
    )

    c_ss = c_ss_[0]
    da_ueg = 3.0 / 5.0 * (6.0 * math.pi**2) ** (2.0/3.0) * r_a ** (5.0/3.0)
    ec_aa_d_fac = (2.0 * t_a - g_aa / (4.0 * r_a)) / da_ueg

    ec_aa_fac = (
        (c_ss * g_aa / (r_a ** (8.0/3.0))) /
        (1.0 + c_ss * g_aa / (r_a ** (8.0/3.0)))
    )
    ec_aa_fac_mix = (
        (c_ss * g_aa / (r_a ** (8.0/3.0))) /
        (1.0 + c_ss * g_aa / (r_a ** (8.0/3.0)))**2
    )

    scal_ss = tf.unstack(scal_ss_)
    ec_aa = lr_a_z * ec_aa_d_fac * (
        scal_ss[0] + scal_ss[1] * ec_aa_fac +
        scal_ss[2] * ec_aa_fac**2 +
        scal_ss[3] * ec_aa_fac**3 +
        scal_ss[4] * ec_aa_fac**4 +
        scal_ss[5] * ec_aa_fac**5 +
        scal_ss[6] * ec_aa_fac_mix
    )

    db_ueg = 3.0 / 5.0 * (6.0 * math.pi**2) ** (2.0/3.0) * r_b ** (5.0/3.0)
    ec_bb_d_fac = (2.0 * t_b - g_bb / (4.0 * r_b)) / db_ueg

    ec_bb_fac = (
        (c_ss * g_bb / (r_b ** (8.0/3.0))) /
        (1.0 + c_ss * g_bb / (r_b ** (8.0/3.0)))
    )
    ec_bb_fac_mix = (
        (c_ss * g_bb / (r_b ** (8.0/3.0))) /
        (1.0 + c_ss * g_bb / (r_b ** (8.0/3.0)))**2
    )
    ec_bb = lr_b_z * ec_bb_d_fac * (
        scal_ss[0] + scal_ss[1] * ec_bb_fac +
        scal_ss[2] * ec_bb_fac**2 +
        scal_ss[3] * ec_bb_fac**3 +
        scal_ss[4] * ec_bb_fac**4 +
        scal_ss[5] * ec_bb_fac**5 +
        scal_ss[6] * ec_bb_fac_mix
    )

    return ec_aa + ec_bb + ec_opp

@tf.function
def _cpwlda(scal: float, rho: tf.Tensor, x: tf.Tensor, zeta: tf.Tensor) -> tf.Tensor:
    """
    PW LDA correlation functional.

    Parameters
    ----------
    scal : float
        Overall scale factor for the correlation (usually 1.0).
    rho : tf.Tensor
        Density (could be total or spin-resolved).
    x : tf.Tensor
        (3/(4 pi rho))^(1/6), used in correlation expansions.
    zeta : tf.Tensor
        Spin-polarization factor. For fully spin-polarized system zeta=1 or -1.
        For total correlation, zeta = (ra-rb)/rho.

    Returns
    -------
    tf.Tensor
        LDA correlation energy density * scal.
    """
    a1 = 0.0168869
    a2 = 0.11125
    a3 = 10.357
    a4 = 3.6231
    a5 = 0.88026
    a6 = 0.49671

    f1 = 0.01554535
    f2 = 0.20548
    f3 = 14.1189
    f4 = 6.1977
    f5 = 3.3662
    f6 = 0.62517

    p1 = 0.0310907
    p2 = 0.21370
    p3 = 7.5957
    p4 = 3.5876
    p5 = 1.6382
    p6 = 0.49294

    c1 = 0.5848223622634646

    t1 = 1.0 + zeta
    t2 = t1 ** (1.0 / 3.0)
    t4 = 1.0 - zeta
    t5 = t4 ** (1.0 / 3.0)
    ff = (9.0 / 8.0) * (t2 * t1 + t5 * t4 - 2.0) / c1

    t10 = x * x
    t23 = 1.0 / x

    t27 = tf.math.log(
        1.0 + 1.0 / a1 / (a3 + (a4 + (a5 + a6 * x) * x) * x) * t23 / 2.0
    )
    alpha = 2.0 * a1 * (1.0 + a2 * t10) * t27

    t44 = tf.math.log(
        1.0 + 1.0 / f1 / (f3 + (f4 + (f5 + f6 * x) * x) * x) * t23 / 2.0
    )
    epsf = -2.0 * f1 * (1.0 + f2 * t10) * t44

    t62 = tf.math.log(
        1.0 + 1.0 / p1 / (p3 + (p4 + (p5 + p6 * x) * x) * x) * t23 / 2.0
    )
    epsp = -2.0 * p1 * (1.0 + p2 * t10) * t62

    t65 = zeta * zeta
    t66 = t65 * t65
    eps = epsp + ff * t66 * (epsf - epsp) + alpha * ff * c1 * (-(t66) + 1.0)
    return scal * rho * eps
