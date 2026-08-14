"""Shared physics and Monte Carlo code for the two v16-coordinate event-rate notebooks.

The module preserves the coordinate convention and analytic HOA DC/RF functions from
``dark_matter_trajectory_all_tests_connected_bowl_dc_rf_v16_hessian_fixed``.  It adds a
punctured axis-aligned upper copper ground structure above the ion and provides both an analytic
screening estimator and an adaptive solve_ivp staged ODE estimator.
"""
from __future__ import annotations

MODULE_VERSION = "2026-08-14.1"

from dataclasses import dataclass, asdict, replace, field
from pathlib import Path
from typing import Iterable, Literal
import json
import math
import os
import subprocess
import sys
import time
import traceback
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.special import k0, k1

import constants as c

# Validated v16 coupled-Mathieu/Floquet/House boundary topology.
# The reference matrices scale with epsilon/m_chi, so every boundary is a
# straight line m_chi = slope * epsilon in the (epsilon, m_chi) plane.
CLASS_UNSTABLE = 0
CLASS_STABLE_NON_PSEUDO = 1
CLASS_PSEUDO_VALID = 2

# High-mass House/Fourier pseudopotential boundaries, ordered from largest
# to smallest m_chi/epsilon. These reproduce the validated v16 reference map.
PSEUDO_TOP_SLOPES_KG = np.array([
    9.887945e-9,
    8.540416e-9,
    8.350090e-9,
    5.756753e-9,
    5.320079e-9,
], dtype=float)
STABILITY_50_SLOPE_KG = 7.535194e-19

# Low-mass stable island from the same validated reference map.
LOW_ISLAND_TOP_SLOPE_KG = 5.8053921631578245e-24
LOW_ISLAND_PSEUDO_SLOPE_KG = 2.17925504757798e-25
STABILITY_53_SLOPE_KG = 7.839928e-26

PI = math.pi
HBAR = 1.054571817e-34
KB = 1.380649e-23
EV = 1.602176634e-19
ME = 9.1093837015e-31
AMU = 1.66053906660e-27
NA = 6.02214076e23
A0 = 5.29177210903e-11
E_CHARGE_PHYSICAL = 1.602176634e-19

M_ION = float(c.m)
Z_ION = float(c.Z)
OMEGA_MODES = 2 * PI * np.array([719430.7131391969, 3031200.0099101723, 3002153.5607483205])


def _trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Composite trapezoidal integration compatible with NumPy 1.x and 2.x.

    NumPy 2.x removes ``np.trapz``. Newer releases provide ``np.trapezoid``;
    the explicit fallback below also keeps the module usable on older releases.
    Integration is performed along the last axis, matching the one-dimensional
    force-pulse arrays used in this module.
    """
    y_arr = np.asarray(y)
    x_arr = np.asarray(x)
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return trapezoid(y_arr, x_arr, axis=-1)
    if x_arr.ndim != 1:
        raise ValueError("The compatibility trapezoid fallback requires a 1D x array.")
    if y_arr.shape[-1] != x_arr.size:
        raise ValueError("The last dimension of y must match the length of x.")
    dx = np.diff(x_arr)
    return np.sum(0.5 * (y_arr[..., 1:] + y_arr[..., :-1]) * dx, axis=-1)


@dataclass(frozen=True)
class CopperGroundPlane:
    """Upper copper block with a rectangular through-gap, in ion-centered v16 coordinates.

    The copper solid is the constructive-solid-geometry set
    ``outer_box \\ gap_box``.  Bounds are the user-supplied mechanical coordinates.
    """

    x_min_m: float = -12063e-6
    x_max_m: float = 13413e-6
    y_min_m: float = -10759e-6
    y_max_m: float = 10759e-6
    z_min_m: float = 1376e-6
    z_max_m: float = 2977e-6
    gap_x_min_m: float = -4115e-6
    gap_x_max_m: float = 2542e-6
    gap_y_min_m: float = -1651e-6
    gap_y_max_m: float = 1651e-6
    gap_z_min_m: float = 1376e-6
    gap_z_max_m: float = 2977e-6
    double_layer_eV: float = 3.19
    density_kg_m3: float = 8960.0
    atomic_mass_u: float = 63.546
    Z: int = 29
    temperature_K: float = 300.0

    @property
    def lo(self) -> np.ndarray:
        """Outer-box lower bound (legacy alias used by generic box helpers)."""
        return np.array([self.x_min_m, self.y_min_m, self.z_min_m], dtype=float)

    @property
    def hi(self) -> np.ndarray:
        """Outer-box upper bound (legacy alias used by generic box helpers)."""
        return np.array([self.x_max_m, self.y_max_m, self.z_max_m], dtype=float)

    @property
    def gap_lo(self) -> np.ndarray:
        return np.array([self.gap_x_min_m, self.gap_y_min_m, self.gap_z_min_m], dtype=float)

    @property
    def gap_hi(self) -> np.ndarray:
        return np.array([self.gap_x_max_m, self.gap_y_max_m, self.gap_z_max_m], dtype=float)

    @property
    def enclosing_radius_m(self) -> float:
        corner = np.maximum(np.abs(self.lo), np.abs(self.hi))
        return float(np.linalg.norm(corner))

    @property
    def atom_density_m3(self) -> float:
        return self.density_kg_m3 / (self.atomic_mass_u * AMU)

    def contains_copper(self, point: np.ndarray, tol: float = 0.0) -> bool:
        p = np.asarray(point, dtype=float)
        in_outer = bool(np.all(p >= self.lo - tol) and np.all(p <= self.hi + tol))
        in_gap = bool(np.all(p >= self.gap_lo - tol) and np.all(p <= self.gap_hi + tol))
        return in_outer and not in_gap

    def signed_distance(self, point: np.ndarray) -> float:
        """Signed distance to ``outer_box \\ gap_box``; negative means copper."""
        p = np.asarray(point, dtype=float)
        return max(_axis_box_sdf(p, self.lo, self.hi), -_axis_box_sdf(p, self.gap_lo, self.gap_hi))


@dataclass(frozen=True)
class PRXPopulationConfig:
    """Optional PRX-Quantum Appendix-A source-density normalization inputs.

    ``phi1_eV`` and ``phi2_eV`` are effective positive-MCP compartment barriers
    per unit fractional charge.  They are intentionally distinct from the local
    copper double-layer parameter used by :func:`ground_plane_branches`.
    """

    state: Literal["pump_on_equilibrium", "pump_off_equilibrium", "filling_after_pump_off"] = "pump_off_equilibrium"
    n_lab_m3: float = 1e9
    room_temperature_K: float = 300.0
    wall_temperature_K: float = 300.0
    phi1_eV: float | None = None
    phi2_eV: float | None = None
    pump_area_ratio: float = 0.0
    vacuum_length_m: float = 1.0
    fill_time_s: float = 0.0
    axial_barrier_eV: float = 0.0


@dataclass(frozen=True)
class ScanConfig:
    m_min_kg: float = 1e-30
    m_max_kg: float = 1e-16
    eps_min: float = 1e-8
    eps_max: float = 1.0
    n_mass: int = 10
    n_eps: int = 10
    temperature_K: float = 300.0
    density_model: Literal["constant_local", "prx_two_wall"] = "constant_local"
    density_m3: float = 1e9
    prx_population: PRXPopulationConfig = field(default_factory=PRXPopulationConfig)
    target_mode: int = 2
    exact_phonon_number: int = 1
    min_mean_phonons_for_bmax: float = 1e-6
    b_min_m: float = 1e-10
    b_cap_m: float = 1e-3
    R_outer_m: float = 40e-3
    R_switch_factor: float = 2.0
    R_far_factor: float = 8.0
    samples_per_point: int = 128
    seed: int = 12345
    importance_area_fraction: float = 0.65
    max_ode_samples_per_point: int = 48
    adaptive_sampling: bool = False
    min_samples_per_point: int = 32
    target_mc_rel_se: float = 0.10
    adaptive_check_every: int = 16
    # Adaptive solve_ivp controls.  The integrated state mixes position,
    # velocity, and Fourier-impulse quadratures, so the absolute tolerances are
    # explicitly componentwise and carry the corresponding physical units.
    ode_method: str = "DOP853"
    ode_rtol: float = 1e-8
    ode_atol_position_m: float = 1e-10
    ode_atol_velocity_m_s: float = 1e-7
    ode_atol_quadrature_N_s: float = 1e-30
    ode_max_step_s: float = 2e-7
    # Hard wall-clock watchdog for one solve_ivp segment. This is independent
    # of max_stage_time_s, which is simulated physical time. Set <=0 to disable.
    ode_segment_walltime_s: float = 30.0
    max_stage_time_s: float = 0.2
    invalid_z_margin_m: float = 1e-6
    max_ground_interactions: int = 3
    delayed_branch_samples: int = 1
    # Full staged ground handling. ``enumerate`` keeps the historical weighted
    # branch tree. ``stochastic`` samples the physical branch distribution and
    # keeps a bounded number of path replicas, preventing combinatorial branch
    # explosion at strongly interacting points while remaining unbiased.
    ground_branch_mode: Literal["enumerate", "stochastic"] = "enumerate"
    ground_path_replicas: int = 1


def lower_hoa_quantum_geometry() -> dict:
    """Return the lower HOA-2 geometry used by the analytic DC/RF field model."""
    return {
        "ion_height_m": float(c.ion_height),
        "M3_z_relative_to_M4_m": float(c.z0),
        "central_slot_width_m": float(c.wcs),
        "control_pitch_m": float(c.L1),
        "inner_control_width_m": float(c.w1),
        "rf_gap_m": float(c.gap_rf),
        "rf_width_m": float(c.wrf_new),
        "control_rect_lo_m": [tuple(map(float, xy)) for xy in c.xy1k],
        "control_rect_hi_m": [tuple(map(float, xy)) for xy in c.xy2k],
        "rf_rectangles_m": [
            ((float(c.x11), float(c.y11)), (float(c.x21), float(c.y21))),
            ((float(c.x12), float(c.y12)), (float(c.x22), float(c.y22))),
        ],
        "note": "Field geometry only; lower material is terminated at the invalid-z loss boundary.",
    }


def scanconfig_audit() -> pd.DataFrame:
    """Machine-readable audit table matching the documentation."""
    cfg = ScanConfig()
    descriptions = {
        "m_min_kg": "lower logarithmic DM-mass grid edge",
        "m_max_kg": "upper logarithmic DM-mass grid edge",
        "eps_min": "lower logarithmic charge-fraction grid edge",
        "eps_max": "upper logarithmic charge-fraction grid edge",
        "n_mass": "number of logarithmic mass grid points",
        "n_eps": "number of logarithmic charge-fraction grid points",
        "temperature_K": "incoming Maxwell population temperature",
        "density_model": "constant_local benchmark or PRX two-wall source density",
        "density_m3": "source density used only for density_model=constant_local",
        "prx_population": "PRX Appendix-A population settings used when selected",
        "target_mode": "selected ion secular mode index",
        "exact_phonon_number": "M in Gamma_M",
        "min_mean_phonons_for_bmax": "straight-line heuristic used only to propose nominal b_max",
        "b_min_m": "lower support scale of the logarithmic b proposal",
        "b_cap_m": "nominal b_max cap; separately tested by annular convergence audit",
        "R_outer_m": "outer/source trajectory radius",
        "R_switch_factor": "multiplier defining the coupled-core handoff radius",
        "R_far_factor": "initial outer-tail handoff multiplier",
        "samples_per_point": "ordinary grid Monte Carlo sample count",
        "seed": "common-random-number seed",
        "importance_area_fraction": "area-proposal mixture fraction p_A for b sampling",
        "max_ode_samples_per_point": "ordinary staged-grid trajectory-sample cap; not an ODE-step cap",
        "adaptive_sampling": "enable ordinary-grid adaptive Monte Carlo stopping",
        "min_samples_per_point": "minimum samples before ordinary adaptive stopping",
        "target_mc_rel_se": "ordinary-grid relative Monte Carlo standard-error target",
        "adaptive_check_every": "ordinary-grid adaptive check cadence in samples",
        "ode_method": "solve_ivp method",
        "ode_rtol": "dimensionless solve_ivp relative local-error tolerance",
        "ode_atol_position_m": "componentwise absolute tolerance for positions [m]",
        "ode_atol_velocity_m_s": "componentwise absolute tolerance for velocities [m/s]",
        "ode_atol_quadrature_N_s": "componentwise absolute tolerance for Fourier impulse quadratures [N s]",
        "ode_max_step_s": "maximum adaptive solve_ivp internal step [s]; not a fixed dt",
        "max_stage_time_s": "maximum integration time span allowed for one stage [s]",
        "invalid_z_margin_m": "margin above the lower field-model boundary used as a loss surface",
        "max_ground_interactions": "maximum recursive upper-copper interactions per trajectory branch",
        "delayed_branch_samples": "effusive replicas for each delayed thermal re-emission branch",
    }
    raw = asdict(cfg)
    return pd.DataFrame([
        {"field": name, "default": raw[name], "meaning": descriptions.get(name, "")}
        for name in raw
    ])


# -----------------------------------------------------------------------------
# Exact v16 coordinate convention
# -----------------------------------------------------------------------------
def direction_basis(theta: float, alpha: float, psi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (u, b_hat, e_theta, e_alpha) using the exact v16 definitions.

    u = [cos(theta), sin(theta) cos(alpha), sin(theta) sin(alpha)]
    r(s) = b_vec - s u and incoming velocity points along +u.
    """
    u = np.array([
        math.cos(theta),
        math.sin(theta) * math.cos(alpha),
        math.sin(theta) * math.sin(alpha),
    ], dtype=float)
    e_theta = np.array([
        -math.sin(theta),
        math.cos(theta) * math.cos(alpha),
        math.cos(theta) * math.sin(alpha),
    ], dtype=float)
    e_alpha = np.array([0.0, -math.sin(alpha), math.cos(alpha)], dtype=float)
    b_hat = math.cos(psi) * e_theta + math.sin(psi) * e_alpha
    u /= np.linalg.norm(u)
    b_hat /= max(np.linalg.norm(b_hat), 1e-300)
    return u, b_hat, e_theta, e_alpha


def initial_state_from_v16(b_m: float, R_m: float, psi: float, alpha: float, theta: float, speed_m_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Initial incoming state on radius R with the user's v16 convention."""
    if not (0 <= b_m < R_m):
        raise ValueError("Require 0 <= b < R")
    u, b_hat, _, _ = direction_basis(theta, alpha, psi)
    s = math.sqrt(max(R_m * R_m - b_m * b_m, 0.0))
    return b_m * b_hat - s * u, speed_m_s * u


# -----------------------------------------------------------------------------
# Full analytic HOA trap potential and forces copied from the v16 source
# -----------------------------------------------------------------------------
def potential_term(x, y, z, xik, yik, z0):
    num = (xik - x) * (yik - y)
    den = (z - z0) * np.sqrt((z - z0) ** 2 + (xik - x) ** 2 + (yik - y) ** 2)
    return num / den


def dc_potential_single_electrode(x, y, z, x1k, y1k, x2k, y2k, z0, vk):
    term1 = np.arctan(potential_term(x, y, z, x2k, y2k, z0))
    term2 = np.arctan(potential_term(x, y, z, x1k, y2k, z0))
    term3 = np.arctan(potential_term(x, y, z, x2k, y1k, z0))
    term4 = np.arctan(potential_term(x, y, z, x1k, y1k, z0))
    return (vk / (2 * PI)) * (term1 - term2 - term3 + term4)


def dc_potential_total(x, y, z):
    x, y, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float))
    z = z + c.ion_height
    pot = np.zeros_like(x, dtype=float)
    for i in range(len(c.vk)):
        x1k, y1k = c.xy1k[i]
        x2k, y2k = c.xy2k[i]
        layer = 0.0 if i in (19, 39) else c.z0
        pot += dc_potential_single_electrode(x, y, z, x1k, y1k, x2k, y2k, layer, c.vk[i])
    return pot


def divatan(up, down):
    return up / (up * up + down * down)


def divatanup(up, down):
    return (down * down - up * up) / (up * up + down * down) ** 2


def divatandown(up, down):
    return -2 * up * down / (up * up + down * down) ** 2


def rf_potential_amplitude(x, y, z):
    """RF voltage amplitude corresponding to the two infinite-x rails used by v16."""
    x, y, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float))
    zz = z + c.ion_height
    return (c.vrf / PI) * (
        np.arctan2(c.y22 - y, zz) - np.arctan2(c.y12 - y, zz)
        + np.arctan2(c.y11 - y, zz) - np.arctan2(c.y21 - y, zz)
    )


def pseudopotential(x, y, z, m_dm, eps):
    x, y, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(z, float))
    zz = z + c.ion_height
    dy = divatan(zz, c.y12 - y) - divatan(zz, c.y21 - y) + divatan(zz, c.y11 - y) - divatan(zz, c.y22 - y)
    dz = divatan(c.y12 - y, zz) - divatan(c.y21 - y, zz) + divatan(c.y11 - y, zz) - divatan(c.y22 - y, zz)
    return ((eps ** 2 * c.e ** 2) / (4 * m_dm * c.omega ** 2)) * (c.vrf / PI) ** 2 * (dy * dy + dz * dz)


def trap_potential_energy(xyz, m_dm: float, eps: float):
    xyz = np.asarray(xyz, float)
    single = xyz.ndim == 1
    pts = xyz.reshape(1, 3) if single else xyz
    dc = eps * c.e * dc_potential_total(pts[:, 0], pts[:, 1], pts[:, 2])
    rf = pseudopotential(pts[:, 0], pts[:, 1], pts[:, 2], m_dm, eps)
    out = dc + rf
    return float(out[0]) if single else out


def anatangrad(xi, yi, xyz, v):
    xyz = np.asarray(xyz, float)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    dy, dx = y - yi, x - xi
    r = np.sqrt(dx * dx + dy * dy + z * z)
    dry2 = z * z + dy * dy
    drx2 = z * z + dx * dx
    tiny = np.finfo(float).tiny
    dry2 = np.maximum(dry2, tiny)
    drx2 = np.maximum(drx2, tiny)
    r = np.maximum(r, tiny)
    divy = z * dx / (r * dry2)
    divz = -dy * dx * (1 / dry2 + 1 / drx2) / r
    divx = z * dy / (r * drx2)
    return (v / (2 * PI)) * np.column_stack([divx, divy, divz])


def FDC_single(x1, y1, x2, y2, xyz, v, Z, q=c.e):
    return -Z * q * (anatangrad(x2, y2, xyz, v) - anatangrad(x2, y1, xyz, v) - anatangrad(x1, y2, xyz, v) + anatangrad(x1, y1, xyz, v))


# Precompute immutable rectangular-electrode arrays once. FDC is called many
# thousands of times inside solve_ivp; rebuilding these arrays in every RHS
# evaluation dominated runtime in the high-mass/high-charge trap-sensitive
# region.
_DC_LAYER = np.full(len(c.vk), c.z0, dtype=float)
_DC_LAYER[[19, 39]] = 0.0
_DC_X1 = np.asarray([p[0] for p in c.xy1k], dtype=float)[None, :]
_DC_Y1 = np.asarray([p[1] for p in c.xy1k], dtype=float)[None, :]
_DC_X2 = np.asarray([p[0] for p in c.xy2k], dtype=float)[None, :]
_DC_Y2 = np.asarray([p[1] for p in c.xy2k], dtype=float)[None, :]
_DC_VOLT = np.asarray(c.vk, dtype=float)[None, :]


def FDC(xyz, m_dm, eps):
    """Vectorized analytic DC force for all 40 rectangular electrodes."""
    pts = np.asarray(xyz, float).reshape(-1, 3)
    x = pts[:, 0][:, None]
    y = pts[:, 1][:, None]
    z = (pts[:, 2][:, None] + c.ion_height - _DC_LAYER[None, :])
    x1 = _DC_X1; y1 = _DC_Y1; x2 = _DC_X2; y2 = _DC_Y2; volt = _DC_VOLT

    def grad(xi, yi):
        dx = x - xi
        dy = y - yi
        r = np.sqrt(dx * dx + dy * dy + z * z)
        dry2 = np.maximum(z * z + dy * dy, 1e-300)
        drx2 = np.maximum(z * z + dx * dx, 1e-300)
        r = np.maximum(r, 1e-300)
        divx = z * dy / (r * drx2)
        divy = z * dx / (r * dry2)
        divz = -dy * dx * (1 / dry2 + 1 / drx2) / r
        return (volt / (2 * PI))[..., None] * np.stack([divx, divy, divz], axis=-1)

    combo = grad(x2, y2) - grad(x2, y1) - grad(x1, y2) + grad(x1, y1)
    return -eps * c.e * np.sum(combo, axis=1)


def FRF(xyz, m_dm, eps):
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    y = xyz[:, 1]
    z = xyz[:, 2] + c.ion_height
    dy = divatan(z, c.y12 - y) - divatan(z, c.y21 - y) + divatan(z, c.y11 - y) - divatan(z, c.y22 - y)
    dz = divatan(c.y12 - y, z) - divatan(c.y21 - y, z) + divatan(c.y11 - y, z) - divatan(c.y22 - y, z)
    dy_y = -2 * dy * (divatandown(z, c.y12 - y) - divatandown(z, c.y21 - y) + divatandown(z, c.y11 - y) - divatandown(z, c.y22 - y))
    dy_z = 2 * dy * (divatanup(z, c.y12 - y) - divatanup(z, c.y21 - y) + divatanup(z, c.y11 - y) - divatanup(z, c.y22 - y))
    dz_y = -2 * dz * (divatanup(c.y12 - y, z) - divatanup(c.y21 - y, z) + divatanup(c.y11 - y, z) - divatanup(c.y22 - y, z))
    dz_z = 2 * dz * (divatandown(c.y12 - y, z) - divatandown(c.y21 - y, z) + divatandown(c.y11 - y, z) - divatandown(c.y22 - y, z))
    pref = -((c.vrf * eps * c.e) / (2 * PI * math.sqrt(m_dm) * c.omega)) ** 2
    return pref * np.column_stack([np.zeros_like(y), dz_y + dy_y, dz_z + dy_z])


def trap_force(xyz, m_dm: float, eps: float):
    xyz = np.asarray(xyz, float)
    single = xyz.ndim == 1
    pts = xyz.reshape(1, 3) if single else xyz
    out = FDC(pts, m_dm, eps) + FRF(pts, m_dm, eps)
    return out[0] if single else out


def coulomb_force_on_dm(r_dm: np.ndarray, r_ion: np.ndarray, eps: float) -> np.ndarray:
    d = np.asarray(r_dm) - np.asarray(r_ion)
    rr = max(float(np.linalg.norm(d)), 1e-15)
    return c.K * Z_ION * eps * c.e ** 2 * d / rr ** 3


def coulomb_force_on_ion(r_dm: np.ndarray, r_ion: np.ndarray, eps: float) -> np.ndarray:
    return -coulomb_force_on_dm(r_dm, r_ion, eps)


# -----------------------------------------------------------------------------
# Stability and pseudopotential-validity mask
# -----------------------------------------------------------------------------
def numerical_hessian_scalar(func, point=np.zeros(3), h=0.1e-6):
    p = np.asarray(point, float)
    H = np.zeros((3, 3))
    f0 = float(func(*p))
    for i in range(3):
        ei = np.zeros(3); ei[i] = h
        H[i, i] = (float(func(*(p + ei))) - 2 * f0 + float(func(*(p - ei)))) / h ** 2
        for j in range(i + 1, 3):
            ej = np.zeros(3); ej[j] = h
            val = (float(func(*(p + ei + ej))) - float(func(*(p + ei - ej))) - float(func(*(p - ei + ej))) + float(func(*(p - ei - ej)))) / (4 * h ** 2)
            H[i, j] = H[j, i] = val
    return H


_HDC = None
_HRF = None

def field_hessians():
    global _HDC, _HRF
    if _HDC is None:
        _HDC = numerical_hessian_scalar(lambda x, y, z: dc_potential_total(x, y, z), h=0.15e-6)
        _HRF = numerical_hessian_scalar(lambda x, y, z: rf_potential_amplitude(x, y, z), h=0.15e-6)
    return _HDC.copy(), _HRF.copy()


def hessian_convergence_report(steps_m: Iterable[float] | None = None) -> pd.DataFrame:
    """Return the finite-difference Hessian eigenvalue step-size audit."""
    if steps_m is None:
        steps_m = (0.05e-6, 0.075e-6, 0.10e-6, 0.15e-6, 0.20e-6, 0.30e-6)
    rows = []
    for h in steps_m:
        h = float(h)
        Hdc = numerical_hessian_scalar(lambda x, y, z: dc_potential_total(x, y, z), h=h)
        Hrf = numerical_hessian_scalar(lambda x, y, z: rf_potential_amplitude(x, y, z), h=h)
        rows.append({
            "h_m": h,
            "h_um": h * 1e6,
            **{f"dc_eig_{i}_V_m2": float(v) for i, v in enumerate(np.linalg.eigvalsh(Hdc))},
            **{f"rf_eig_{i}_V_m2": float(v) for i, v in enumerate(np.linalg.eigvalsh(Hrf))},
        })
    return pd.DataFrame(rows)


def _reference_class_from_mass_over_charge(r):
    """Classify the validated v16 coupled-Floquet/House map.

    Parameters
    ----------
    r : array-like
        m_chi / epsilon in kilograms.

    Returns
    -------
    ndarray[int8]
        0 unstable, 1 stable but pseudopotential-invalid,
        2 stable and pseudopotential-valid.

    Notes
    -----
    This is the reference topology used for *sampling*. The local Hessian
    q_max and secular/RF quantities returned by :func:`mathieu_metrics` are
    diagnostics only and do not override this coupled-Floquet classification.
    """
    r = np.asarray(r, dtype=float)
    cls = np.full(r.shape, CLASS_UNSTABLE, dtype=np.int8)

    high_stable = r >= STABILITY_50_SLOPE_KG
    cls[high_stable] = CLASS_STABLE_NON_PSEUDO

    # The five high-mass House/Fourier boundaries alternate pseudo-valid and
    # pseudo-invalid bands. The first region above the largest boundary is valid.
    high_valid = r >= PSEUDO_TOP_SLOPES_KG[0]
    high_valid |= ((r < PSEUDO_TOP_SLOPES_KG[1]) &
                   (r >= PSEUDO_TOP_SLOPES_KG[2]))
    high_valid |= ((r < PSEUDO_TOP_SLOPES_KG[3]) &
                   (r >= PSEUDO_TOP_SLOPES_KG[4]))
    cls[high_stable & high_valid] = CLASS_PSEUDO_VALID

    # Low-mass island: valid upper band, stable/non-pseudo lower band, then
    # unstable below Stability 53.
    low_valid = ((r <= LOW_ISLAND_TOP_SLOPE_KG) &
                 (r >= LOW_ISLAND_PSEUDO_SLOPE_KG))
    low_stable_non = ((r < LOW_ISLAND_PSEUDO_SLOPE_KG) &
                      (r >= STABILITY_53_SLOPE_KG))
    cls[low_valid] = CLASS_PSEUDO_VALID
    cls[low_stable_non] = CLASS_STABLE_NON_PSEUDO
    return cls


def mathieu_metrics(m_dm: float, eps: float, cfg: ScanConfig) -> dict:
    """Return the validated v16 class plus local Hessian diagnostics."""
    if m_dm <= 0.0 or eps <= 0.0:
        return {
            "mathieu_stable": False,
            "pseudopotential_valid": False,
            "q_max": np.nan,
            "secular_over_rf": np.nan,
            "a_eigs": np.full(3, np.nan),
            "q_eigs": np.full(3, np.nan),
            "reference_class": CLASS_UNSTABLE,
        }

    r = float(m_dm / eps)
    cls = int(_reference_class_from_mass_over_charge(np.asarray([r]))[0])
    stable = cls != CLASS_UNSTABLE
    pseudo = cls == CLASS_PSEUDO_VALID

    # Keep the local lowest-order quantities as diagnostics. They are *not* the
    # acceptance criterion; the coupled-Floquet/House reference class above is.
    Hdc, Hrf = field_hessians()
    q_charge = eps * c.e
    A = 4 * q_charge * Hdc / (m_dm * c.omega ** 2)
    Q = 2 * q_charge * Hrf / (m_dm * c.omega ** 2)
    a_eigs = np.linalg.eigvalsh(A)
    q_eigs = np.linalg.eigvalsh(Q)
    qmax = float(np.max(np.abs(q_eigs)))
    secular_est = 0.5 * c.omega * np.sqrt(np.maximum(a_eigs + 0.5 * q_eigs ** 2, 0.0))
    secular_ratio = float(np.max(secular_est) / c.omega)
    return {
        "mathieu_stable": bool(stable),
        "pseudopotential_valid": bool(pseudo),
        "q_max": qmax,
        "secular_over_rf": secular_ratio,
        "a_eigs": a_eigs,
        "q_eigs": q_eigs,
        "reference_class": cls,
    }


def mathieu_validity_mesh(masses_kg, eps_values, cfg: ScanConfig):
    """Classify an epsilon-by-mass grid using the validated v16 map."""
    masses = np.asarray(masses_kg, dtype=float)
    eps = np.asarray(eps_values, dtype=float)
    if masses.ndim != 1 or eps.ndim != 1:
        raise ValueError("masses_kg and eps_values must be one-dimensional")
    if np.any(masses <= 0.0) or np.any(eps <= 0.0):
        raise ValueError("masses and charge fractions must be positive")
    r = masses[None, :] / eps[:, None]
    cls = _reference_class_from_mass_over_charge(r)
    stable = cls != CLASS_UNSTABLE
    pseudo = cls == CLASS_PSEUDO_VALID
    nan = np.full(r.shape, np.nan, dtype=float)
    return stable, pseudo, nan.copy(), nan.copy()


def mathieu_reference_boundaries(cfg: ScanConfig):
    """Return validated straight boundaries m_chi = slope * epsilon."""
    rows = []
    for i, slope in enumerate(PSEUDO_TOP_SLOPES_KG, 1):
        rows.append({"kind": "pseudopotential", "name": f"Pseudopotential {i}",
                     "slope_kg": float(slope)})
    rows.extend([
        {"kind": "stability", "name": "Stability 50",
         "slope_kg": STABILITY_50_SLOPE_KG},
        {"kind": "stability+pseudopotential", "name": "Low-island upper edge",
         "slope_kg": LOW_ISLAND_TOP_SLOPE_KG},
        {"kind": "pseudopotential", "name": "Low-island House edge",
         "slope_kg": LOW_ISLAND_PSEUDO_SLOPE_KG},
        {"kind": "stability", "name": "Stability 53",
         "slope_kg": STABILITY_53_SLOPE_KG},
    ])
    output = []
    for row in rows:
        slope = float(row["slope_kg"])
        m0, m1 = slope * cfg.eps_min, slope * cfg.eps_max
        if max(m0, m1) < cfg.m_min_kg or min(m0, m1) > cfg.m_max_kg:
            continue
        output.append(dict(row))
    return output


def dense_validity_grid(cfg: ScanConfig, n_mass: int = 400, n_eps: int = 400):
    """Return dense log grids and the validated v16 validity masks."""
    masses = np.logspace(np.log10(cfg.m_min_kg), np.log10(cfg.m_max_kg), int(n_mass))
    eps = np.logspace(np.log10(cfg.eps_min), np.log10(cfg.eps_max), int(n_eps))
    stable, pseudo, qmax, ratio = mathieu_validity_mesh(masses, eps, cfg)
    return masses, eps, stable, pseudo, qmax, ratio



# -----------------------------------------------------------------------------
# Source-density normalization
# -----------------------------------------------------------------------------
def prx_two_wall_density(m_dm: float, eps: float, pop: PRXPopulationConfig) -> dict:
    """Evaluate the PRX-Quantum Appendix-A two-wall population model.

    The result supplies a source/trap-vacuum density for the trajectory flux and
    an analytic ion-density cross-check.  The latter is not multiplied into a
    trajectory calculation that already propagates the trap fields.
    """
    if m_dm <= 0.0:
        raise ValueError("m_dm must be positive")
    if pop.phi1_eV is None or pop.phi2_eV is None:
        raise ValueError("prx_two_wall requires explicit phi1_eV and phi2_eV compartment barriers")
    if pop.room_temperature_K <= 0.0 or pop.wall_temperature_K <= 0.0:
        raise ValueError("PRX population temperatures must be positive")
    if pop.vacuum_length_m <= 0.0:
        raise ValueError("vacuum_length_m must be positive")
    eps_abs = abs(float(eps))
    phi1 = float(pop.phi1_eV)
    phi2 = float(pop.phi2_eV)
    kT_room_eV = KB * float(pop.room_temperature_K) / EV
    kT_wall_eV = KB * float(pop.wall_temperature_K) / EV
    nlab = float(pop.n_lab_m3)
    temp_factor = math.sqrt(float(pop.room_temperature_K) / float(pop.wall_temperature_K))
    exp = lambda x: math.exp(float(np.clip(x, -700.0, 700.0)))
    n_on_num = nlab * temp_factor * exp(-eps_abs * phi1 / kT_room_eV + eps_abs * phi1 / kT_wall_eV)
    n_on_den = 1.0 + 2.0 * float(pop.pump_area_ratio) * exp(eps_abs * (phi1 + phi2) / kT_wall_eV)
    n_on = n_on_num / max(n_on_den, np.finfo(float).tiny)
    n_off = nlab * temp_factor * exp(eps_abs * phi1 / kT_wall_eV - eps_abs * phi1 / kT_room_eV)
    n_B = 0.5 * nlab * temp_factor * exp(-eps_abs * phi1 / kT_room_eV - eps_abs * (phi2 - phi1) / kT_wall_eV)
    n_vac = n_on + n_B * (max(float(pop.fill_time_s), 0.0) / float(pop.vacuum_length_m)) * math.sqrt(
        KB * float(pop.wall_temperature_K) / (2.0 * PI * m_dm)
    )
    cap = nlab * temp_factor
    if pop.state == "pump_on_equilibrium":
        n_trap = min(n_on, cap)
    elif pop.state == "pump_off_equilibrium":
        n_trap = min(n_off, cap)
    elif pop.state == "filling_after_pump_off":
        n_trap = min(n_vac, n_off, cap)
    else:
        raise ValueError(f"Unknown PRX population state: {pop.state!r}")
    n_trap = max(float(n_trap), 0.0)
    n_ion = n_trap * exp(-eps_abs * max(float(pop.axial_barrier_eV), 0.0) / kT_wall_eV)
    return {
        "n_lab_m3": nlab,
        "n_on_m3": float(n_on),
        "n_off_m3": float(n_off),
        "n_B_m3": float(n_B),
        "n_vac_m3": float(n_vac),
        "n_trap_m3": float(n_trap),
        "n_ion_analytic_m3": float(n_ion),
        "state": pop.state,
    }


def density_for_point(m_dm: float, eps: float, cfg: ScanConfig) -> dict:
    """Return the source density actually used to normalize one grid point."""
    if cfg.density_model == "constant_local":
        n = float(cfg.density_m3)
        if n < 0.0 or not np.isfinite(n):
            raise ValueError("density_m3 must be finite and non-negative")
        return {"source_density_m3": n, "density_model": "constant_local", "n_ion_analytic_m3": np.nan}
    if cfg.density_model == "prx_two_wall":
        out = prx_two_wall_density(m_dm, eps, cfg.prx_population)
        return {**out, "source_density_m3": float(out["n_trap_m3"]), "density_model": "prx_two_wall"}
    raise ValueError(f"Unknown density_model: {cfg.density_model!r}")


# -----------------------------------------------------------------------------
# Copper transport baseline
# -----------------------------------------------------------------------------
def screening_length_m(eps: float, Z_target: int) -> float:
    zp = max(abs(eps), 1e-12)
    return 0.8853 * A0 / (zp ** 0.23 + Z_target ** 0.23)


def screened_transport_cross_section(m_dm: float, eps: float, speed: float, metal: CopperGroundPlane) -> float:
    M = metal.atomic_mass_u * AMU
    mu = m_dm * M / (m_dm + M)
    v = max(speed, 1e-12)
    a = screening_length_m(eps, metal.Z)
    eta = (2 * mu * v * a / HBAR) ** 2
    logterm = math.log1p(eta) - eta / (1 + eta)
    coupling = c.K * abs(eps) * metal.Z * E_CHARGE_PHYSICAL ** 2
    return max(2 * PI * coupling ** 2 * logterm / (mu ** 2 * v ** 4), 0.0)


def copper_transport(m_dm: float, eps: float, kinetic_J: float, metal: CopperGroundPlane) -> dict:
    v = math.sqrt(max(2 * kinetic_J / m_dm, 0.0))
    sigma_tr = screened_transport_cross_section(m_dm, eps, v, metal)
    nA = metal.atom_density_m3
    lam = 1 / max(nA * sigma_tr, 1e-300)
    M = metal.atomic_mass_u * AMU
    mu = m_dm * M / (m_dm + M)
    S_n = nA * (mu ** 2 * v ** 2 / M) * sigma_tr
    # Screened electronic stopping baseline. The low-speed regularization avoids
    # extending the Bethe-like logarithm outside its useful range.
    ne = metal.Z * nA
    omega_p = math.sqrt(ne * E_CHARGE_PHYSICAL ** 2 / (ME * 8.8541878128e-12))
    vreg = math.sqrt(v * v + (HBAR * omega_p / ME) ** 2)
    logL = max(math.log1p((2 * ME * vreg ** 2 / max(HBAR * omega_p, 1e-40)) ** 2), 0.0)
    S_e = 4 * PI * ne * (c.K * abs(eps) * E_CHARGE_PHYSICAL ** 2) ** 2 * logL / max(ME * vreg ** 2, 1e-300)
    return {"speed_m_s": v, "sigma_tr_m2": sigma_tr, "lambda_tr_m": lam, "S_n_J_m": S_n, "S_e_J_m": S_e, "S_total_J_m": S_n + S_e}


# -----------------------------------------------------------------------------
# Geometry and ground-plane branch map
# -----------------------------------------------------------------------------
def _axis_box_sdf(point: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Euclidean signed distance to an axis-aligned box (negative inside)."""
    p = np.asarray(point, dtype=float)
    lo = np.asarray(lo, dtype=float); hi = np.asarray(hi, dtype=float)
    center = 0.5 * (lo + hi); half = 0.5 * (hi - lo)
    q = np.abs(p - center) - half
    outside = float(np.linalg.norm(np.maximum(q, 0.0)))
    inside = float(min(np.max(q), 0.0))
    return outside + inside


def ray_box_intersection(origin: np.ndarray, direction: np.ndarray, box: CopperGroundPlane) -> tuple[float, float] | None:
    """Legacy outer-box slab intersection helper."""
    o = np.asarray(origin, float); d = np.asarray(direction, float)
    tmin, tmax = -np.inf, np.inf
    for j, (lo, hi) in enumerate(zip(box.lo, box.hi)):
        if abs(d[j]) < 1e-18:
            if o[j] < lo or o[j] > hi:
                return None
            continue
        t1, t2 = (lo - o[j]) / d[j], (hi - o[j]) / d[j]
        if t1 > t2: t1, t2 = t2, t1
        tmin, tmax = max(tmin, t1), min(tmax, t2)
        if tmax < tmin: return None
    if tmax < 0: return None
    return max(tmin, 0.0), tmax


def face_normal(point: np.ndarray, box: CopperGroundPlane) -> np.ndarray:
    """Legacy outer-box outward normal helper."""
    p = np.asarray(point, float)
    distances = np.r_[np.abs(p - box.lo), np.abs(p - box.hi)]
    idx = int(np.argmin(distances))
    n = np.zeros(3)
    if idx < 3: n[idx] = -1.0
    else: n[idx - 3] = 1.0
    return n


def _point_in_face(point: np.ndarray, lo: np.ndarray, hi: np.ndarray, axis: int, tol: float = 2e-10) -> bool:
    p = np.asarray(point, dtype=float)
    for j in range(3):
        if j == axis:
            continue
        if p[j] < lo[j] - tol or p[j] > hi[j] + tol:
            return False
    return True


def _plane_crossing_candidates(origin: np.ndarray, direction: np.ndarray, ground: CopperGroundPlane) -> list[float]:
    """All forward ray parameters at outer-box or gap-box face planes."""
    o = np.asarray(origin, dtype=float); d = np.asarray(direction, dtype=float)
    vals: list[float] = []
    for lo, hi in ((ground.lo, ground.hi), (ground.gap_lo, ground.gap_hi)):
        for axis in range(3):
            if abs(d[axis]) < 1e-18:
                continue
            for plane in (lo[axis], hi[axis]):
                t = (plane - o[axis]) / d[axis]
                if t <= 1e-12:
                    continue
                p = o + t * d
                if _point_in_face(p, lo, hi, axis):
                    vals.append(float(t))
    if not vals:
        return []
    vals.sort()
    unique = [vals[0]]
    for t in vals[1:]:
        if abs(t - unique[-1]) > max(1e-12, 1e-10 * max(abs(t), abs(unique[-1]), 1.0)):
            unique.append(t)
    return unique


def _ground_crossings(origin: np.ndarray, direction: np.ndarray, ground: CopperGroundPlane) -> list[dict]:
    """Return true vacuum<->copper transitions of the block-minus-gap CSG solid."""
    o = np.asarray(origin, dtype=float); d = np.asarray(direction, dtype=float)
    dn = max(float(np.linalg.norm(d)), 1e-300)
    dt_probe = 2e-9 / dn  # 2 nm physical probe on either side of a candidate plane.
    rows = []
    for t in _plane_crossing_candidates(o, d, ground):
        p = o + t * d
        before = ground.contains_copper(o + max(t - dt_probe, 0.0) * d)
        after = ground.contains_copper(o + (t + dt_probe) * d)
        if before == after:
            continue
        rows.append({
            "t": float(t), "point": p, "before_copper": bool(before),
            "after_copper": bool(after), "normal": copper_surface_normal(p, ground),
        })
    return rows


def ray_ground_first_entry(origin: np.ndarray, direction: np.ndarray, ground: CopperGroundPlane) -> dict | None:
    for row in _ground_crossings(origin, direction, ground):
        if (not row["before_copper"]) and row["after_copper"]:
            return row
    return None


def ray_ground_exit_from_inside(origin: np.ndarray, direction: np.ndarray, ground: CopperGroundPlane) -> dict | None:
    for row in _ground_crossings(origin, direction, ground):
        if row["before_copper"] and (not row["after_copper"]):
            return row
    return None


def copper_surface_normal(point: np.ndarray, ground: CopperGroundPlane) -> np.ndarray:
    """Return the local normal pointing from copper into adjacent vacuum."""
    p = np.asarray(point, dtype=float)
    candidates: list[tuple[float, np.ndarray]] = []
    # Outer faces: conventional outward normals.
    for axis in range(3):
        if _point_in_face(p, ground.lo, ground.hi, axis, tol=5e-9):
            nlo = np.zeros(3); nlo[axis] = -1.0
            nhi = np.zeros(3); nhi[axis] = +1.0
            candidates.append((abs(p[axis] - ground.lo[axis]), nlo))
            candidates.append((abs(p[axis] - ground.hi[axis]), nhi))
    # Gap side faces: normals point into the vacuum aperture.  Since the supplied
    # gap spans the full z thickness, its z-min/z-max planes are not material
    # interfaces and are omitted automatically when coincident with outer faces.
    for axis in range(3):
        if (abs(ground.gap_lo[axis] - ground.lo[axis]) < 1e-15 and
                abs(ground.gap_hi[axis] - ground.hi[axis]) < 1e-15):
            continue
        if _point_in_face(p, ground.gap_lo, ground.gap_hi, axis, tol=5e-9):
            nlo = np.zeros(3); nlo[axis] = +1.0
            nhi = np.zeros(3); nhi[axis] = -1.0
            candidates.append((abs(p[axis] - ground.gap_lo[axis]), nlo))
            candidates.append((abs(p[axis] - ground.gap_hi[axis]), nhi))
    if not candidates:
        raise ValueError("Point is not on a recognized copper CSG surface")
    candidates.sort(key=lambda item: item[0])
    probe = 2e-9
    for _, n in candidates:
        plus = ground.contains_copper(p + probe * n)
        minus = ground.contains_copper(p - probe * n)
        if (not plus) and minus:
            return n
        if plus and (not minus):
            return -n
    return candidates[0][1]


def _ground_sdf(r: np.ndarray, ground: CopperGroundPlane) -> float:
    return ground.signed_distance(r)


def sample_effusive_velocity(normal: np.ndarray, m_dm: float, T: float, rng: np.random.Generator) -> np.ndarray:
    n = np.asarray(normal, float); n /= np.linalg.norm(n)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, helper); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    En = -KB * T * math.log(max(rng.random(), 1e-15))
    Et = -KB * T * math.log(max(rng.random(), 1e-15))
    phi = 2 * PI * rng.random()
    return math.sqrt(2 * En / m_dm) * n + math.sqrt(2 * Et / m_dm) * (math.cos(phi) * e1 + math.sin(phi) * e2)


def _accelerate_out_of_metal(v_inside: np.ndarray, normal: np.ndarray, m_dm: float, barrier_J: float) -> np.ndarray:
    """Apply the outward double-layer potential step to a velocity at a surface."""
    v = np.asarray(v_inside, float).copy()
    n = np.asarray(normal, float)
    n /= max(np.linalg.norm(n), 1e-300)
    vn = float(np.dot(v, n))
    if vn < 0.0:
        # Numerical guard: an emission branch should point outward.
        vn = abs(vn)
        v = v - 2.0 * float(np.dot(v, n)) * n
    vn_after = math.sqrt(max(vn * vn + 2.0 * barrier_J / m_dm, 0.0))
    return v + (vn_after - vn) * n


def ground_plane_branches(
    r_hit: np.ndarray,
    v_in: np.ndarray,
    m_dm: float,
    eps: float,
    box: CopperGroundPlane,
    rng: np.random.Generator,
    delta: float = 1e-9,
    delayed_samples: int = 1,
) -> list[dict]:
    """Return weighted post-interaction branches at the upper copper prism.

    The interface energy bookkeeping is symmetric: a positive MCP that enters
    the metal loses the double-layer step from its *normal* kinetic energy; an
    MCP that exits gains the same step in the outward normal component.  Prompt
    transmission uses the inside-metal energy for stopping and transport.

    The non-prompt probability is represented by delayed diffusive re-emission.
    ``delayed_samples`` controls how many effusive velocity replicas are used per
    delayed face; their weights sum to the same physical branch probability.
    """
    n_entry = copper_surface_normal(r_hit, box)
    v_in = np.asarray(v_in, float)
    vn = float(np.dot(v_in, n_entry))
    if vn >= 0:
        return [{"kind": "grazing", "weight": 1.0, "r": r_hit + delta * n_entry,
                 "v": v_in.copy(), "delay_s": 0.0}]

    En = 0.5 * m_dm * vn * vn
    barrier = abs(eps) * box.double_layer_eV * EV
    if En < barrier:
        v = v_in - 2 * vn * n_entry
        return [{"kind": "reflection", "weight": 1.0, "r": r_hit + delta * n_entry,
                 "v": v, "delay_s": 0.0}]

    # Entering the higher-potential metal reduces only the normal kinetic energy.
    vn_metal = -math.sqrt(max(vn * vn - 2.0 * barrier / m_dm, 0.0))
    v_metal = v_in + (vn_metal - vn) * n_entry
    speed_metal = float(np.linalg.norm(v_metal))
    if speed_metal <= 0.0:
        # Exactly at threshold: no forward kinetic energy remains.
        v = v_in - 2 * vn * n_entry
        return [{"kind": "threshold_reflection", "weight": 1.0,
                 "r": r_hit + delta * n_entry, "v": v, "delay_s": 0.0}]

    d = v_metal / speed_metal
    r_inside = r_hit - delta * n_entry
    hit = ray_ground_exit_from_inside(r_inside, d, box)
    if hit is None:
        return [{"kind": "unresolved_metal", "weight": 1.0, "r": r_hit,
                 "v": v_in.copy(), "delay_s": 0.0}]
    t_exit = float(hit["t"])
    L = max(t_exit, delta)
    r_exit = np.asarray(hit["point"], dtype=float)
    n_exit = copper_surface_normal(r_exit, box)

    E_inside = 0.5 * m_dm * speed_metal * speed_metal
    tr = copper_transport(m_dm, eps, E_inside, box)
    tau = L / max(tr["lambda_tr_m"], 1e-300)
    E_after_inside = max(E_inside - tr["S_total_J_m"] * L, 0.0)
    p_prompt = math.exp(-min(tau, 700.0)) if E_after_inside > 0 else 0.0

    branches: list[dict] = []
    if p_prompt > 0:
        speed_out_inside = math.sqrt(2 * E_after_inside / m_dm)
        v_out_inside = speed_out_inside * d
        v_out = _accelerate_out_of_metal(v_out_inside, n_exit, m_dm, barrier)
        branches.append({
            "kind": "prompt_transmission",
            "weight": p_prompt,
            "r": r_exit + delta * n_exit,
            "v": v_out,
            "delay_s": 0.0,
        })

    p_th = 1.0 - p_prompt
    if p_th > 1e-15:
        x_th = min(E_inside / max(tr["S_total_J_m"], 1e-300), L)
        p_forward = min(max(x_th / L, 0.0), 1.0)
        D_est = max(tr["speed_m_s"] * tr["lambda_tr_m"] / 3.0, 1e-30)
        delay = x_th * max(L - x_th, 0.0) / max(2 * D_est, 1e-30)
        nrep = max(int(delayed_samples), 1)
        for face_label, prob, point, normal in (
            ("delayed_forward", p_forward, r_exit, n_exit),
            ("delayed_return", 1.0 - p_forward, r_hit, n_entry),
        ):
            if prob <= 0:
                continue
            for _ in range(nrep):
                # This is the thermal velocity just inside the metal surface; the
                # outward double-layer step is then applied explicitly.
                vv_inside = sample_effusive_velocity(normal, m_dm, box.temperature_K, rng)
                vv = _accelerate_out_of_metal(vv_inside, normal, m_dm, barrier)
                branches.append({
                    "kind": face_label,
                    "weight": p_th * prob / nrep,
                    "r": point + delta * normal,
                    "v": vv,
                    "delay_s": delay,
                })

    total = sum(b["weight"] for b in branches)
    if total <= 0:
        return [{"kind": "metal_stopped", "weight": 1.0, "r": r_hit,
                 "v": np.zeros(3), "delay_s": 0.0}]
    for branch in branches:
        branch["weight"] /= total
    return branches


# -----------------------------------------------------------------------------
# Sampling and event-rate measure
# -----------------------------------------------------------------------------
def make_common_samples(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Flux-weighted Maxwell: x^2 ~ Gamma(2,1), v=x sqrt(2kT/m).
    x = np.sqrt(rng.gamma(shape=2.0, scale=1.0, size=n))
    mu = rng.uniform(-1.0, 1.0, n)  # cos(theta), where theta is from +x
    theta = np.arccos(mu)
    alpha = rng.uniform(0.0, 2 * PI, n)
    psi = rng.uniform(0.0, 2 * PI, n)
    return {
        "x_speed": x,
        "theta": theta,
        "alpha": alpha,
        "psi": psi,
        "mix": rng.random(n),
        "u_b": rng.random(n),
        "ground_seed": rng.integers(0, 2**32 - 1, n, dtype=np.uint32),
    }


def mean_speed_maxwell(m_dm: float, T: float) -> float:
    return math.sqrt(8 * KB * T / (PI * m_dm))


def poisson_exact_probability(mean_n: float, M: int) -> float:
    """Poisson/coherent-state probability P(N=M | mean_n), evaluated stably."""
    M = int(M)
    if M < 0:
        raise ValueError("exact_phonon_number must be non-negative")
    n = max(float(mean_n), 0.0)
    if n == 0.0:
        return 1.0 if M == 0 else 0.0
    logp = -n + M * math.log(n) - math.lgamma(M + 1.0)
    if logp < -745.0:
        return 0.0
    return float(math.exp(logp))


def bmax_rutherford(m_dm: float, eps: float, speed: float, cfg: ScanConfig) -> float:
    omega = float(OMEGA_MODES[cfg.target_mode])
    coupling = c.K * Z_ION * abs(eps) * c.e ** 2
    b = math.sqrt(max(2 * coupling ** 2 / (M_ION * HBAR * omega * max(speed * speed, 1e-300) * cfg.min_mean_phonons_for_bmax), 0.0))
    return min(max(b, cfg.b_min_m * 10), cfg.b_cap_m)


def sample_b_with_weight(bmax: float, mix_u: float, u: float, cfg: ScanConfig) -> tuple[float, float]:
    pA = cfg.importance_area_fraction
    bmin = min(cfg.b_min_m, 0.1 * bmax)
    if mix_u < pA or bmax <= 1.01 * bmin:
        b = bmax * math.sqrt(max(u, 1e-15))
    else:
        b = bmin * (bmax / bmin) ** u
    if not (0.0 < pA <= 1.0):
        raise ValueError("importance_area_fraction must satisfy 0 < pA <= 1 for full support")
    q_area = 2 * b / (bmax * bmax)
    # The logarithmic proposal has support only on [bmin, bmax].  It must be
    # identically zero below bmin; otherwise the denominator would not equal the
    # density that actually generated the sample and the estimator would be biased.
    q_log = (
        1 / (b * math.log(bmax / bmin))
        if (bmax > bmin and b >= bmin and b > 0)
        else 0.0
    )
    q = pA * q_area + (1 - pA) * q_log
    # Physical b,psi measure divided by q_b q_psi, q_psi=1/(2pi).
    area_weight = 2 * PI * b / max(q, 1e-300)
    return b, area_weight


# -----------------------------------------------------------------------------
# Analytic screening kernel
# -----------------------------------------------------------------------------
def analytic_force_integral(m_dm: float, eps: float, speed: float, b: float, theta: float, alpha: float, psi: float) -> np.ndarray:
    u, bhat, _, _ = direction_basis(theta, alpha, psi)
    coupling = c.K * Z_ION * eps * c.e ** 2
    b_eff = max(b, 1e-15)
    out = np.zeros(3, dtype=complex)
    for j, omega in enumerate(OMEGA_MODES):
        x = max(omega * b_eff / max(speed, 1e-30), 1e-12)
        Iperp = 2 * coupling * omega / max(speed * speed, 1e-300) * k1(x)
        Ipara = 2j * coupling * omega / max(speed * speed, 1e-300) * k0(x)
        vec = Iperp * bhat + Ipara * u
        out[j] = vec[j]
    return out


def phonon_yield_from_integral(I: np.ndarray) -> np.ndarray:
    return np.abs(I) ** 2 / (2 * M_ION * HBAR * OMEGA_MODES)


def straight_line_barrier(r0: np.ndarray, v0: np.ndarray, m_dm: float, eps: float, n=64) -> tuple[bool, float]:
    u = v0 / max(np.linalg.norm(v0), 1e-300)
    s_closest = max(-float(np.dot(r0, u)), 0.0)
    s = np.linspace(0.0, s_closest, n)
    pts = r0[None, :] + s[:, None] * u[None, :]
    U = trap_potential_energy(pts, m_dm, eps)
    dU = float(np.nanmax(U) - U[0])
    E = 0.5 * m_dm * float(np.dot(v0, v0))
    return E > dU, dU


def screening_single_sample(m_dm: float, eps: float, speed: float, b: float, theta: float, alpha: float, psi: float, R_far: float, cfg: ScanConfig, ground: CopperGroundPlane, seed: int) -> dict:
    required_R = ground.enclosing_radius_m + 1e-3
    if cfg.R_outer_m <= required_R:
        raise ValueError(f"R_outer_m={cfg.R_outer_m:g} m does not enclose the upper copper structure (need > {required_R:g} m)")
    R_far = max(float(R_far), required_R)
    r0, v0 = initial_state_from_v16(b, R_far, psi, alpha, theta, speed)
    valid, barrier = straight_line_barrier(r0, v0, m_dm, eps)
    if not valid:
        return {"yield_modes": np.zeros(3), "branch": "trap_blocked", "barrier_J": barrier}
    rng = np.random.default_rng(seed)
    d = v0 / speed
    hit = ray_ground_first_entry(r0, d, ground)
    candidates = [{"weight": 1.0, "r": r0, "v": v0, "kind": "bypass"}]
    if hit is not None:
        rhit = np.asarray(hit["point"], dtype=float)
        candidates = ground_plane_branches(rhit, v0, m_dm, eps, ground, rng, delayed_samples=cfg.delayed_branch_samples)
    y = np.zeros(3)
    labels = []
    for br in candidates:
        vv = np.asarray(br["v"], float)
        rr = np.asarray(br["r"], float)
        if np.linalg.norm(vv) <= 0: continue
        # Determine the line's closest approach to the ion after the branch.
        u2 = vv / np.linalg.norm(vv)
        s = max(-float(np.dot(rr, u2)), 0.0)
        bvec = rr + s * u2
        b2 = float(np.linalg.norm(bvec))
        # Reconstruct local basis angle only through vector projection. The analytic
        # kernel uses the original angles for bypass and a direct numerical pulse
        # approximation for modified branches.
        if br["kind"] == "bypass":
            I = analytic_force_integral(m_dm, eps, np.linalg.norm(vv), b2, theta, alpha, psi)
        else:
            # Numerical straight-line quadrature for reflected/transmitted rays.
            T = max(10 * b2 / max(np.linalg.norm(vv), 1e-30), 10 / min(OMEGA_MODES))
            tt = np.linspace(-T, T, 512)
            pos = bvec[None, :] + tt[:, None] * vv[None, :]
            rad = np.linalg.norm(pos, axis=1)
            F = -c.K * Z_ION * eps * c.e ** 2 * pos / np.maximum(rad[:, None] ** 3, 1e-300)
            I = np.array([_trapezoid(F[:, j] * np.exp(1j * OMEGA_MODES[j] * tt), tt) for j in range(3)])
        y += float(br["weight"]) * phonon_yield_from_integral(I)
        labels.append(br["kind"])
    return {"yield_modes": y, "branch": "+".join(labels) if labels else "lost", "barrier_J": barrier}


# -----------------------------------------------------------------------------
# Full staged ODE kernel
# -----------------------------------------------------------------------------
def classify_collision_regime(speed: float, b: float) -> str:
    x = float(np.min(OMEGA_MODES) * max(b, 1e-15) / max(speed, 1e-30))
    if x > 18.0: return "adiabatic"
    if x < 0.15: return "rutherford"
    return "trap_sensitive"


def certify_radii(m_dm: float, eps: float, speed: float, b: float, theta: float, alpha: float, psi: float, cfg: ScanConfig) -> dict:
    R_full = min(max(bmax_rutherford(m_dm, eps, speed, cfg), 2 * b, 1e-7), 0.25 * cfg.R_outer_m)
    R_switch = min(cfg.R_switch_factor * R_full, 0.5 * cfg.R_outer_m)
    # Direction-dependent outer-tail check on the nominal ray.
    u, bhat, _, _ = direction_basis(theta, alpha, psi)
    radii = np.geomspace(max(cfg.R_far_factor * R_switch, 1e-5), cfg.R_outer_m, 24)
    E = 0.5 * m_dm * speed * speed
    chosen = cfg.R_outer_m
    for R in radii:
        if R <= b: continue
        r = b * bhat - math.sqrt(R * R - b * b) * u
        U = abs(float(trap_potential_energy(r, m_dm, eps)))
        F = float(np.linalg.norm(trap_force(r, m_dm, eps)))
        work = F * R
        if U < 1e-5 * max(E, 1e-40) and work < 1e-4 * max(E, 1e-40):
            chosen = float(R); break
    return {"R_full_m": R_full, "R_switch_m": R_switch, "R_far_m": chosen, "R_far_certified": chosen < cfg.R_outer_m}


def _quadrature_rhs(t: float, r_dm: np.ndarray, r_ion: np.ndarray, eps: float) -> np.ndarray:
    F = coulomb_force_on_ion(r_dm, r_ion, eps)
    vals = np.zeros(6)
    phase = np.exp(1j * OMEGA_MODES * t)
    z = F * phase
    vals[0::2] = z.real
    vals[1::2] = z.imag
    return vals


def _ode_atol_vector(cfg: ScanConfig, state_kind: Literal["dm", "core"]) -> np.ndarray:
    """Componentwise absolute tolerances for mixed-unit solve_ivp states."""
    if state_kind == "dm":
        return np.r_[
            np.full(3, cfg.ode_atol_position_m),
            np.full(3, cfg.ode_atol_velocity_m_s),
            np.full(6, cfg.ode_atol_quadrature_N_s),
        ]
    if state_kind == "core":
        return np.r_[
            np.full(3, cfg.ode_atol_position_m),
            np.full(3, cfg.ode_atol_velocity_m_s),
            np.full(3, cfg.ode_atol_position_m),
            np.full(3, cfg.ode_atol_velocity_m_s),
            np.full(6, cfg.ode_atol_quadrature_N_s),
        ]
    raise ValueError(f"Unknown state_kind: {state_kind!r}")


def _quadrature_derivative(t: float, r_dm: np.ndarray, r_ion: np.ndarray, eps: float) -> np.ndarray:
    return _quadrature_rhs(t, r_dm, r_ion, eps)


class _SegmentWallclockTimeout(RuntimeError):
    """Raised inside an ODE RHS when one solve_ivp call exceeds its wall-clock budget."""


def _check_segment_wallclock(start_wall: float, cfg: ScanConfig, label: str) -> None:
    limit = float(getattr(cfg, "ode_segment_walltime_s", 0.0))
    if limit > 0.0 and (time.perf_counter() - start_wall) > limit:
        raise _SegmentWallclockTimeout(
            f"{label} solve_ivp exceeded {limit:g} s wall-clock limit"
        )


def _integrate_dm_segment(r0, v0, q0, t0, m_dm, eps, target_radius, inbound, cfg, ground, include_quadrature=True, escape_radius=None):
    """Adaptive solve_ivp DM-only outer stage with CSG and radial events."""
    y0 = np.r_[np.asarray(r0, float), np.asarray(v0, float), np.asarray(q0, float)]
    t0 = float(t0)
    t1 = t0 + float(cfg.max_stage_time_s)
    zfloor = -float(c.ion_height) + float(cfg.invalid_z_margin_m)
    wall_start = time.perf_counter()

    def rhs(t, y):
        _check_segment_wallclock(wall_start, cfg, "outer")
        r = y[0:3]; v = y[3:6]
        dy = np.zeros_like(y)
        dy[0:3] = v
        dy[3:6] = trap_force(r, m_dm, eps) / m_dm
        if include_quadrature:
            dy[6:12] = _quadrature_derivative(t, r, np.zeros(3), eps)
        return dy

    def radius_event(t, y):
        return float(np.linalg.norm(y[0:3]) - target_radius)
    radius_event.terminal = True
    radius_event.direction = -1 if inbound else +1

    def ground_event(t, y):
        return _ground_sdf(y[0:3], ground)
    ground_event.terminal = True
    ground_event.direction = -1

    def escape_event(t, y):
        return float(np.linalg.norm(y[0:3]) - float(escape_radius))
    escape_event.terminal = True
    escape_event.direction = +1

    def invalid_event(t, y):
        return float(y[2] - zfloor)
    invalid_event.terminal = True
    invalid_event.direction = -1

    try:
        events = (radius_event, ground_event, invalid_event) if escape_radius is None else (radius_event, ground_event, invalid_event, escape_event)
        sol = solve_ivp(
            rhs, (t0, t1), y0, method=cfg.ode_method,
            rtol=float(cfg.ode_rtol), atol=_ode_atol_vector(cfg, "dm"),
            max_step=float(cfg.ode_max_step_s),
            events=events,
        )
    except _SegmentWallclockTimeout as exc:
        return {"status": "wallclock_timeout", "t": t0, "r": y0[0:3], "v": y0[3:6], "q": y0[6:12], "nfev": 0, "message": str(exc)}
    except Exception as exc:
        return {"status": "solver_failure", "t": t0, "r": y0[0:3], "v": y0[3:6], "q": y0[6:12], "nfev": 0, "message": str(exc)}

    y = sol.y[:, -1]
    status = "timeout"
    event_order = ("radius", "ground", "invalid_z") if escape_radius is None else ("radius", "ground", "invalid_z", "escape")
    event_times = [(float(ev[0]), name) for ev, name in zip(sol.t_events, event_order) if len(ev)]
    if event_times:
        _, status = min(event_times, key=lambda pair: pair[0])
    elif sol.status < 0:
        status = "solver_failure"
    return {"status": status, "t": float(sol.t[-1]), "r": y[0:3].copy(), "v": y[3:6].copy(), "q": y[6:12].copy(), "nfev": int(sol.nfev), "message": str(sol.message)}


def _coupled_accel(rd, ri, m_dm, eps):
    Fc = coulomb_force_on_dm(rd, ri, eps)
    return (trap_force(rd, m_dm, eps) + Fc) / m_dm, -OMEGA_MODES**2 * ri - Fc / M_ION


def _integrate_coupled_core(r0, v0, q0, t0, m_dm, eps, R_switch, cfg):
    """Adaptive solve_ivp coupled ion-DM core dynamics with outward escape event."""
    rd0 = np.asarray(r0, float).copy()
    # Avoid an event root exactly at the handoff surface while changing the
    # initial point by a completely negligible relative amount.
    rad0 = float(np.linalg.norm(rd0))
    if rad0 >= R_switch and rad0 > 0.0:
        rd0 *= (R_switch * (1.0 - 1e-10)) / rad0
    y0 = np.r_[rd0, np.asarray(v0, float), np.zeros(3), np.zeros(3), np.asarray(q0, float)]
    t0 = float(t0); t1 = t0 + float(cfg.max_stage_time_s)
    zfloor = -float(c.ion_height) + float(cfg.invalid_z_margin_m)
    wall_start = time.perf_counter()

    def rhs(t, y):
        _check_segment_wallclock(wall_start, cfg, "core")
        rd=y[0:3]; vd=y[3:6]; ri=y[6:9]; vi=y[9:12]
        ad, ai = _coupled_accel(rd, ri, m_dm, eps)
        dy=np.zeros_like(y)
        dy[0:3]=vd; dy[3:6]=ad; dy[6:9]=vi; dy[9:12]=ai
        dy[12:18]=_quadrature_derivative(t, rd, ri, eps)
        return dy

    def exit_event(t, y):
        rel = y[0:3] - y[6:9]
        return float(np.linalg.norm(rel) - R_switch)
    exit_event.terminal = True
    exit_event.direction = +1

    def close_event(t, y):
        return float(np.linalg.norm(y[0:3] - y[6:9]) - 1e-12)
    close_event.terminal = True
    close_event.direction = -1

    def invalid_event(t, y):
        return float(y[2] - zfloor)
    invalid_event.terminal = True
    invalid_event.direction = -1

    try:
        sol=solve_ivp(
            rhs, (t0,t1), y0, method=cfg.ode_method,
            rtol=float(cfg.ode_rtol), atol=_ode_atol_vector(cfg,"core"),
            max_step=float(cfg.ode_max_step_s),
            events=(exit_event, close_event, invalid_event),
        )
    except _SegmentWallclockTimeout as exc:
        return {"status":"core_wallclock_timeout","t":t0,"r_dm":y0[0:3],"v_dm":y0[3:6],"r_ion":y0[6:9],"v_ion":y0[9:12],"q":y0[12:18],"nfev":0,"message":str(exc)}
    except Exception as exc:
        return {"status":"core_solver_failure","t":t0,"r_dm":y0[0:3],"v_dm":y0[3:6],"r_ion":y0[6:9],"v_ion":y0[9:12],"q":y0[12:18],"nfev":0,"message":str(exc)}

    y=sol.y[:,-1]
    status="core_timeout"
    names=("core_exit","close_collision","invalid_z")
    event_times=[(float(ev[0]),name) for ev,name in zip(sol.t_events,names) if len(ev)]
    if event_times:
        _,status=min(event_times,key=lambda pair:pair[0])
    elif sol.status < 0:
        status="core_solver_failure"
    return {"status":status,"t":float(sol.t[-1]),"r_dm":y[0:3].copy(),"v_dm":y[3:6].copy(),"r_ion":y[6:9].copy(),"v_ion":y[9:12].copy(),"q":y[12:18].copy(),"nfev":int(sol.nfev),"message":str(sol.message)}


def _sample_one_ground_branch(
    r_hit: np.ndarray, v_in: np.ndarray, m_dm: float, eps: float, ground: CopperGroundPlane,
    rng: np.random.Generator,
) -> dict | None:
    """Draw one physical ground branch, including a fresh effusive velocity."""
    candidates = ground_plane_branches(
        r_hit, v_in, m_dm, eps, ground, rng, delayed_samples=1
    )
    candidates = [b for b in candidates if b.get("weight", 0.0) > 0.0 and np.linalg.norm(b.get("v", np.zeros(3))) > 0.0]
    if not candidates:
        return None
    probs = np.asarray([float(b["weight"]) for b in candidates], dtype=float)
    probs /= probs.sum()
    return candidates[int(rng.choice(len(candidates), p=probs))]


def staged_single_sample(m_dm: float, eps: float, speed: float, b: float, theta: float, alpha: float, psi: float, cfg: ScanConfig, ground: CopperGroundPlane, seed: int) -> dict:
    regime = classify_collision_regime(speed, b)
    radii = certify_radii(m_dm, eps, speed, b, theta, alpha, psi, cfg)
    R_far, R_switch = radii["R_far_m"], radii["R_switch_m"]
    required_R = ground.enclosing_radius_m + 1e-3
    if cfg.R_outer_m <= required_R:
        raise ValueError(f"R_outer_m={cfg.R_outer_m:g} m does not enclose the upper copper structure (need > {required_R:g} m)")
    R_far = max(float(R_far), required_R)
    if R_far <= b: R_far = min(cfg.R_outer_m, 1.1 * b)
    radii["R_far_m"] = R_far

    # Safe analytic shortcuts are part of the staged policy, matching the v16
    # separation of adiabatic, trap-sensitive, and Rutherford regimes.
    if regime in ("adiabatic", "rutherford"):
        scr = screening_single_sample(m_dm, eps, speed, b, theta, alpha, psi, R_far, cfg, ground, seed)
        return {**scr, **radii, "regime": regime, "status": "analytic_shortcut", "nfev": 0}

    r0, v0 = initial_state_from_v16(b, R_far, psi, alpha, theta, speed)
    rng = np.random.default_rng(seed)
    active = [{"weight": 1.0, "r": r0, "v": v0, "q": np.zeros(6), "t": 0.0, "interactions": 0, "labels": [], "ground_replicated": False}]
    reached = []
    outer_terminal_labels = []
    nfev = 0
    while active:
        st = active.pop()
        # The original launch begins at R_far and therefore has no outward
        # escape event at t=0. After any ground interaction, however, a reflected
        # or re-emitted branch may be moving outward. Give it both legitimate
        # outcomes: reach R_switch inward, or escape back through R_far outward.
        # This prevents reflected branches from integrating for max_stage_time_s
        # while waiting for an inward crossing that will never occur.
        escape_radius = R_far if st["interactions"] > 0 else None
        seg = _integrate_dm_segment(
            st["r"], st["v"], st["q"], st["t"], m_dm, eps, R_switch, True, cfg, ground,
            escape_radius=escape_radius,
        )
        nfev += seg["nfev"]
        if seg["status"] == "radius":
            reached.append({**st, **seg}); continue
        if seg["status"] == "escape":
            q = seg["q"]
            I = q[0::2] + 1j * q[1::2]
            total_escape_y = st["weight"] * phonon_yield_from_integral(I)
            # Escaping reflected/re-emitted trajectories still contribute their
            # accumulated force history even though they never enter the core.
            # Store it until total_y is initialized below.
            st["escape_yield"] = total_escape_y
            reached.append({**st, **seg, "escaped_without_core": True})
            continue
        if seg["status"] == "ground" and st["interactions"] < cfg.max_ground_interactions:
            if str(getattr(cfg, "ground_branch_mode", "enumerate")).lower() == "stochastic":
                # At the first material hit create a small fixed number of path
                # replicas. Thereafter each replica samples exactly one physical
                # branch at each hit. The path count is therefore bounded by
                # ground_path_replicas instead of growing exponentially with
                # delayed_branch_samples**interactions.
                nrep = max(int(getattr(cfg, "ground_path_replicas", 1)), 1) if not st.get("ground_replicated", False) else 1
                for _ in range(nrep):
                    br = _sample_one_ground_branch(seg["r"], seg["v"], m_dm, eps, ground, rng)
                    if br is None:
                        continue
                    active.append({
                        "weight": st["weight"] / nrep, "r": br["r"], "v": br["v"],
                        "q": seg["q"], "t": seg["t"] + br["delay_s"],
                        "interactions": st["interactions"] + 1,
                        "labels": st["labels"] + [br["kind"]], "ground_replicated": True,
                    })
            else:
                for br in ground_plane_branches(seg["r"], seg["v"], m_dm, eps, ground, rng, delayed_samples=cfg.delayed_branch_samples):
                    if br["weight"] <= 0 or np.linalg.norm(br["v"]) <= 0: continue
                    active.append({"weight": st["weight"] * br["weight"], "r": br["r"], "v": br["v"], "q": seg["q"], "t": seg["t"] + br["delay_s"], "interactions": st["interactions"] + 1, "labels": st["labels"] + [br["kind"]], "ground_replicated": st.get("ground_replicated", False)})
        elif seg["status"] != "ground":
            # Keep conservative losses visible to the convergence audit,
            # especially wall-clock/physical-time timeouts.
            outer_terminal_labels.append(str(seg["status"]))

    total_y = np.zeros(3)
    status_labels = list(outer_terminal_labels)
    for st in reached:
        if st.get("escaped_without_core", False):
            total_y += np.asarray(st.get("escape_yield", np.zeros(3)), dtype=float)
            status_labels.append("escape")
            continue
        core = _integrate_coupled_core(st["r"], st["v"], st["q"], st["t"], m_dm, eps, R_switch, cfg)
        nfev += core["nfev"]
        if core["status"] != "core_exit":
            status_labels.append(core["status"]); continue
        out = _integrate_dm_segment(core["r_dm"], core["v_dm"], core["q"], core["t"], m_dm, eps, R_far, False, cfg, ground)
        nfev += out["nfev"]
        q = out["q"]
        I = q[0::2] + 1j * q[1::2]
        total_y += st["weight"] * phonon_yield_from_integral(I)
        status_labels.append(out["status"])
    return {"yield_modes": total_y, **radii, "regime": regime, "status": "+".join(status_labels) if status_labels else "no_reach", "branch": "staged", "barrier_J": np.nan, "nfev": nfev}


# -----------------------------------------------------------------------------
# Grid drivers and plotting helpers
# -----------------------------------------------------------------------------
def parameter_grid(cfg: ScanConfig) -> tuple[np.ndarray, np.ndarray]:
    return np.logspace(math.log10(cfg.m_min_kg), math.log10(cfg.m_max_kg), cfg.n_mass), np.logspace(math.log10(cfg.eps_min), math.log10(cfg.eps_max), cfg.n_eps)


def _format_duration(seconds: float) -> str:
    """Human-readable duration used by the grid progress reporter."""
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def run_grid(
    method: Literal["screening", "staged"],
    cfg: ScanConfig,
    ground: CopperGroundPlane | None = None,
    progress: bool = True,
    progress_every_samples: int | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_every_points: int = 1,
) -> pd.DataFrame:
    """Run the mass-charge grid with live, flushed progress reporting.

    Parameters
    ----------
    method:
        ``"screening"`` or ``"staged"``.
    progress:
        Print scan, point, and sample-level progress to the notebook output.
    progress_every_samples:
        Print an interim line every N samples within a valid parameter point.
        ``None`` chooses roughly ten updates per point. Set to 0 to suppress
        sample-level lines while retaining point-level output.
    checkpoint_path:
        Optional CSV path for partial results. The file is overwritten after
        every ``checkpoint_every_points`` completed grid points and once at the
        end, so an interrupted long run still leaves inspectable output.
    checkpoint_every_points:
        Number of completed grid points between checkpoint writes.
    """
    ground = ground or CopperGroundPlane()
    masses, eps_values = parameter_grid(cfg)
    common = make_common_samples(cfg.samples_per_point, cfg.seed)
    rows: list[dict] = []
    total = len(masses) * len(eps_values)
    count = 0
    scan_start = time.perf_counter()
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    checkpoint_every_points = max(int(checkpoint_every_points), 1)

    n_default = (
        cfg.samples_per_point
        if method == "screening"
        else min(cfg.samples_per_point, cfg.max_ode_samples_per_point)
    )
    n_min = min(max(int(cfg.min_samples_per_point), 2), n_default)
    adaptive_check_every = max(int(cfg.adaptive_check_every), 1)
    if progress_every_samples is None:
        progress_every_samples = max(n_default // 10, 1)
    else:
        progress_every_samples = max(int(progress_every_samples), 0)

    if progress:
        sampling_text = (
            f"adaptive {n_min}-{n_default} samples/valid point, "
            f"target rel.SE <= {cfg.target_mc_rel_se:.3g}"
            if cfg.adaptive_sampling
            else f"{n_default} samples/valid point"
        )
        print(
            f"[{method}] starting grid: {len(masses)} masses x "
            f"{len(eps_values)} charges = {total} points; {sampling_text}",
            flush=True,
        )
        if checkpoint is not None:
            print(f"[{method}] partial CSV checkpoint: {checkpoint}", flush=True)

    for mass_index, m_dm in enumerate(masses, start=1):
        v0 = math.sqrt(2 * KB * cfg.temperature_K / m_dm)
        speeds = common["x_speed"] * v0
        mean_v = mean_speed_maxwell(m_dm, cfg.temperature_K)

        for eps_index, eps in enumerate(eps_values, start=1):
            count += 1
            point_start = time.perf_counter()
            metrics = mathieu_metrics(m_dm, eps, cfg)
            density_meta = density_for_point(m_dm, eps, cfg)
            base = {
                "m_dm_kg": m_dm,
                "eps": eps,
                "density_model": density_meta["density_model"],
                "source_density_m3": density_meta["source_density_m3"],
                "n_ion_analytic_m3": density_meta.get("n_ion_analytic_m3", np.nan),
                **{k: v for k, v in metrics.items() if k not in ("a_eigs", "q_eigs")},
            }

            if progress:
                print(
                    f"\n[{method}] point {count}/{total} "
                    f"(mass {mass_index}/{len(masses)}, eps {eps_index}/{len(eps_values)}): "
                    f"m={m_dm:.6e} kg, eps={eps:.6e}",
                    flush=True,
                )

            if not metrics["pseudopotential_valid"]:
                rows.append({
                    **base,
                    "phonon_rate_s": np.nan,
                    "event_rate_ge1_s": np.nan,
                    "event_rate_exact_M_s": np.nan,
                    "mc_rel_se": np.nan,
                    "event_mc_rel_se": np.nan,
                    "event_exact_M_mc_rel_se": np.nan,
                    "effective_samples": 0,
                    "adaptive_stop_reason": "invalid_parameter_point",
                    "mean_n_per_crossing": np.nan,
                    "method": method,
                    "mean_nfev": 0.0,
                })
                if progress:
                    elapsed = time.perf_counter() - scan_start
                    eta = elapsed / count * (total - count) if count else np.nan
                    print(
                        f"[{method}]   skipped: outside pseudopotential-valid region | "
                        f"overall {100.0 * count / total:5.1f}% | ETA {_format_duration(eta)}",
                        flush=True,
                    )
                if checkpoint is not None and (
                    count % checkpoint_every_points == 0 or count == total
                ):
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(rows).to_csv(checkpoint, index=False)
                continue

            vals: list[float] = []
            vals_event: list[float] = []
            vals_exact_M: list[float] = []
            weights: list[float] = []
            nfevs: list[float] = []
            n_to_run = n_default

            if progress:
                print(
                    f"[{method}]   valid; running {n_to_run} importance samples",
                    flush=True,
                )

            samples_used = 0
            adaptive_stop_reason = "max_samples"
            for i in range(n_to_run):
                sample_start = time.perf_counter()
                v = float(speeds[i])
                bmax = bmax_rutherford(m_dm, eps, v, cfg)
                b, area_weight = sample_b_with_weight(
                    bmax,
                    float(common["mix"][i]),
                    float(common["u_b"][i]),
                    cfg,
                )
                theta, alpha, psi = map(
                    float,
                    (common["theta"][i], common["alpha"][i], common["psi"][i]),
                )
                R_guess = max(
                    cfg.R_far_factor * bmax,
                    ground.enclosing_radius_m + 1e-3,
                    1e-4,
                )
                R_guess = min(R_guess, cfg.R_outer_m)

                if method == "screening":
                    res = screening_single_sample(
                        m_dm,
                        eps,
                        v,
                        b,
                        theta,
                        alpha,
                        psi,
                        R_guess,
                        cfg,
                        ground,
                        int(common["ground_seed"][i]),
                    )
                    nfev = 0
                else:
                    res = staged_single_sample(
                        m_dm,
                        eps,
                        v,
                        b,
                        theta,
                        alpha,
                        psi,
                        cfg,
                        ground,
                        int(common["ground_seed"][i]),
                    )
                    nfev = int(res.get("nfev", 0))

                nfevs.append(nfev)
                nk = float(res["yield_modes"][cfg.target_mode])
                # Rate measure after flux-weighted speed and isotropic direction sampling.
                w = density_for_point(m_dm, eps, cfg)["source_density_m3"] * mean_v * area_weight
                vals.append(w * nk)
                vals_event.append(w * (1 - math.exp(-min(nk, 700.0))))
                vals_exact_M.append(w * poisson_exact_probability(nk, cfg.exact_phonon_number))
                weights.append(w)

                sample_number = i + 1
                samples_used = sample_number
                should_print_sample = (
                    progress
                    and progress_every_samples > 0
                    and (
                        sample_number == 1
                        or sample_number % progress_every_samples == 0
                        or sample_number == n_to_run
                    )
                )
                if should_print_sample:
                    partial = np.asarray(vals, dtype=float)
                    partial_rate = float(np.mean(partial))
                    partial_event = float(np.mean(vals_event))
                    partial_exact = float(np.mean(vals_exact_M))
                    point_elapsed = time.perf_counter() - point_start
                    per_sample = point_elapsed / sample_number
                    sample_eta = per_sample * (n_to_run - sample_number)
                    status = str(res.get("status", ""))
                    branch = str(res.get("branch", ""))
                    regime = str(res.get("regime", ""))
                    sample_elapsed = time.perf_counter() - sample_start
                    print(
                        f"[{method}]     sample {sample_number:>4}/{n_to_run} | "
                        f"rate~{partial_rate:.4e} s^-1 | G>=1~{partial_event:.4e} s^-1 | "
                        f"G_M~{partial_exact:.4e} s^-1 | "
                        f"nfev={nfev} | dt={_format_duration(sample_elapsed)} | "
                        f"point ETA {_format_duration(sample_eta)} | "
                        f"{regime}/{branch}/{status}",
                        flush=True,
                    )

                # Adaptive per-point Monte Carlo convergence. The check is
                # deliberately based on the same weighted rate samples used
                # in the final estimator, so it controls the reported event
                # rate uncertainty rather than an auxiliary quantity.
                if (
                    cfg.adaptive_sampling
                    and sample_number >= n_min
                    and (
                        sample_number % adaptive_check_every == 0
                        or sample_number == n_to_run
                    )
                ):
                    partial = np.asarray(vals, dtype=float)
                    partial_rate = float(np.mean(partial))
                    partial_event_arr = np.asarray(vals_event, dtype=float)
                    partial_event_rate = float(np.mean(partial_event_arr))
                    if len(partial) > 1:
                        partial_se = float(np.std(partial, ddof=1) / math.sqrt(len(partial)))
                        partial_event_se = float(
                            np.std(partial_event_arr, ddof=1) / math.sqrt(len(partial_event_arr))
                        )
                        partial_rel_se = (
                            partial_se / abs(partial_rate)
                            if partial_rate != 0.0
                            else np.inf
                        )
                        partial_event_rel_se = (
                            partial_event_se / abs(partial_event_rate)
                            if partial_event_rate != 0.0
                            else np.inf
                        )
                    else:
                        partial_rel_se = np.inf
                        partial_event_rel_se = np.inf
                    convergence_metric = max(partial_rel_se, partial_event_rel_se)
                    if np.isfinite(convergence_metric) and convergence_metric <= cfg.target_mc_rel_se:
                        adaptive_stop_reason = "target_rel_se"
                        if progress:
                            print(
                                f"[{method}]     adaptive convergence reached at "
                                f"{sample_number} samples: phonon rel.SE={partial_rel_se:.3g}, "
                                f"event rel.SE={partial_event_rel_se:.3g} "
                                f"<= {cfg.target_mc_rel_se:.3g}",
                                flush=True,
                            )
                        break

            arr = np.asarray(vals, dtype=float)
            rate = float(np.mean(arr))
            se = (
                float(np.std(arr, ddof=1) / math.sqrt(len(arr)))
                if len(arr) > 1
                else np.nan
            )
            event_arr = np.asarray(vals_event, dtype=float)
            event_rate = float(np.mean(event_arr))
            event_se = (
                float(np.std(event_arr, ddof=1) / math.sqrt(len(event_arr)))
                if len(event_arr) > 1
                else np.nan
            )
            exact_arr = np.asarray(vals_exact_M, dtype=float)
            exact_rate = float(np.mean(exact_arr))
            exact_se = (
                float(np.std(exact_arr, ddof=1) / math.sqrt(len(exact_arr)))
                if len(exact_arr) > 1
                else np.nan
            )
            mean_cross = float(np.sum(vals) / max(np.sum(weights), 1e-300))
            rel_se = se / abs(rate) if rate != 0 else np.nan
            event_rel_se = event_se / abs(event_rate) if event_rate != 0 else np.nan
            exact_rel_se = exact_se / abs(exact_rate) if exact_rate != 0 else np.nan
            mean_nfev = float(np.mean(nfevs)) if nfevs else 0.0
            rows.append({
                **base,
                "phonon_rate_s": rate,
                "event_rate_ge1_s": event_rate,
                "event_rate_exact_M_s": exact_rate,
                "mc_rel_se": rel_se,
                "event_mc_rel_se": event_rel_se,
                "event_exact_M_mc_rel_se": exact_rel_se,
                "effective_samples": samples_used,
                "adaptive_stop_reason": adaptive_stop_reason,
                "mean_n_per_crossing": mean_cross,
                "method": method,
                "mean_nfev": mean_nfev,
            })

            point_elapsed = time.perf_counter() - point_start
            elapsed = time.perf_counter() - scan_start
            eta = elapsed / count * (total - count) if count else np.nan
            if progress:
                print(
                    f"[{method}]   completed point {count}/{total} in "
                    f"{_format_duration(point_elapsed)} | rate={rate:.6e} s^-1 | "
                    f"G>=1={event_rate:.6e} s^-1 | G_M={exact_rate:.6e} s^-1 | "
                    f"phonon rel.SE={rel_se:.3g} | G>=1 rel.SE={event_rel_se:.3g} | "
                    f"G_M rel.SE={exact_rel_se:.3g} | "
                    f"samples={samples_used}/{n_to_run} ({adaptive_stop_reason}) | "
                    f"mean nfev={mean_nfev:.1f} | overall {100.0 * count / total:5.1f}% | "
                    f"ETA {_format_duration(eta)}",
                    flush=True,
                )

            if checkpoint is not None and (
                count % checkpoint_every_points == 0 or count == total
            ):
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(checkpoint, index=False)
                if progress:
                    print(
                        f"[{method}]   checkpoint saved ({len(rows)} rows)",
                        flush=True,
                    )

    result = pd.DataFrame(rows)
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(checkpoint, index=False)
    if progress:
        print(
            f"\n[{method}] finished {total} grid points in "
            f"{_format_duration(time.perf_counter() - scan_start)}",
            flush=True,
        )
    return result


def result_matrix(df: pd.DataFrame, column="phonon_rate_s"):
    masses = np.sort(df.m_dm_kg.unique())
    eps = np.sort(df.eps.unique())
    pivot = df.pivot(index="eps", columns="m_dm_kg", values=column).reindex(index=eps, columns=masses)
    return masses, eps, pivot.to_numpy()


def save_config(cfg: ScanConfig, ground: CopperGroundPlane, path: str | Path):
    Path(path).write_text(json.dumps({"scan": asdict(cfg), "ground": asdict(ground)}, indent=2))


def compare_frames(staged: pd.DataFrame, screening: pd.DataFrame) -> pd.DataFrame:
    """Merge logarithmic grids robustly and form staged/screening ratios."""
    cols = ["m_dm_kg", "eps"]
    metrics = ["phonon_rate_s", "event_rate_ge1_s", "event_rate_exact_M_s"]
    available = [m for m in metrics if m in staged.columns and m in screening.columns]
    a = staged[cols + available].copy()
    b = screening[cols + available].copy()
    for frame in (a, b):
        frame["_logm_key"] = np.round(np.log10(frame["m_dm_kg"].astype(float)), 10)
        frame["_loge_key"] = np.round(np.log10(frame["eps"].astype(float)), 10)
    a = a.rename(columns={m: f"staged_{m}" for m in available})
    b = b.rename(columns={m: f"screening_{m}" for m in available})
    b = b.drop(columns=cols)
    out = a.merge(b, on=["_logm_key", "_loge_key"], how="outer")
    out["m_dm_kg"] = 10.0 ** out["_logm_key"]
    out["eps"] = 10.0 ** out["_loge_key"]
    for metric in available:
        out[f"{metric}_ratio_staged_over_screening"] = (
            out[f"staged_{metric}"] / out[f"screening_{metric}"]
        )
    # Backward-compatible column name used by earlier notebooks.
    if "phonon_rate_s" in available:
        out["phonon_rate_ratio_staged_over_screening"] = out[
            "phonon_rate_s_ratio_staged_over_screening"
        ]
    return (
        out.drop(columns=["_logm_key", "_loge_key"])
        .sort_values(["m_dm_kg", "eps"])
        .reset_index(drop=True)
    )



# -----------------------------------------------------------------------------
# Convergence-certified staged grid
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ConvergenceConfig:
    """Controls the convergence audit for the reliable staged grid.

    The audit separates three issues:
      1) Monte Carlo sampling error of the physical rate estimator;
      2) missing impact-parameter support beyond the nominal straight-line b_max;
      3) numerical/model-resolution sensitivity of the staged propagation.

    Reliability is reported separately for phonon production, Gamma_{>=1}, and
    Gamma_M.  A point is never sampled unless the validated v16
    pseudopotential classifier marks it valid.
    """
    mc_target_rel_se: float = 0.05
    mc_min_samples: int = 128
    mc_max_samples: int = 512
    mc_check_every: int = 32
    diagnostic_samples: int = 48
    b_tail_samples: int = 64
    b_tail_factor_1: float = 2.0
    b_tail_factor_2: float = 4.0
    b_tail_fraction_tol: float = 0.05
    b_tail_decay_factor: float = 1.25
    # Adaptive impact-parameter support.  The nominal Rutherford b_max is only
    # the initial support radius.  When the explicit outer-annulus audit fails,
    # the central integral is enlarged geometrically while preserving the old
    # [0,B] Monte Carlo estimate and adding independently sampled annuli.
    adaptive_bmax_enabled: bool = True
    adaptive_bmax_max_expansions: int = 8
    adaptive_bmax_max_factor: float = 256.0
    # Before enlarging support, increase the annulus statistics when the
    # apparent failure is consistent with tail-estimator noise.  This lets us
    # distinguish a genuinely long physical tail from a 128-sample upper-bound
    # fluctuation, while reusing previous annulus summaries whenever possible.
    adaptive_bmax_tail_max_samples: int = 4096
    adaptive_bmax_tail_sample_growth: float = 2.0
    # Tail annuli use a mixture proposal: area-uniform samples preserve broad
    # coverage while log-uniform samples resolve the inner edge of each octave,
    # where a decaying long-range signal usually contributes most strongly.
    # Every draw carries the exact 2*pi*b/q(b) importance weight, so new v3
    # samples can be pooled with older area-uniform annulus summaries without
    # changing the target integral.
    tail_importance_area_fraction: float = 0.50
    # The outer launch sphere is a numerical boundary, not a material surface.
    # Tail-only samples may enlarge it when an otherwise valid annulus would be
    # clipped by b < 0.8 R_outer. The old central estimate remains reusable.
    adaptive_bmax_allow_outer_radius_expand: bool = True
    adaptive_bmax_outer_radius_max_m: float = 0.50
    # Diagnostic asymptotic analysis for points that reach the explicit support
    # ceiling. This never silently promotes a point to support-converged; it
    # quantifies whether the measured octave contributions appear to decay and
    # estimates a conservative residual-tail bound for triage / v16 follow-up.
    asymptotic_tail_enabled: bool = True
    asymptotic_tail_min_annuli: int = 3
    asymptotic_tail_fit_annuli: int = 5
    asymptotic_tail_min_r2: float = 0.75
    asymptotic_tail_min_exponent: float = 0.15
    asymptotic_tail_max_ratio: float = 0.90
    adaptive_bmax_version: int = 3

    # Cost-aware central estimator. When the cheap regime census shows that a
    # large fraction of the proposal is trap-sensitive, evaluating every one of
    # the nominal 2048 samples with the full ODE is unnecessarily expensive.
    # The multifidelity estimator uses a large cheap screening sample plus an
    # independent staged-minus-screening correction sample. It is unbiased for
    # the same staged observable because the staged and screening kernels are
    # identical by policy in the Rutherford/adiabatic regimes, so expensive ODE
    # work is required only for trap-sensitive correction draws.
    central_estimator_mode: Literal["auto", "direct", "multifidelity"] = "auto"
    multifidelity_trigger_trap_fraction: float = 0.15
    multifidelity_regime_pilot_samples: int = 512
    multifidelity_screening_samples: int = 8192
    multifidelity_correction_min_samples: int = 64
    multifidelity_correction_max_samples: int = 384
    multifidelity_correction_check_every: int = 32

    numerical_rel_tol: float = 0.05
    confidence_z: float = 2.0
    outer_radius_factor: float = 1.5
    r_far_factor: float = 2.0
    timestep_factor: float = 0.5
    tolerance_factor: float = 0.1
    stage_time_factor: float = 2.0
    r_switch_factor: float = 1.5
    delayed_samples_factor: int = 2
    extra_ground_interactions: int = 2
    max_timeout_fraction: float = 0.01
    skip_secondary_if_all_mc_fail: bool = True

    # Result-quality classification thresholds.  These are deliberately
    # distinct from the strict reliability flag.  A finite central estimate
    # can still be scientifically useful even when it does not reach the 5%
    # publication-grade certification target.
    precision_rel_se: float = 0.05
    well_converged_rel_se: float = 0.10
    estimated_rel_se: float = 0.20
    noisy_rel_se: float = 0.50


def valid_parameter_points(cfg: ScanConfig) -> pd.DataFrame:
    """Return only grid points in the validated pseudopotential region."""
    masses, eps_values = parameter_grid(cfg)
    rows = []
    for m_dm in masses:
        for eps in eps_values:
            metrics = mathieu_metrics(float(m_dm), float(eps), cfg)
            if metrics["pseudopotential_valid"]:
                rows.append({
                    "m_dm_kg": float(m_dm),
                    "eps": float(eps),
                    "mathieu_stable": bool(metrics["mathieu_stable"]),
                    "pseudopotential_valid": True,
                    "q_max": float(metrics["q_max"]),
                    "secular_over_rf": float(metrics["secular_over_rf"]),
                    "reference_class": int(metrics["reference_class"]),
                })
    return pd.DataFrame(rows)


def _metric_stats(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    mean = float(np.mean(arr)) if n else np.nan
    se = float(np.std(arr, ddof=1) / math.sqrt(n)) if n > 1 else np.nan
    if np.isfinite(mean) and mean != 0.0 and np.isfinite(se):
        rel = float(se / abs(mean))
    else:
        rel = np.nan
    return {"mean": mean, "se": se, "rel_se": rel, "n": n}


def _importance_sample_contribution(
    m_dm: float,
    eps: float,
    cfg: ScanConfig,
    ground: CopperGroundPlane,
    common: dict[str, np.ndarray],
    i: int,
    *,
    b_annulus_factors: tuple[float, float] | None = None,
    b_support_factor: float = 1.0,
    annulus_area_fraction: float | None = None,
    annulus_expand_outer_radius: bool = False,
    annulus_outer_radius_max_m: float | None = None,
) -> dict:
    """Evaluate one staged importance sample.

    ``b_support_factor`` multiplies the nominal Rutherford ``B(v)`` for an
    ordinary central-support draw.  This is used by the paired numerical audit
    after an adaptive b_max expansion so baseline and numerical variants probe
    the same enlarged physical support.

    When ``b_annulus_factors`` is supplied, the sample is drawn uniformly in
    transverse area from the annulus [f_lo B(v), f_hi B(v)], where B(v) is the
    *original* nominal Rutherford b_max for that speed.  The returned weight is
    exactly the annulus area pi(b_hi^2-b_lo^2), so independently estimated
    annuli can be added to an already-computed [0,B] checkpoint result without
    rerunning those expensive central trajectories.
    """
    vscale = math.sqrt(2 * KB * cfg.temperature_K / m_dm)
    speed = float(common["x_speed"][i]) * vscale
    mean_v = mean_speed_maxwell(m_dm, cfg.temperature_K)
    b_nominal = bmax_rutherford(m_dm, eps, speed, cfg)
    clipped = False

    if b_annulus_factors is None:
        factor = max(float(b_support_factor), 1.0)
        b_requested = factor * b_nominal
        b_limit = 0.80 * cfg.R_outer_m
        b_support = min(b_requested, b_limit)
        clipped = bool(b_requested > b_limit * (1.0 + 1e-12))
        b, area_weight = sample_b_with_weight(
            b_support,
            float(common["mix"][i]),
            float(common["u_b"][i]),
            cfg,
        )
    else:
        flo, fhi = map(float, b_annulus_factors)
        if not (0.0 <= flo < fhi):
            raise ValueError("Require 0 <= lower b factor < upper b factor")
        blo = flo * b_nominal
        bhi_requested = fhi * b_nominal
        # The source sphere is a numerical launch surface. For tail-only work,
        # enlarge it when necessary instead of declaring a geometrically valid
        # annulus missing merely because the production R_outer was chosen for
        # the nominal support. The enlargement is capped explicitly.
        local_cfg = cfg
        if annulus_expand_outer_radius and bhi_requested > 0.80 * cfg.R_outer_m:
            cap = float(annulus_outer_radius_max_m) if annulus_outer_radius_max_m is not None else float(cfg.R_outer_m)
            needed = max(float(cfg.R_outer_m), 1.35 * bhi_requested, ground.enclosing_radius_m + 1e-3)
            local_R = min(max(cap, float(cfg.R_outer_m)), needed)
            if local_R > cfg.R_outer_m * (1.0 + 1e-14):
                local_cfg = replace(cfg, R_outer_m=float(local_R))
        bhi_limit = 0.80 * local_cfg.R_outer_m
        bhi = min(bhi_requested, bhi_limit)
        clipped = bool(bhi_requested > bhi_limit * (1.0 + 1e-12))
        if bhi <= blo * (1.0 + 1e-14):
            return {
                "phonon_rate_s": 0.0,
                "event_rate_ge1_s": 0.0,
                "event_rate_exact_M_s": 0.0,
                "nfev": 0.0,
                "status": "annulus_empty",
                "regime": "",
                "branch": "",
                "b_tail_clipped": clipped,
                "tail_outer_radius_m": float(local_cfg.R_outer_m),
            }
        u = max(min(float(common["u_b"][i]), 1.0 - np.finfo(float).eps), np.finfo(float).tiny)
        # Exact mixture importance sampling over the annular area measure
        # dA = 2*pi*b db.  p=1 reproduces the legacy area-uniform sampler;
        # p<1 adds a log-uniform component that concentrates resolution near
        # the inner edge of an octave without biasing the integral.
        p_area = 1.0 if annulus_area_fraction is None else float(np.clip(annulus_area_fraction, 0.0, 1.0))
        if blo <= 0.0 or p_area >= 1.0 - 1e-15:
            b = math.sqrt(blo * blo + u * (bhi * bhi - blo * blo))
            area_weight = PI * (bhi * bhi - blo * blo)
        else:
            choose_area = float(common["mix"][i]) < p_area
            if choose_area:
                b = math.sqrt(blo * blo + u * (bhi * bhi - blo * blo))
            else:
                b = blo * math.exp(u * math.log(bhi / blo))
            q_area = 2.0 * b / max(bhi * bhi - blo * blo, np.finfo(float).tiny)
            q_log = 1.0 / max(b * math.log(bhi / blo), np.finfo(float).tiny)
            q_mix = p_area * q_area + (1.0 - p_area) * q_log
            area_weight = 2.0 * PI * b / max(q_mix, np.finfo(float).tiny)
        cfg = local_cfg

    theta = float(common["theta"][i])
    alpha = float(common["alpha"][i])
    psi = float(common["psi"][i])
    seed = int(common["ground_seed"][i])
    res = staged_single_sample(
        m_dm, eps, speed, b, theta, alpha, psi, cfg, ground, seed
    )
    nk = max(float(res["yield_modes"][cfg.target_mode]), 0.0)
    w = density_for_point(m_dm, eps, cfg)["source_density_m3"] * mean_v * area_weight
    return {
        "phonon_rate_s": w * nk,
        "event_rate_ge1_s": w * (1.0 - math.exp(-min(nk, 700.0))),
        "event_rate_exact_M_s": w * poisson_exact_probability(nk, cfg.exact_phonon_number),
        "nfev": float(res.get("nfev", 0.0)),
        "status": str(res.get("status", "")),
        "regime": str(res.get("regime", "")),
        "branch": str(res.get("branch", "")),
        "b_tail_clipped": clipped,
        "tail_outer_radius_m": float(cfg.R_outer_m),
    }


def _screening_importance_sample_contribution(
    m_dm: float,
    eps: float,
    cfg: ScanConfig,
    ground: CopperGroundPlane,
    common: dict[str, np.ndarray],
    i: int,
    *,
    b_support_factor: float = 1.0,
) -> dict:
    """Evaluate the cheap screening contribution for one ordinary support draw.

    The draw and weight are identical to :func:`_importance_sample_contribution`.
    For trap-sensitive samples this serves as a control variate; for Rutherford
    and adiabatic samples it is exactly the staged policy and therefore needs no
    ODE correction.
    """
    vscale = math.sqrt(2 * KB * cfg.temperature_K / m_dm)
    speed = float(common["x_speed"][i]) * vscale
    mean_v = mean_speed_maxwell(m_dm, cfg.temperature_K)
    b_nominal = bmax_rutherford(m_dm, eps, speed, cfg)
    factor = max(float(b_support_factor), 1.0)
    b_requested = factor * b_nominal
    b_limit = 0.80 * cfg.R_outer_m
    b_support = min(b_requested, b_limit)
    b, area_weight = sample_b_with_weight(
        b_support, float(common["mix"][i]), float(common["u_b"][i]), cfg
    )
    theta = float(common["theta"][i]); alpha = float(common["alpha"][i]); psi = float(common["psi"][i])
    seed = int(common["ground_seed"][i])
    radii = certify_radii(m_dm, eps, speed, b, theta, alpha, psi, cfg)
    R_far = max(float(radii["R_far_m"]), ground.enclosing_radius_m + 1e-3)
    res = screening_single_sample(m_dm, eps, speed, b, theta, alpha, psi, R_far, cfg, ground, seed)
    nk = max(float(res["yield_modes"][cfg.target_mode]), 0.0)
    w = density_for_point(m_dm, eps, cfg)["source_density_m3"] * mean_v * area_weight
    regime = classify_collision_regime(speed, b)
    return {
        "phonon_rate_s": w * nk,
        "event_rate_ge1_s": w * (1.0 - math.exp(-min(nk, 700.0))),
        "event_rate_exact_M_s": w * poisson_exact_probability(nk, cfg.exact_phonon_number),
        "regime": regime, "status": str(res.get("branch", "")), "nfev": 0.0,
    }


def _cheap_trap_sensitive_fraction(m_dm: float, eps: float, cfg: ScanConfig, n: int, seed: int) -> float:
    """Cheap proposal census used only to select the central estimator."""
    n = max(int(n), 1)
    common = make_common_samples(n, int(seed))
    vscale = math.sqrt(2 * KB * cfg.temperature_K / m_dm)
    hits = 0
    for i in range(n):
        speed = float(common["x_speed"][i]) * vscale
        b_nominal = bmax_rutherford(m_dm, eps, speed, cfg)
        b, _ = sample_b_with_weight(b_nominal, float(common["mix"][i]), float(common["u_b"][i]), cfg)
        hits += int(classify_collision_regime(speed, b) == "trap_sensitive")
    return float(hits / n)


def multifidelity_point_estimate(
    m_dm: float,
    eps: float,
    cfg: ScanConfig,
    ground: CopperGroundPlane,
    conv: ConvergenceConfig,
    *,
    seed: int | None = None,
    progress: bool = False,
) -> dict:
    """Unbiased screening-control-variate estimator for expensive points.

    Let S be the cheap screening contribution and F the staged contribution.
    The estimator is E[S] + E[F-S]. The second expectation is evaluated with an
    independent sample. Since F=S by construction in Rutherford/adiabatic
    regimes, the expensive staged ODE is called only for trap-sensitive draws.
    Independent base/correction streams give
        SE^2 = SE(Sbar)^2 + SE(Deltabar)^2.
    """
    metrics = mathieu_metrics(m_dm, eps, cfg)
    if not metrics["pseudopotential_valid"]:
        raise ValueError("Refusing to sample a pseudopotential-invalid point")
    seed0 = cfg.seed if seed is None else int(seed)
    n_base = max(int(conv.multifidelity_screening_samples), 2)
    n_corr_max = max(int(conv.multifidelity_correction_max_samples), 2)
    n_corr_min = min(max(int(conv.multifidelity_correction_min_samples), 2), n_corr_max)
    check_every = max(int(conv.multifidelity_correction_check_every), 1)
    base_common = make_common_samples(n_base, seed0 + 700001)
    corr_common = make_common_samples(n_corr_max, seed0 + 900001)
    metric_names = ("phonon_rate_s", "event_rate_ge1_s", "event_rate_exact_M_s")

    base_vals = {k: np.empty(n_base, dtype=float) for k in metric_names}
    for i in range(n_base):
        rec = _screening_importance_sample_contribution(m_dm, eps, cfg, ground, base_common, i)
        for k in metric_names: base_vals[k][i] = rec[k]
    base_stats = {k: _metric_stats(base_vals[k]) for k in metric_names}

    deltas = {k: [] for k in metric_names}
    statuses = []; regimes = []; nfevs = []
    ode_samples = 0; used = 0; stop_reason = "correction_max_samples"
    for i in range(n_corr_max):
        scr = _screening_importance_sample_contribution(m_dm, eps, cfg, ground, corr_common, i)
        regime = str(scr["regime"]); regimes.append(regime)
        if regime == "trap_sensitive":
            full = _importance_sample_contribution(m_dm, eps, cfg, ground, corr_common, i)
            ode_samples += 1
            statuses.append(str(full.get("status", "")))
            nfevs.append(float(full.get("nfev", 0.0)))
            for k in metric_names: deltas[k].append(float(full[k]) - float(scr[k]))
        else:
            statuses.append("analytic_shortcut")
            nfevs.append(0.0)
            for k in metric_names: deltas[k].append(0.0)
        used = i + 1

        if progress and (used == 1 or used % check_every == 0 or used == n_corr_max):
            combined = {}
            for k in metric_names:
                ds = _metric_stats(np.asarray(deltas[k], dtype=float))
                mean = base_stats[k]["mean"] + ds["mean"]
                se = math.sqrt(base_stats[k]["se"] ** 2 + ds["se"] ** 2) if np.isfinite(ds["se"]) else np.inf
                combined[k] = se / abs(mean) if mean != 0.0 else np.inf
            print(
                f"      MF correction {used}/{n_corr_max}: trap-sensitive ODE={ode_samples}; "
                f"rel.SE phonon={combined['phonon_rate_s']:.3g}, "
                f"Gamma>=1={combined['event_rate_ge1_s']:.3g}, Gamma_M={combined['event_rate_exact_M_s']:.3g}",
                flush=True,
            )
        if used >= n_corr_min and (used % check_every == 0 or used == n_corr_max):
            rels=[]
            for k in ("phonon_rate_s", "event_rate_ge1_s"):
                ds = _metric_stats(np.asarray(deltas[k], dtype=float))
                mean = base_stats[k]["mean"] + ds["mean"]
                se = math.sqrt(base_stats[k]["se"] ** 2 + ds["se"] ** 2) if np.isfinite(ds["se"]) else np.inf
                rels.append(se / abs(mean) if mean != 0.0 else np.inf)
            if all(np.isfinite(x) and x <= conv.mc_target_rel_se for x in rels):
                stop_reason = "target_rel_se"
                break

    result = {
        "m_dm_kg": float(m_dm), "eps": float(eps),
        "mathieu_stable": bool(metrics["mathieu_stable"]), "pseudopotential_valid": True,
        "q_max": float(metrics["q_max"]), "secular_over_rf": float(metrics["secular_over_rf"]),
        "reference_class": int(metrics["reference_class"]),
        "effective_samples": int(n_base + used),
        "adaptive_stop_reason": stop_reason,
        "central_estimator": "screening_control_variate",
        "multifidelity_screening_samples": int(n_base),
        "multifidelity_correction_samples": int(used),
        "multifidelity_trap_sensitive_ode_samples": int(ode_samples),
        "trap_sensitive_fraction": float(np.mean([r == "trap_sensitive" for r in regimes])) if regimes else 0.0,
        "mean_nfev": float(np.mean(nfevs)) if nfevs else 0.0,
        "timeout_fraction": float(np.mean(["timeout" in s for s in statuses])) if statuses else 0.0,
        "analytic_shortcut_fraction": float(np.mean([r != "trap_sensitive" for r in regimes])) if regimes else 1.0,
    }
    for metric, prefix in (("phonon_rate_s","phonon"),("event_rate_ge1_s","ge1"),("event_rate_exact_M_s","exact_M")):
        ds = _metric_stats(np.asarray(deltas[metric], dtype=float))
        mean = float(base_stats[metric]["mean"] + ds["mean"])
        se = float(math.sqrt(base_stats[metric]["se"] ** 2 + ds["se"] ** 2)) if np.isfinite(ds["se"]) else np.inf
        result[metric] = mean
        result[f"{prefix}_se_s"] = se; result[f"{prefix}_mc_se_s"] = se
        result[f"{prefix}_mc_rel_se"] = se / abs(mean) if mean != 0.0 else np.inf
        result[f"{prefix}_screening_base_s"] = float(base_stats[metric]["mean"])
        result[f"{prefix}_staged_correction_s"] = float(ds["mean"])
        result[f"{prefix}_correction_se_s"] = float(ds["se"])
    result["mc_rel_se"] = result["phonon_mc_rel_se"]
    result["event_mc_rel_se"] = result["ge1_mc_rel_se"]
    result["event_exact_M_mc_rel_se"] = result["exact_M_mc_rel_se"]
    return result


def central_point_estimate_auto(
    m_dm: float, eps: float, cfg: ScanConfig, ground: CopperGroundPlane, conv: ConvergenceConfig,
    *, seed: int | None = None, progress: bool = False,
) -> dict:
    seed0 = cfg.seed if seed is None else int(seed)
    mode = str(conv.central_estimator_mode).lower()
    if mode not in {"auto", "direct", "multifidelity"}:
        raise ValueError(f"Unknown central_estimator_mode={conv.central_estimator_mode!r}")
    frac = _cheap_trap_sensitive_fraction(
        m_dm, eps, cfg, conv.multifidelity_regime_pilot_samples, seed0 + 500001
    )
    use_mf = mode == "multifidelity" or (mode == "auto" and frac >= conv.multifidelity_trigger_trap_fraction)
    if progress:
        chosen = "multifidelity screening-control-variate" if use_mf else "direct staged"
        print(f"    central estimator: {chosen}; pilot trap-sensitive fraction={frac:.3f}", flush=True)
    if use_mf:
        out = multifidelity_point_estimate(m_dm, eps, cfg, ground, conv, seed=seed0, progress=progress)
    else:
        out = adaptive_point_estimate(m_dm, eps, cfg, ground, conv, seed=seed0, progress=progress)
        out["central_estimator"] = "direct_staged"
        out["trap_sensitive_fraction"] = frac
        out["multifidelity_screening_samples"] = 0
        out["multifidelity_correction_samples"] = 0
        out["multifidelity_trap_sensitive_ode_samples"] = 0
    return out


def point_sample_contributions(
    m_dm: float,
    eps: float,
    cfg: ScanConfig,
    ground: CopperGroundPlane,
    n_samples: int,
    *,
    seed: int | None = None,
    b_annulus_factors: tuple[float, float] | None = None,
    b_support_factor: float = 1.0,
    annulus_area_fraction: float | None = None,
    annulus_expand_outer_radius: bool = False,
    annulus_outer_radius_max_m: float | None = None,
    progress: bool = False,
) -> dict:
    """Return raw per-sample staged rate contributions for one valid point."""
    metrics = mathieu_metrics(m_dm, eps, cfg)
    if not metrics["pseudopotential_valid"]:
        raise ValueError("Refusing to sample a pseudopotential-invalid point")
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    common = make_common_samples(n_samples, cfg.seed if seed is None else int(seed))
    out = {
        "phonon_rate_s": [],
        "event_rate_ge1_s": [],
        "event_rate_exact_M_s": [],
        "nfev": [],
        "status": [],
        "regime": [],
        "branch": [],
        "b_tail_clipped": [],
        "tail_outer_radius_m": [],
    }
    for i in range(n_samples):
        rec = _importance_sample_contribution(
            m_dm, eps, cfg, ground, common, i,
            b_annulus_factors=b_annulus_factors,
            b_support_factor=b_support_factor,
            annulus_area_fraction=annulus_area_fraction,
            annulus_expand_outer_radius=annulus_expand_outer_radius,
            annulus_outer_radius_max_m=annulus_outer_radius_max_m,
        )
        for key in out:
            out[key].append(rec[key])
        if progress and ((i + 1) == 1 or (i + 1) % max(n_samples // 10, 1) == 0 or (i + 1) == n_samples):
            print(f"      diagnostic sample {i+1}/{n_samples}", flush=True)
    for key in ("phonon_rate_s", "event_rate_ge1_s", "event_rate_exact_M_s", "nfev"):
        out[key] = np.asarray(out[key], dtype=float)
    out["b_tail_clipped"] = np.asarray(out["b_tail_clipped"], dtype=bool)
    out["tail_outer_radius_m"] = np.asarray(out["tail_outer_radius_m"], dtype=float)
    return out


def adaptive_point_estimate(
    m_dm: float,
    eps: float,
    cfg: ScanConfig,
    ground: CopperGroundPlane,
    conv: ConvergenceConfig,
    *,
    seed: int | None = None,
    progress: bool = False,
) -> dict:
    """Adaptive fixed-point staged estimate used as the central result."""
    metrics = mathieu_metrics(m_dm, eps, cfg)
    if not metrics["pseudopotential_valid"]:
        raise ValueError("Refusing to sample a pseudopotential-invalid point")
    nmax = max(int(conv.mc_max_samples), int(conv.mc_min_samples))
    nmin = min(max(int(conv.mc_min_samples), 2), nmax)
    check_every = max(int(conv.mc_check_every), 1)
    common = make_common_samples(nmax, cfg.seed if seed is None else int(seed))
    vals = {k: [] for k in ("phonon_rate_s", "event_rate_ge1_s", "event_rate_exact_M_s")}
    nfev = []
    statuses, regimes = [], []
    stop_reason = "max_samples"
    used = 0
    for i in range(nmax):
        rec = _importance_sample_contribution(m_dm, eps, cfg, ground, common, i)
        for key in vals:
            vals[key].append(rec[key])
        nfev.append(rec["nfev"])
        statuses.append(rec["status"])
        regimes.append(rec["regime"])
        used = i + 1
        if progress and (used == 1 or used % check_every == 0 or used == nmax):
            s1 = _metric_stats(np.asarray(vals["phonon_rate_s"], float))
            s2 = _metric_stats(np.asarray(vals["event_rate_ge1_s"], float))
            s3 = _metric_stats(np.asarray(vals["event_rate_exact_M_s"], float))
            print(
                f"      MC {used}/{nmax}: rel.SE phonon={s1['rel_se']:.3g}, "
                f"Gamma>=1={s2['rel_se']:.3g}, Gamma_M={s3['rel_se']:.3g}",
                flush=True,
            )
        if used >= nmin and (used % check_every == 0 or used == nmax):
            s_ph = _metric_stats(np.asarray(vals["phonon_rate_s"], float))
            s_ge = _metric_stats(np.asarray(vals["event_rate_ge1_s"], float))
            # Stop on the two most generally well-conditioned observables.  The
            # exact-M rate receives its own reliability flag and may legitimately
            # require more samples in high-multiplicity regions.
            rels = [s_ph["rel_se"], s_ge["rel_se"]]
            if all(np.isfinite(x) and x <= conv.mc_target_rel_se for x in rels):
                stop_reason = "target_rel_se"
                break
    result = {
        "m_dm_kg": float(m_dm),
        "eps": float(eps),
        "mathieu_stable": bool(metrics["mathieu_stable"]),
        "pseudopotential_valid": True,
        "q_max": float(metrics["q_max"]),
        "secular_over_rf": float(metrics["secular_over_rf"]),
        "reference_class": int(metrics["reference_class"]),
        "effective_samples": int(used),
        "adaptive_stop_reason": stop_reason,
        "mean_nfev": float(np.mean(nfev)) if nfev else 0.0,
        "timeout_fraction": float(np.mean(["timeout" in s for s in statuses])) if statuses else 0.0,
        "analytic_shortcut_fraction": float(np.mean([s == "analytic_shortcut" for s in statuses])) if statuses else 0.0,
    }
    for metric, prefix in (
        ("phonon_rate_s", "phonon"),
        ("event_rate_ge1_s", "ge1"),
        ("event_rate_exact_M_s", "exact_M"),
    ):
        stats = _metric_stats(np.asarray(vals[metric], dtype=float))
        result[metric] = stats["mean"]
        # Keep both names.  ``*_se_s`` is the historical convergence-audit
        # column, while ``*_mc_se_s`` is the explicit Monte Carlo standard-
        # error name used by the quality-classification/output layer.
        result[f"{prefix}_se_s"] = stats["se"]
        result[f"{prefix}_mc_se_s"] = stats["se"]
        result[f"{prefix}_mc_rel_se"] = stats["rel_se"]
    # Backward-compatible names used by previous notebooks.
    result["mc_rel_se"] = result["phonon_mc_rel_se"]
    result["event_mc_rel_se"] = result["ge1_mc_rel_se"]
    result["event_exact_M_mc_rel_se"] = result["exact_M_mc_rel_se"]
    return result


def _paired_convergence_metric(base: np.ndarray, variant: np.ndarray, z: float) -> dict:
    base = np.asarray(base, float)
    variant = np.asarray(variant, float)
    if base.shape != variant.shape:
        raise ValueError("Paired convergence arrays must have identical shape")
    diff = variant - base
    mb = float(np.mean(base))
    mv = float(np.mean(variant))
    md = float(np.mean(diff))
    sed = float(np.std(diff, ddof=1) / math.sqrt(diff.size)) if diff.size > 1 else np.nan
    scale = max(abs(mb), abs(mv), np.finfo(float).tiny)
    rel_shift = abs(md) / scale
    rel_upper = (abs(md) + z * sed) / scale if np.isfinite(sed) else np.inf
    return {
        "base_mean": mb,
        "variant_mean": mv,
        "mean_difference": md,
        "difference_se": sed,
        "relative_shift": rel_shift,
        "relative_upper": rel_upper,
    }


def _baseline_standard_error(baseline: dict, label: str) -> float:
    for key in (f"{label}_mc_se_s", f"{label}_se_s"):
        try:
            value = float(baseline.get(key, np.nan))
        except (TypeError, ValueError):
            value = np.nan
        if np.isfinite(value) and value >= 0.0:
            return value
    rate_key = {
        "phonon": "phonon_rate_s", "ge1": "event_rate_ge1_s", "exact_M": "event_rate_exact_M_s"
    }[label]
    try:
        rate = float(baseline.get(rate_key, np.nan))
        rel = float(baseline.get(f"{label}_mc_rel_se", np.nan))
    except (TypeError, ValueError):
        return np.nan
    return abs(rate) * rel if np.isfinite(rate) and np.isfinite(rel) else np.nan


def _summary_from_mean_se(mean: float, se: float, n: int) -> dict:
    n = max(int(n), 0)
    return {"mean": float(mean), "se": float(se), "n": n}


def _combine_metric_summaries(a: dict, b: dict) -> dict:
    """Pool two independent sample summaries without requiring raw samples."""
    n1 = int(a.get("n", 0)); n2 = int(b.get("n", 0))
    if n1 <= 0:
        return dict(b)
    if n2 <= 0:
        return dict(a)
    m1 = float(a.get("mean", np.nan)); m2 = float(b.get("mean", np.nan))
    e1 = float(a.get("se", np.nan)); e2 = float(b.get("se", np.nan))
    if not (np.isfinite(m1) and np.isfinite(m2)):
        return {"mean": np.nan, "se": np.nan, "n": n1 + n2}
    # se = sample_std/sqrt(n). Recover the within-block M2 when possible.
    def m2_from(n, se):
        if n <= 1 or not np.isfinite(se):
            return 0.0 if n <= 1 else np.nan
        var = (se * math.sqrt(n)) ** 2
        return (n - 1) * var
    M21 = m2_from(n1, e1); M22 = m2_from(n2, e2)
    n = n1 + n2
    mean = (n1 * m1 + n2 * m2) / n
    if np.isfinite(M21) and np.isfinite(M22) and n > 1:
        delta = m2 - m1
        M2 = M21 + M22 + delta * delta * n1 * n2 / n
        sample_var = max(M2 / (n - 1), 0.0)
        se = math.sqrt(sample_var / n)
    else:
        se = np.nan
    return {"mean": float(mean), "se": float(se), "n": int(n)}


def _tail_sample_count_from_legacy_row(row: dict, conv: ConvergenceConfig) -> int:
    """Infer per-annulus N from an older adaptive-b checkpoint row."""
    try:
        n1 = int(float(row.get("b_tail1_n", 0) or 0))
        n2 = int(float(row.get("b_tail2_n", 0) or 0))
        if n1 > 0 and n2 > 0:
            return min(n1, n2)
    except Exception:
        pass
    try:
        total = int(float(row.get("adaptive_bmax_tail_samples", 0) or 0))
        expansions = int(float(row.get("adaptive_bmax_expansions", 0) or 0))
        denom = 2 + max(expansions, 0)
        if total > 0 and denom > 0:
            guess = int(round(total / denom))
            if guess > 0:
                return guess
    except Exception:
        pass
    return max(int(conv.b_tail_samples), 2)


def _reusable_tail_summaries(row: dict, conv: ConvergenceConfig):
    """Return the final two annulus summaries already stored in a checkpoint."""
    labels = ("phonon", "ge1", "exact_M")
    n = _tail_sample_count_from_legacy_row(row, conv)
    ann1 = {"metrics": {}, "clipped": _safe_bool(row.get("b_tail_clipped", False), False)}
    ann2 = {"metrics": {}, "clipped": _safe_bool(row.get("b_tail_clipped", False), False)}
    for label in labels:
        vals = []
        for prefix in ("b_tail1", "b_tail2"):
            try:
                mean = float(row.get(f"{prefix}_{label}_rate_s", np.nan))
                se = float(row.get(f"{prefix}_{label}_se_s", np.nan))
            except Exception:
                mean = se = np.nan
            vals.append((mean, se))
        if not all(np.isfinite(x) for pair in vals for x in pair):
            return None, None
        ann1["metrics"][label] = _summary_from_mean_se(vals[0][0], vals[0][1], n)
        ann2["metrics"][label] = _summary_from_mean_se(vals[1][0], vals[1][1], n)
    ann1["n"] = n; ann2["n"] = n
    return ann1, ann2



def _adaptive_annulus_records(row: dict, label: str, conv: ConvergenceConfig) -> list[dict]:
    """Recover unique octave-annulus summaries from checkpoint history.

    Each support step stores [S,2S] and [2S,4S]. Consecutive steps therefore
    duplicate one octave. Keep the highest-statistics non-clipped version of
    each octave so old checkpoint evidence can be reused for asymptotic triage.
    """
    try:
        hist = json.loads(str(row.get("adaptive_bmax_history", "[]")))
    except Exception:
        hist = []
    if not isinstance(hist, list):
        hist = []
    best = {}
    f1 = float(conv.b_tail_factor_1)
    for h in hist:
        if not isinstance(h, dict) or _safe_bool(h.get("clipped", False), False):
            continue
        try:
            support = float(h.get("support_factor", np.nan))
        except Exception:
            continue
        if not np.isfinite(support) or support <= 0:
            continue
        mm = h.get("metrics", {}).get(label, {}) if isinstance(h.get("metrics", {}), dict) else {}
        if not isinstance(mm, dict):
            continue
        for which, lower in (("ann1", support), ("ann2", support * f1)):
            try:
                mean = float(mm.get(f"{which}_mean", np.nan))
                se = float(mm.get(f"{which}_se", np.nan))
                n = int(mm.get(f"{which}_n", h.get("annulus_n", 0)) or 0)
            except Exception:
                continue
            if not (np.isfinite(mean) and np.isfinite(se) and n > 0 and mean >= 0):
                continue
            key = round(math.log(lower, f1), 10)
            rec = {"lower_factor": float(lower), "upper_factor": float(lower * f1),
                   "mean": mean, "se": se, "n": n}
            old = best.get(key)
            if old is None or n > old["n"] or (n == old["n"] and se < old["se"]):
                best[key] = rec
    return sorted(best.values(), key=lambda r: r["lower_factor"])


def _asymptotic_tail_diagnostics(row: dict, conv: ConvergenceConfig) -> dict:
    """Fit octave-rate decay and estimate a conservative residual-tail bound.

    This is diagnostic-only. Even a good fit does not set b_tail_*_pass=True;
    the point remains support-unresolved until explicit support or a dedicated
    single-point calculation certifies it.
    """
    out = {}
    enabled = bool(conv.asymptotic_tail_enabled)
    out["asymptotic_tail_enabled"] = enabled
    out["asymptotic_tail_analysis_version"] = 1
    if not enabled:
        out["asymptotic_tail_status"] = "disabled"
        return out
    support = float(row.get("adaptive_bmax_factor", 1.0) or 1.0)
    metric_map = {
        "phonon": "phonon_rate_s",
        "ge1": "event_rate_ge1_s",
        "exact_M": "event_rate_exact_M_s",
    }
    statuses = []
    for label, metric in metric_map.items():
        recs = _adaptive_annulus_records(row, label, conv)
        # Prefer the outermost measured octaves; include enough points just
        # inside the current support to diagnose whether a decay law has settled.
        fit_n = max(int(conv.asymptotic_tail_fit_annuli), int(conv.asymptotic_tail_min_annuli))
        usable = [r for r in recs if r["mean"] > 0 and r["lower_factor"] >= support / (float(conv.b_tail_factor_1) ** max(fit_n - 2, 0))]
        usable = usable[-fit_n:]
        status = "insufficient_history"
        p = r2 = ratio_fit = ratio_recent = ratio_upper = rem_upper = frac_upper = np.nan
        if len(usable) >= int(conv.asymptotic_tail_min_annuli):
            x = np.log(np.asarray([r["lower_factor"] for r in usable], float))
            y = np.log(np.asarray([r["mean"] for r in usable], float))
            try:
                slope, intercept = np.polyfit(x, y, 1)
                yhat = slope * x + intercept
                ss_res = float(np.sum((y - yhat) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
                p = float(-slope)
                ratio_fit = float(conv.b_tail_factor_1 ** (-p)) if np.isfinite(p) else np.nan
            except Exception:
                p = r2 = ratio_fit = np.nan
            # Consecutive outer-octave ratio, with a conservative 2-sigma ratio.
            a, b = usable[-2], usable[-1]
            if math.isclose(b["lower_factor"], a["upper_factor"], rel_tol=1e-8, abs_tol=0.0) and a["mean"] > 0:
                ratio_recent = b["mean"] / a["mean"]
                den_lo = a["mean"] - conv.confidence_z * a["se"]
                num_hi = b["mean"] + conv.confidence_z * b["se"]
                ratio_upper = num_hi / den_lo if den_lo > 0 else np.inf
            # Find the first omitted octave beginning at the current central support.
            first = next((r for r in recs if math.isclose(r["lower_factor"], support, rel_tol=1e-8, abs_tol=0.0)), None)
            quality = bool(
                np.isfinite(p) and p >= conv.asymptotic_tail_min_exponent
                and np.isfinite(r2) and r2 >= conv.asymptotic_tail_min_r2
                and np.isfinite(ratio_fit) and ratio_fit < 1.0
            )
            ratio_bound = max(x for x in (ratio_fit, ratio_recent, ratio_upper) if np.isfinite(x)) if any(np.isfinite(x) for x in (ratio_fit, ratio_recent, ratio_upper)) else np.inf
            if quality and first is not None and np.isfinite(ratio_bound) and ratio_bound < min(1.0, conv.asymptotic_tail_max_ratio):
                first_hi = max(0.0, first["mean"] + conv.confidence_z * first["se"])
                rem_upper = first_hi / max(1.0 - ratio_bound, np.finfo(float).tiny)
                central = max(float(row.get(metric, 0.0) or 0.0), 0.0)
                frac_upper = rem_upper / max(central + rem_upper, np.finfo(float).tiny)
                status = "candidate_bounded" if frac_upper <= conv.b_tail_fraction_tol else "decaying_above_tolerance"
            elif quality:
                status = "decay_fit_but_no_conservative_ratio"
            elif np.isfinite(p) and p > 0:
                status = "weak_or_nonstationary_decay"
            else:
                status = "nondecaying_or_noisy"
        out[f"asymptotic_{label}_annuli_used"] = int(len(usable))
        out[f"asymptotic_{label}_exponent_p"] = p
        out[f"asymptotic_{label}_fit_r2"] = r2
        out[f"asymptotic_{label}_ratio_fit"] = ratio_fit
        out[f"asymptotic_{label}_ratio_recent"] = ratio_recent
        out[f"asymptotic_{label}_ratio_upper"] = ratio_upper
        out[f"asymptotic_{label}_remainder_rate_upper_s"] = rem_upper
        out[f"asymptotic_{label}_remainder_fraction_upper"] = frac_upper
        out[f"asymptotic_{label}_status"] = status
        statuses.append(status)
    if statuses and all(x == "candidate_bounded" for x in statuses):
        overall = "candidate_bounded_all"
    elif any(x in ("nondecaying_or_noisy", "insufficient_history") for x in statuses):
        overall = "not_bounded"
    elif any(x == "decaying_above_tolerance" for x in statuses):
        overall = "decaying_above_tolerance"
    else:
        overall = "mixed_or_weak_decay"
    out["asymptotic_tail_status"] = overall
    out["asymptotic_tail_diagnostic_only"] = True
    out["asymptotic_tail_recommendation"] = (
        "single_point_v16_or_manual_tail_study" if overall != "candidate_bounded_all"
        else "candidate_for_manual_support_review"
    )
    return out


def add_asymptotic_tail_diagnostics(df: pd.DataFrame, conv: ConvergenceConfig) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    records = [_asymptotic_tail_diagnostics(dict(row), conv) for _, row in out.iterrows()]
    extra = pd.DataFrame(records, index=out.index)
    for col in extra.columns:
        out[col] = extra[col]
    return out

def _adaptive_b_tail_test(
    m_dm: float,
    eps: float,
    cfg: ScanConfig,
    ground: CopperGroundPlane,
    conv: ConvergenceConfig,
    baseline: dict,
    *,
    seed: int,
    progress: bool = False,
    checkpoint_callback=None,
) -> dict:
    """Repair/enlarge impact-parameter support while reusing previous results.

    Version 3 is deliberately incremental and status-aware.  If ``baseline`` already contains a
    older adaptive-b result, its enlarged central estimate and final two annulus
    summaries are reused.  The algorithm first increases annulus statistics
    when a failed 2-sigma bound is statistically ambiguous.  It expands the
    physical support only when the tail is demonstrably too large (or after the
    configured tail-sample ceiling is reached).  Thus a 32B result is continued
    from 32B rather than restarted at B, and the original 2048 central samples
    are never repeated.
    """
    f1 = float(conv.b_tail_factor_1); f2 = float(conv.b_tail_factor_2)
    if not (1.0 < f1 < f2):
        raise ValueError("Require 1 < b_tail_factor_1 < b_tail_factor_2")
    if not math.isclose(f2, f1 * f1, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("Adaptive b_max requires b_tail_factor_2 == b_tail_factor_1**2")

    metrics = (
        ("phonon_rate_s", "phonon"),
        ("event_rate_ge1_s", "ge1"),
        ("event_rate_exact_M_s", "exact_M"),
    )
    for metric, _ in metrics:
        if not np.isfinite(float(baseline.get(metric, np.nan))):
            raise ValueError(f"Cannot adapt b_max without finite baseline {metric}")

    # Reuse any previously enlarged central support.  An older row's rate already
    # contains all promoted annuli up to adaptive_bmax_factor; do not double count.
    previous_factor = float(baseline.get("adaptive_bmax_factor", 1.0) or 1.0)
    previous_factor = previous_factor if np.isfinite(previous_factor) and previous_factor >= 1.0 else 1.0
    try:
        previous_version = int(float(baseline.get("adaptive_bmax_version", 0) or 0))
    except Exception:
        previous_version = 0
    reusable_ann1, reusable_ann2 = _reusable_tail_summaries(baseline, conv)
    # Reuse the enlarged central integral whenever an older adaptive run exists.
    # If its final audit was geometry-clipped, keep the central rate/support but
    # resample the two OUTER annuli with the v3 expandable launch sphere.
    reuse_central = bool(previous_version > 0)
    reuse_tail = bool(reusable_ann1 is not None and reusable_ann2 is not None)
    if reuse_tail and conv.adaptive_bmax_allow_outer_radius_expand:
        reuse_tail = not bool(reusable_ann1.get("clipped", False) or reusable_ann2.get("clipped", False))
    reuse_previous = reuse_central
    current_rate = {metric: float(baseline[metric]) for metric, _ in metrics}
    current_var = {
        label: (_baseline_standard_error(baseline, label) ** 2 if np.isfinite(_baseline_standard_error(baseline, label)) else np.inf)
        for _, label in metrics
    }
    support_factor = float(previous_factor if reuse_central else 1.0)
    expansion_count = int(float(baseline.get("adaptive_bmax_expansions", 0) or 0)) if reuse_central else 0
    old_tail_samples = int(float(baseline.get("adaptive_bmax_tail_samples", 0) or 0)) if reuse_central else 0
    total_tail_samples_new = 0
    try:
        history = json.loads(str(baseline.get("adaptive_bmax_history", "[]"))) if reuse_central else []
        if not isinstance(history, list): history = []
    except Exception:
        history = []

    nominal_rate = {}
    nominal_se = {}
    for metric, label in metrics:
        try:
            nr = float(baseline.get(f"nominal_bmax_{metric}", np.nan))
        except Exception:
            nr = np.nan
        nominal_rate[metric] = nr if np.isfinite(nr) else (current_rate[metric] if support_factor == 1.0 else np.nan)
        try:
            ns = float(baseline.get(f"nominal_bmax_{label}_se_s", np.nan))
        except Exception:
            ns = np.nan
        nominal_se[label] = ns if np.isfinite(ns) else (_baseline_standard_error(baseline, label) if support_factor == 1.0 else np.nan)

    def sample_annulus(lo_factor: float, hi_factor: float, n: int, ann_seed: int):
        nonlocal total_tail_samples_new
        rec = point_sample_contributions(
            m_dm, eps, cfg, ground, int(n), seed=int(ann_seed),
            b_annulus_factors=(lo_factor, hi_factor),
            annulus_area_fraction=conv.tail_importance_area_fraction,
            annulus_expand_outer_radius=conv.adaptive_bmax_allow_outer_radius_expand,
            annulus_outer_radius_max_m=conv.adaptive_bmax_outer_radius_max_m,
            progress=progress,
        )
        total_tail_samples_new += int(n)
        out = {"metrics": {}, "clipped": bool(np.any(rec["b_tail_clipped"])), "n": int(n),
               "max_outer_radius_m": float(np.nanmax(rec["tail_outer_radius_m"])) if len(rec["tail_outer_radius_m"]) else float(cfg.R_outer_m)}
        for metric, label in metrics:
            out["metrics"][label] = _metric_stats(rec[metric])
        return out

    ann1 = reusable_ann1 if reuse_tail else None
    ann2 = reusable_ann2 if reuse_tail else None
    if ann1 is None or ann2 is None:
        ann1 = sample_annulus(support_factor, support_factor * f1, conv.b_tail_samples, seed)
        ann2 = sample_annulus(support_factor * f1, support_factor * f2, conv.b_tail_samples, seed + 1)

    final_stats = None
    status = "support_unresolved"
    resample_round = 0
    max_factor = max(float(conv.adaptive_bmax_max_factor), 1.0)
    max_expansions = max(int(conv.adaptive_bmax_max_expansions), 0)
    max_tail_n = max(int(conv.adaptive_bmax_tail_max_samples), int(conv.b_tail_samples), 2)
    growth = max(float(conv.adaptive_bmax_tail_sample_growth), 1.01)

    def evaluate_step():
        clipped = bool(ann1.get("clipped", False) or ann2.get("clipped", False))
        step_stats = {
            "support_factor": float(support_factor), "clipped": clipped,
            "annulus_n": int(min(ann1.get("n", 0), ann2.get("n", 0))),
            "metrics": {},
        }
        all_pass = not clipped; any_clear_fail = False; any_ambiguous = False
        for metric, label in metrics:
            s1 = ann1["metrics"][label]; s2 = ann2["metrics"][label]
            e1 = s1["se"] if np.isfinite(s1["se"]) else np.inf
            e2 = s2["se"] if np.isfinite(s2["se"]) else np.inf
            m1 = max(float(s1["mean"]), 0.0); m2 = max(float(s2["mean"]), 0.0)
            ub1 = max(0.0, m1 + conv.confidence_z * e1)
            ub2 = max(0.0, m2 + conv.confidence_z * e2)
            lb1 = max(0.0, m1 - conv.confidence_z * e1) if np.isfinite(e1) else 0.0
            lb2 = max(0.0, m2 - conv.confidence_z * e2) if np.isfinite(e2) else 0.0
            central_now = max(float(current_rate[metric]), 0.0)
            denom = max(central_now + m1 + m2, np.finfo(float).tiny)
            frac_mean = (m1 + m2) / denom
            frac_upper = (ub1 + ub2) / denom
            frac_lower = max(0.0, lb1 + lb2) / denom
            decay_ok = ub2 <= conv.b_tail_decay_factor * max(ub1, np.finfo(float).tiny)
            # A clear failure requires the lower confidence bound to violate
            # either the tail-size or decay criterion. Otherwise the result is
            # statistically ambiguous and should receive more annulus samples.
            clear_tail_fail = frac_lower > conv.b_tail_fraction_tol
            clear_decay_fail = lb2 > conv.b_tail_decay_factor * max(ub1, np.finfo(float).tiny)
            passed = bool((not clipped) and np.isfinite(frac_upper) and frac_upper <= conv.b_tail_fraction_tol and decay_ok)
            clear_fail = bool(clear_tail_fail or clear_decay_fail)
            ambiguous = bool((not passed) and (not clear_fail) and (not clipped))
            all_pass = all_pass and passed
            any_clear_fail = any_clear_fail or clear_fail
            any_ambiguous = any_ambiguous or ambiguous
            step_stats["metrics"][label] = {
                "ann1_mean": float(s1["mean"]), "ann1_se": float(s1["se"]), "ann1_n": int(s1["n"]),
                "ann2_mean": float(s2["mean"]), "ann2_se": float(s2["se"]), "ann2_n": int(s2["n"]),
                "ub1": ub1, "ub2": ub2, "lb1": lb1, "lb2": lb2,
                "frac_mean": frac_mean, "frac_lower": frac_lower, "frac_upper": frac_upper,
                "decay_ok": bool(decay_ok), "clear_fail": clear_fail,
                "ambiguous": ambiguous, "pass": passed,
            }
        step_stats["all_pass"] = bool(all_pass)
        step_stats["any_clear_fail"] = bool(any_clear_fail)
        step_stats["any_ambiguous"] = bool(any_ambiguous)
        return step_stats

    def build_output(status_value: str, complete: bool, fs: dict | None):
        out = dict(baseline)
        out["adaptive_bmax_version"] = int(conv.adaptive_bmax_version)
        out["adaptive_bmax_enabled"] = bool(conv.adaptive_bmax_enabled)
        out["adaptive_bmax_complete"] = bool(complete)
        out["adaptive_bmax_status"] = str(status_value)
        out["adaptive_bmax_factor"] = float(support_factor)
        out["adaptive_bmax_expansions"] = int(expansion_count)
        out["adaptive_bmax_tail_samples"] = int(old_tail_samples + total_tail_samples_new)
        out["adaptive_bmax_tail_samples_new"] = int(total_tail_samples_new)
        out["adaptive_bmax_reused_previous"] = bool(reuse_previous)
        out["adaptive_bmax_reused_tail_summaries"] = bool(reuse_tail)
        out["adaptive_bmax_tail_importance_area_fraction"] = float(conv.tail_importance_area_fraction)
        out["adaptive_bmax_tail_outer_radius_m"] = float(max(
            ann1.get("max_outer_radius_m", cfg.R_outer_m) if isinstance(ann1, dict) else cfg.R_outer_m,
            ann2.get("max_outer_radius_m", cfg.R_outer_m) if isinstance(ann2, dict) else cfg.R_outer_m,
        ))
        out["adaptive_bmax_history"] = json.dumps(history, separators=(",", ":"), allow_nan=True)
        out["b_tail_clipped"] = bool(fs["clipped"] if fs else False)
        all_support_pass = True
        for metric, label in metrics:
            rate = float(current_rate[metric])
            se = math.sqrt(current_var[label]) if np.isfinite(current_var[label]) else np.nan
            rel = se / abs(rate) if np.isfinite(se) and rate != 0.0 else np.nan
            out[metric] = rate
            out[f"{label}_se_s"] = se; out[f"{label}_mc_se_s"] = se; out[f"{label}_mc_rel_se"] = rel
            if np.isfinite(nominal_rate[metric]):
                out[f"nominal_bmax_{metric}"] = nominal_rate[metric]
                out[f"adaptive_bmax_added_{label}_rate_s"] = rate - nominal_rate[metric]
                out[f"adaptive_bmax_added_{label}_fraction"] = (rate - nominal_rate[metric]) / rate if rate != 0 else np.nan
            if np.isfinite(nominal_se[label]): out[f"nominal_bmax_{label}_se_s"] = nominal_se[label]
            if fs:
                mm = fs["metrics"][label]
                out[f"b_tail1_{label}_rate_s"] = mm["ann1_mean"]
                out[f"b_tail2_{label}_rate_s"] = mm["ann2_mean"]
                out[f"b_tail1_{label}_se_s"] = mm["ann1_se"]
                out[f"b_tail2_{label}_se_s"] = mm["ann2_se"]
                out[f"b_tail1_n"] = int(mm["ann1_n"]); out[f"b_tail2_n"] = int(mm["ann2_n"])
                out[f"b_tail_{label}_fraction_mean"] = mm["frac_mean"]
                out[f"b_tail_{label}_fraction_lower"] = mm["frac_lower"]
                out[f"b_tail_{label}_fraction_upper"] = mm["frac_upper"]
                out[f"b_tail_{label}_decay_ok"] = bool(mm["decay_ok"])
                out[f"b_tail_{label}_pass"] = bool(mm["pass"])
                all_support_pass = all_support_pass and bool(mm["pass"])
            else:
                out[f"b_tail_{label}_pass"] = False; all_support_pass = False
        out["support_converged"] = bool(all_support_pass and status_value == "converged")
        out["support_unresolved"] = not out["support_converged"]
        out["support_unresolved_reason"] = "" if out["support_converged"] else str(status_value)
        out["mc_rel_se"] = out.get("phonon_mc_rel_se", np.nan)
        out["event_mc_rel_se"] = out.get("ge1_mc_rel_se", np.nan)
        out["event_exact_M_mc_rel_se"] = out.get("exact_M_mc_rel_se", np.nan)
        return out

    while True:
        final_stats = evaluate_step()
        history.append({**final_stats, "source": "v3_repair", "resample_round": int(resample_round)})
        if progress:
            msg = ", ".join(
                f"{lab}: mean={100*final_stats['metrics'][lab]['frac_mean']:.1f}% "
                f"[2sigma {100*final_stats['metrics'][lab]['frac_lower']:.1f},{100*final_stats['metrics'][lab]['frac_upper']:.1f}]%"
                for lab in ("phonon", "ge1", "exact_M")
            )
            print(f"      adaptive b support={support_factor:g} B, Ntail={final_stats['annulus_n']} | {msg}", flush=True)

        snapshot = build_output("in_progress", False, final_stats)
        if checkpoint_callback is not None:
            checkpoint_callback(snapshot)

        if final_stats["all_pass"]:
            history[-1]["decision"] = "converged"
            status = "converged"
            break
        if final_stats["clipped"]:
            history[-1]["decision"] = "geometry_limited"
            status = "geometry_limited"
            break

        current_n = int(final_stats["annulus_n"])
        # If the 2-sigma failure could be sampling noise, spend more samples at
        # the SAME physical annuli before changing the support.
        if final_stats["any_ambiguous"] and not final_stats["any_clear_fail"] and current_n < max_tail_n:
            history[-1]["decision"] = "resample_same_annuli"
            target_n = min(max_tail_n, max(current_n + 1, int(math.ceil(current_n * growth))))
            extra_n = target_n - current_n
            resample_round += 1
            extra1 = sample_annulus(support_factor, support_factor * f1, extra_n, seed + 10000 + 20 * expansion_count + 2 * resample_round)
            extra2 = sample_annulus(support_factor * f1, support_factor * f2, extra_n, seed + 10001 + 20 * expansion_count + 2 * resample_round)
            for _, label in metrics:
                ann1["metrics"][label] = _combine_metric_summaries(ann1["metrics"][label], extra1["metrics"][label])
                ann2["metrics"][label] = _combine_metric_summaries(ann2["metrics"][label], extra2["metrics"][label])
            ann1["n"] = target_n; ann2["n"] = target_n
            ann1["clipped"] = bool(ann1.get("clipped", False) or extra1.get("clipped", False))
            ann2["clipped"] = bool(ann2.get("clipped", False) or extra2.get("clipped", False))
            continue

        if final_stats["any_ambiguous"] and not final_stats["any_clear_fail"] and current_n >= max_tail_n:
            history[-1]["decision"] = "tail_noise_limited"
            status = "tail_noise_limited"
            break

        next_factor = support_factor * f1
        if expansion_count >= max_expansions or next_factor > max_factor * (1.0 + 1e-12):
            history[-1]["decision"] = "max_factor_reached"
            status = "max_factor_reached"
            break

        history[-1]["decision"] = "expand_support"
        # The lower confidence bound establishes that the current support is
        # inadequate. Promote the inner tail annulus and continue outward.
        for metric, label in metrics:
            s1 = ann1["metrics"][label]
            current_rate[metric] += float(s1["mean"])
            if np.isfinite(current_var[label]) and np.isfinite(s1["se"]):
                current_var[label] += float(s1["se"]) ** 2
            else:
                current_var[label] = np.inf
        support_factor = next_factor
        expansion_count += 1
        resample_round = 0
        ann1 = ann2
        ann2 = sample_annulus(support_factor * f1, support_factor * f2, max(int(ann1.get("n", conv.b_tail_samples)), int(conv.b_tail_samples)), seed + 20000 + expansion_count)

    final_out = build_output(status, True, final_stats)
    final_out.update(_asymptotic_tail_diagnostics(final_out, conv))
    return final_out

def _b_tail_test(
    m_dm: float,
    eps: float,
    cfg: ScanConfig,
    ground: CopperGroundPlane,
    conv: ConvergenceConfig,
    baseline: dict,
    *,
    seed: int,
    progress: bool = False,
) -> dict:
    """Backward-compatible one-shot tail audit without central expansion."""
    conv_one = replace(conv, adaptive_bmax_max_expansions=0)
    upgraded = _adaptive_b_tail_test(
        m_dm, eps, cfg, ground, conv_one, baseline, seed=seed, progress=progress
    )
    keys = {k: v for k, v in upgraded.items() if k.startswith("b_tail_") or k == "b_tail_clipped"}
    return keys


def _has_complete_numerical_audit(row: dict) -> bool:
    for name in NUMERICAL_TEST_NAMES:
        for label in ("phonon", "ge1", "exact_M"):
            try:
                v = float(row.get(f"{name}_{label}_relative_upper", np.nan))
            except (AttributeError, TypeError, ValueError):
                return False
            if not np.isfinite(v):
                return False
    return True


def _clear_numerical_audit_fields(row: dict) -> None:
    for name in NUMERICAL_TEST_NAMES:
        for label in ("phonon", "ge1", "exact_M"):
            row[f"{name}_{label}_relative_shift"] = np.nan
            row[f"{name}_{label}_relative_upper"] = np.nan
            row[f"{name}_{label}_pass"] = False


def _complete_convergence_audit_from_baseline(
    m_dm: float,
    eps: float,
    cfg: ScanConfig,
    ground: CopperGroundPlane,
    conv: ConvergenceConfig,
    baseline: dict,
    *,
    seed0: int,
    progress: bool = True,
    checkpoint_callback=None,
) -> dict:
    """Complete adaptive support and numerical audits from an existing central estimate.

    Crucially, this function can consume a checkpoint central estimate computed
    by an older notebook.  The expensive [0,B] Monte Carlo is reused exactly;
    only missing outer annuli and, when warranted, small paired numerical
    diagnostic samples are simulated.
    """
    if progress:
        print("    adaptive impact-parameter support audit", flush=True)
    if conv.adaptive_bmax_enabled:
        result = _adaptive_b_tail_test(
            m_dm, eps, cfg, ground, conv, baseline, seed=seed0 + 101, progress=progress,
            checkpoint_callback=checkpoint_callback,
        )
        # Persist the completed support state BEFORE any paired numerical audit.
        # If a later numerical variant hits the point watchdog, resume keeps the
        # newly certified/repaired b-domain instead of falling back to the last
        # in-progress support snapshot.
        if checkpoint_callback is not None:
            checkpoint_callback(result)
    else:
        tail = _b_tail_test(m_dm, eps, cfg, ground, conv, baseline, seed=seed0 + 101, progress=progress)
        result = {**baseline, **tail}
        result["adaptive_bmax_version"] = int(conv.adaptive_bmax_version)
        result["adaptive_bmax_enabled"] = False
        result["adaptive_bmax_complete"] = True
        result["adaptive_bmax_status"] = "disabled"
        result["adaptive_bmax_factor"] = 1.0
        result["adaptive_bmax_expansions"] = 0

    preliminary_mc = {
        "phonon": bool(np.isfinite(result["phonon_mc_rel_se"]) and result["phonon_mc_rel_se"] <= conv.mc_target_rel_se),
        "ge1": bool(np.isfinite(result["ge1_mc_rel_se"]) and result["ge1_mc_rel_se"] <= conv.mc_target_rel_se),
        "exact_M": bool(np.isfinite(result["exact_M_mc_rel_se"]) and result["exact_M_mc_rel_se"] <= conv.mc_target_rel_se),
    }
    secondary_candidate = {
        "phonon": bool(np.isfinite(result["phonon_mc_rel_se"]) and result["phonon_mc_rel_se"] <= conv.well_converged_rel_se),
        "ge1": bool(np.isfinite(result["ge1_mc_rel_se"]) and result["ge1_mc_rel_se"] <= conv.well_converged_rel_se),
        "exact_M": bool(np.isfinite(result["exact_M_mc_rel_se"]) and result["exact_M_mc_rel_se"] <= conv.well_converged_rel_se),
    }
    timeout_ok = bool(float(result.get("timeout_fraction", np.inf)) <= conv.max_timeout_fraction)
    support_factor = float(result.get("adaptive_bmax_factor", 1.0))
    support_all_pass = all(bool(result.get(f"b_tail_{lab}_pass", False)) for lab in ("phonon", "ge1", "exact_M"))
    try:
        baseline_numeric_factor = float(baseline.get("numerical_audit_b_support_factor", np.nan))
    except Exception:
        baseline_numeric_factor = np.nan
    can_reuse_numeric = bool(
        support_all_pass
        and np.isfinite(baseline_numeric_factor)
        and math.isclose(support_factor, baseline_numeric_factor, rel_tol=1e-12, abs_tol=1e-15)
        and _has_complete_numerical_audit(baseline)
        and not _safe_bool(baseline.get("secondary_audit_skipped", False), False)
    )

    # Numerical convergence on an integration domain whose b support is still
    # open is not useful.  Preserve/return the central estimate and explicitly
    # mark the numerical audit as blocked by support.
    if not support_all_pass:
        if progress:
            print("    paired numerical audits blocked: adaptive b support is not converged", flush=True)
        _clear_numerical_audit_fields(result)
        result["secondary_audit_skipped"] = True
        result["numerical_audit_skipped"] = True
        result["numerical_audit_skip_reason"] = "support_not_converged"
        result["numerical_audit_reused"] = False
        result["numerical_audit_b_support_factor"] = np.nan
        result["timeout_pass"] = timeout_ok
        for label in ("phonon", "ge1", "exact_M"):
            result[f"mc_{label}_pass"] = preliminary_mc[label]
            result[f"numerical_{label}_pass"] = False
            result[f"reliable_{label}"] = False
        result["reliable_all"] = False
        failed=[]
        if not timeout_ok: failed.append("timeouts")
        for label in ("phonon", "ge1", "exact_M"):
            if not preliminary_mc[label]: failed.append(f"MC:{label}")
            if not bool(result.get(f"b_tail_{label}_pass", False)): failed.append(f"b-tail:{label}")
        result["reliability_failures"] = ";".join(failed)
        return result

    if conv.skip_secondary_if_all_mc_fail and not any(secondary_candidate.values()):
        if progress:
            print(
                "    paired numerical audits skipped: all observables exceed the well-converged MC threshold",
                flush=True,
            )
        # If the support was enlarged, any legacy numerical audit referred to
        # the wrong b-domain and must not be plotted as current.  If support
        # stayed at B, retain an existing complete numerical audit as useful
        # provenance even though it is not needed for classification.
        if not can_reuse_numeric:
            _clear_numerical_audit_fields(result)
        result["secondary_audit_skipped"] = True
        result["numerical_audit_skipped"] = True
        result["numerical_audit_reused"] = bool(can_reuse_numeric)
        result["numerical_audit_b_support_factor"] = support_factor if can_reuse_numeric else np.nan
        result["timeout_pass"] = timeout_ok
        for label in ("phonon", "ge1", "exact_M"):
            result[f"mc_{label}_pass"] = preliminary_mc[label]
            result[f"numerical_{label}_pass"] = bool(
                can_reuse_numeric and all(
                    float(result[f"{name}_{label}_relative_upper"]) <= conv.numerical_rel_tol
                    for name in NUMERICAL_TEST_NAMES
                )
            )
            result[f"reliable_{label}"] = False
        result["reliable_all"] = False
        failed = []
        if not timeout_ok:
            failed.append("timeouts")
        for label in ("phonon", "ge1", "exact_M"):
            if not preliminary_mc[label]: failed.append(f"MC:{label}")
            if not bool(result.get(f"b_tail_{label}_pass", False)): failed.append(f"b-tail:{label}")
        result["reliability_failures"] = ";".join(failed)
        return result

    variants = {
        "far_outer": replace(
            cfg,
            R_outer_m=cfg.R_outer_m * conv.outer_radius_factor,
            R_far_factor=cfg.R_far_factor * conv.r_far_factor,
        ),
        "timestep": replace(cfg, ode_max_step_s=cfg.ode_max_step_s * conv.timestep_factor),
        "tolerances": replace(
            cfg,
            ode_rtol=cfg.ode_rtol * conv.tolerance_factor,
            ode_atol_position_m=cfg.ode_atol_position_m * conv.tolerance_factor,
            ode_atol_velocity_m_s=cfg.ode_atol_velocity_m_s * conv.tolerance_factor,
            ode_atol_quadrature_N_s=cfg.ode_atol_quadrature_N_s * conv.tolerance_factor,
        ),
        "stage_time": replace(cfg, max_stage_time_s=cfg.max_stage_time_s * conv.stage_time_factor),
        "switch": replace(cfg, R_switch_factor=cfg.R_switch_factor * conv.r_switch_factor),
        "ground": replace(
            cfg,
            delayed_branch_samples=max(1, int(cfg.delayed_branch_samples * conv.delayed_samples_factor)),
            ground_path_replicas=max(1, int(getattr(cfg, "ground_path_replicas", 1) * conv.delayed_samples_factor)),
            max_ground_interactions=cfg.max_ground_interactions + int(conv.extra_ground_interactions),
        ),
    }

    if can_reuse_numeric:
        if progress:
            print("    reusing existing paired numerical audit at nominal b support", flush=True)
        numeric = {}
        for name in NUMERICAL_TEST_NAMES:
            for label in ("phonon", "ge1", "exact_M"):
                for suffix in ("relative_shift", "relative_upper", "pass"):
                    key = f"{name}_{label}_{suffix}"
                    numeric[key] = baseline.get(key, np.nan if suffix != "pass" else False)
        result["numerical_audit_reused"] = True
    else:
        if progress:
            print(f"    paired numerical convergence audit at b support={support_factor:g} B", flush=True)
        n_diag = max(int(conv.diagnostic_samples), 2)
        diag_base = point_sample_contributions(
            m_dm, eps, cfg, ground, n_diag, seed=seed0 + 202, b_support_factor=support_factor
        )
        numeric = {}
        for name, cfg_var in variants.items():
            if progress:
                print(f"      {name}", flush=True)
            var = point_sample_contributions(
                m_dm, eps, cfg_var, ground, n_diag, seed=seed0 + 202,
                b_support_factor=support_factor,
            )
            for metric, label in (
                ("phonon_rate_s", "phonon"),
                ("event_rate_ge1_s", "ge1"),
                ("event_rate_exact_M_s", "exact_M"),
            ):
                test = _paired_convergence_metric(diag_base[metric], var[metric], conv.confidence_z)
                numeric[f"{name}_{label}_relative_shift"] = test["relative_shift"]
                numeric[f"{name}_{label}_relative_upper"] = test["relative_upper"]
                numeric[f"{name}_{label}_pass"] = bool(
                    np.isfinite(test["relative_upper"])
                    and test["relative_upper"] <= conv.numerical_rel_tol
                )
        result["numerical_audit_reused"] = False

    result.update(numeric)
    result["timeout_pass"] = timeout_ok
    result["secondary_audit_skipped"] = False
    result["numerical_audit_skipped"] = False
    result["numerical_audit_b_support_factor"] = support_factor
    for label, rel_key in (
        ("phonon", "phonon_mc_rel_se"),
        ("ge1", "ge1_mc_rel_se"),
        ("exact_M", "exact_M_mc_rel_se"),
    ):
        mc_rel = result[rel_key]
        mc_pass = bool(np.isfinite(mc_rel) and mc_rel <= conv.mc_target_rel_se)
        numeric_pass = all(bool(result[f"{name}_{label}_pass"]) for name in variants)
        b_pass = bool(result[f"b_tail_{label}_pass"])
        result[f"mc_{label}_pass"] = mc_pass
        result[f"numerical_{label}_pass"] = numeric_pass
        result[f"reliable_{label}"] = bool(mc_pass and numeric_pass and b_pass and timeout_ok)
    result["reliable_all"] = bool(
        result["reliable_phonon"] and result["reliable_ge1"] and result["reliable_exact_M"]
    )
    failed = []
    if not timeout_ok:
        failed.append("timeouts")
    for label in ("phonon", "ge1", "exact_M"):
        if not result[f"mc_{label}_pass"]: failed.append(f"MC:{label}")
        if not result[f"b_tail_{label}_pass"]: failed.append(f"b-tail:{label}")
        if not result[f"numerical_{label}_pass"]: failed.append(f"numerical:{label}")
    result["reliability_failures"] = ";".join(failed)
    return result


def upgrade_existing_audit_point(
    m_dm: float,
    eps: float,
    cfg: ScanConfig,
    ground: CopperGroundPlane,
    conv: ConvergenceConfig,
    baseline: dict,
    *,
    seed: int | None = None,
    progress: bool = True,
    checkpoint_callback=None,
) -> dict:
    """Upgrade a legacy finite checkpoint row without rerunning its central MC."""
    if not _has_finite_central_estimate(baseline):
        raise ValueError("upgrade_existing_audit_point requires a finite central checkpoint result")
    seed0 = cfg.seed if seed is None else int(seed)
    # Strip fatal flags caused only by an older secondary-audit timeout.  The
    # central estimate itself is being intentionally reused and the new worker
    # receives a fresh watchdog budget for the adaptive support audit.
    base = dict(baseline)
    base["wallclock_timeout"] = False
    base["worker_failure"] = False
    base["central_estimate_reused"] = True
    return _complete_convergence_audit_from_baseline(
        m_dm, eps, cfg, ground, conv, base, seed0=seed0, progress=progress,
        checkpoint_callback=checkpoint_callback,
    )


def convergence_audit_point(
    m_dm: float,
    eps: float,
    cfg: ScanConfig,
    ground: CopperGroundPlane,
    conv: ConvergenceConfig,
    *,
    seed: int | None = None,
    progress: bool = True,
) -> dict:
    """Return a central staged estimate plus convergence/reliability flags."""
    if not mathieu_metrics(m_dm, eps, cfg)["pseudopotential_valid"]:
        raise ValueError("Refusing to audit a pseudopotential-invalid point")
    seed0 = cfg.seed if seed is None else int(seed)
    if progress:
        print("    central adaptive estimate", flush=True)
    baseline = central_point_estimate_auto(m_dm, eps, cfg, ground, conv, seed=seed0, progress=progress)
    return _complete_convergence_audit_from_baseline(
        m_dm, eps, cfg, ground, conv, baseline, seed0=seed0, progress=progress
    )


RESULT_CLASS_LABELS = {
    0: "unresolved",
    1: "noisy",
    2: "support_unresolved",
    3: "estimated",
    4: "well_converged",
    5: "precision_certified",
}

NUMERICAL_TEST_NAMES = ("far_outer", "timestep", "tolerances", "stage_time", "switch", "ground")
NUMERICAL_FAILURE_LABELS = {
    0: "not_audited_mc",
    1: "pass_all",
    2: "far_outer",
    3: "timestep",
    4: "tolerances",
    5: "stage_time",
    6: "switch",
    7: "ground",
    8: "timeout_or_failure",
    9: "blocked_by_support",
}


def _safe_bool(value, default: bool = False) -> bool:
    """Parse checkpoint booleans without treating NaN as True.

    Pandas fills columns that did not exist in older checkpoint rows with NaN;
    Python's ``bool(np.nan)`` is True, which can otherwise misclassify legacy
    rows as timeouts/failures.
    """
    if value is None:
        return bool(default)
    try:
        if pd.isna(value):
            return bool(default)
    except Exception:
        pass
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y", "t"}:
            return True
        if text in {"false", "0", "no", "n", "f", ""}:
            return False
    return bool(value)


def classify_audit_observable(row: dict | pd.Series, label: str, conv: ConvergenceConfig) -> dict:
    """Classify one observable while keeping support uncertainty explicit.

    A finite 10--20% MC estimate is no longer called ``estimated`` when the
    adaptive impact-parameter support is still open.  Such a point receives the
    separate ``support_unresolved`` class.  This prevents a small sampling SE
    from hiding a potentially biased truncation in b.
    """
    label = str(label)
    mapping = {
        "phonon": ("phonon_rate_s", ("phonon_mc_se_s", "phonon_se_s"), "phonon_mc_rel_se"),
        "ge1": ("event_rate_ge1_s", ("ge1_mc_se_s", "ge1_se_s"), "ge1_mc_rel_se"),
        "exact_M": ("event_rate_exact_M_s", ("exact_M_mc_se_s", "exact_M_se_s"), "exact_M_mc_rel_se"),
    }
    if label not in mapping:
        raise ValueError(f"Unknown observable label {label!r}")
    rate_key, se_keys, rel_key = mapping[label]
    def get(key, default=np.nan):
        try: return row.get(key, default)
        except AttributeError: return default
    def first_finite(keys):
        for key in keys:
            try: value = float(get(key, np.nan))
            except Exception: value = np.nan
            if np.isfinite(value): return value
        return np.nan
    try: rate = float(get(rate_key, np.nan))
    except Exception: rate = np.nan
    se = first_finite(se_keys)
    try: rel = float(get(rel_key, np.nan))
    except Exception: rel = np.nan
    if not np.isfinite(se) and np.isfinite(rate) and np.isfinite(rel): se = abs(rate) * rel
    if not np.isfinite(rel) and np.isfinite(rate) and rate != 0.0 and np.isfinite(se): rel = se / abs(rate)

    worker_failure = _safe_bool(get("worker_failure", False), False)
    wallclock_timeout = _safe_bool(get("wallclock_timeout", False), False)
    try: timeout_fraction = float(get("timeout_fraction", np.nan))
    except Exception: timeout_fraction = np.nan
    timeout_pass = _safe_bool(
        get("timeout_pass", np.isfinite(timeout_fraction) and timeout_fraction <= conv.max_timeout_fraction),
        np.isfinite(timeout_fraction) and timeout_fraction <= conv.max_timeout_fraction,
    )
    if worker_failure:
        return {"code": 0, "classification": RESULT_CLASS_LABELS[0], "reason": "worker failure"}
    if not np.isfinite(rate) or not np.isfinite(se) or not np.isfinite(rel):
        return {"code": 0, "classification": RESULT_CLASS_LABELS[0], "reason": "non-finite central estimate or Monte Carlo error"}
    if not timeout_pass or (np.isfinite(timeout_fraction) and timeout_fraction > conv.max_timeout_fraction):
        return {"code": 0, "classification": RESULT_CLASS_LABELS[0], "reason": "trajectory timeout fraction exceeds allowed limit"}
    if rel > conv.noisy_rel_se:
        return {"code": 0, "classification": RESULT_CLASS_LABELS[0], "reason": f"MC relative SE exceeds {100*conv.noisy_rel_se:.0f}%"}

    support_pass = _safe_bool(get(f"b_tail_{label}_pass", False), False)
    support_complete = _safe_bool(get("adaptive_bmax_complete", False), False)
    support_status = str(get("adaptive_bmax_status", "not_audited"))
    # A point-level watchdog after a finite central estimate is not allowed to
    # masquerade as precision convergence, but it can remain a finite estimate.
    secondary_skipped = _safe_bool(get("secondary_audit_skipped", True), True)
    numerical_pass = _safe_bool(get(f"numerical_{label}_pass", False), False)
    full_audit_pass = support_complete and support_pass and (not secondary_skipped) and numerical_pass and timeout_pass and (not wallclock_timeout)

    if full_audit_pass and rel <= conv.precision_rel_se:
        return {"code": 5, "classification": RESULT_CLASS_LABELS[5], "reason": f"MC rel.SE <= {100*conv.precision_rel_se:.0f}% and support, numerical, and timeout audits all pass"}
    if full_audit_pass and rel <= conv.well_converged_rel_se:
        return {"code": 4, "classification": RESULT_CLASS_LABELS[4], "reason": f"MC rel.SE <= {100*conv.well_converged_rel_se:.0f}% and support, numerical, and timeout audits all pass"}

    # Support failure outranks the ordinary statistical classes: a finite rate
    # exists, but its b-domain may still omit physical contribution.
    if support_complete and not support_pass:
        return {"code": 2, "classification": RESULT_CLASS_LABELS[2], "reason": f"finite central estimate, but adaptive b support is unresolved ({support_status})"}

    if rel <= conv.estimated_rel_se:
        extra = "; numerical audit incomplete" if (secondary_skipped or wallclock_timeout) else ""
        return {"code": 3, "classification": RESULT_CLASS_LABELS[3], "reason": f"MC rel.SE <= {100*conv.estimated_rel_se:.0f}% with bounded b support{extra}"}
    if rel <= conv.noisy_rel_se:
        return {"code": 1, "classification": RESULT_CLASS_LABELS[1], "reason": f"finite estimate with bounded b support but MC relative SE is {100*rel:.1f}%"}
    return {"code": 0, "classification": RESULT_CLASS_LABELS[0], "reason": "unresolved"}

def add_numerical_failure_diagnostics(df: pd.DataFrame, conv: ConvergenceConfig) -> pd.DataFrame:
    """Identify the limiting numerical test, or why no numerical audit exists."""
    out = df.copy()
    name_to_code = {name: i + 2 for i, name in enumerate(NUMERICAL_TEST_NAMES)}
    for label in ("phonon", "ge1", "exact_M"):
        max_vals=[]; dom_names=[]; dom_codes=[]; dom_pass=[]
        for _, row in out.iterrows():
            if _safe_bool(row.get("worker_failure", False), False):
                max_vals.append(np.nan); dom_names.append("timeout_or_failure"); dom_codes.append(8); dom_pass.append(False); continue
            vals=[]
            for name in NUMERICAL_TEST_NAMES:
                try: v=float(row.get(f"{name}_{label}_relative_upper", np.nan))
                except Exception: v=np.nan
                if np.isfinite(v): vals.append((name,v))
            if not vals:
                if _safe_bool(row.get("adaptive_bmax_complete", False), False) and not _safe_bool(row.get(f"b_tail_{label}_pass", False), False):
                    dom="blocked_by_support"; code=9
                elif _safe_bool(row.get("wallclock_timeout", False), False):
                    dom="timeout_or_failure"; code=8
                else:
                    dom="not_audited_mc"; code=0
                max_vals.append(np.nan); dom_names.append(dom); dom_codes.append(code); dom_pass.append(False); continue
            name,vmax=max(vals,key=lambda x:x[1])
            all_pass=len(vals)==len(NUMERICAL_TEST_NAMES) and all(v<=conv.numerical_rel_tol for _,v in vals)
            max_vals.append(float(vmax))
            if all_pass:
                dom_names.append("pass_all"); dom_codes.append(1); dom_pass.append(True)
            else:
                dom_names.append(name); dom_codes.append(name_to_code[name]); dom_pass.append(False)
        out[f"max_numerical_{label}_relative_upper"]=max_vals
        out[f"dominant_numerical_{label}"]=dom_names
        out[f"dominant_numerical_{label}_code"]=dom_codes
        out[f"dominant_numerical_{label}_pass"]=dom_pass
    return out

def add_result_classifications(df: pd.DataFrame, conv: ConvergenceConfig) -> pd.DataFrame:
    """Return a copy of an audit table with per-observable quality classes.

    This can be applied to old checkpoint rows, so upgrading the notebook does
    not require repeating completed central estimates merely to obtain the new
    classification columns.
    """
    out = df.copy()
    for label in ("phonon", "ge1", "exact_M"):
        recs = [classify_audit_observable(row, label, conv) for _, row in out.iterrows()]
        out[f"{label}_class_code"] = [r["code"] for r in recs]
        out[f"{label}_classification"] = [r["classification"] for r in recs]
        out[f"{label}_classification_reason"] = [r["reason"] for r in recs]
        rel_key = f"{label}_mc_rel_se"
        rel_values = pd.to_numeric(out.get(rel_key, np.nan), errors="coerce")
        out[f"{label}_mc_error_pct"] = 100.0 * rel_values
        skipped = out.get("secondary_audit_skipped", pd.Series(True, index=out.index)).fillna(True).astype(bool)
        support_ok = out.get(f"b_tail_{label}_pass", pd.Series(False, index=out.index)).fillna(False).astype(bool)
        out[f"{label}_needs_secondary_audit"] = (
            skipped & support_ok & np.isfinite(rel_values) & (rel_values <= conv.well_converged_rel_se)
            & (out[f"{label}_class_code"] < 4)
        )
    if len(out):
        class_cols = [f"{label}_class_code" for label in ("phonon", "ge1", "exact_M")]
        out["overall_class_code"] = out[class_cols].min(axis=1).astype(int)
        out["overall_classification"] = out["overall_class_code"].map(RESULT_CLASS_LABELS)
    else:
        out["overall_class_code"] = pd.Series(dtype=int)
        out["overall_classification"] = pd.Series(dtype=str)
    return out

def _audit_point_timed(ordinal: int, total: int, m_dm: float, eps: float, cfg: ScanConfig, ground: CopperGroundPlane, conv: ConvergenceConfig) -> tuple[int, float, float, dict, float]:
    """In-process worker wrapper retained for the optional thread backend."""
    t0 = time.perf_counter()
    audited = convergence_audit_point(
        m_dm, eps, cfg, ground, conv, seed=cfg.seed, progress=False
    )
    return ordinal, m_dm, eps, audited, time.perf_counter() - t0


def _scanconfig_from_json_dict(data: dict) -> ScanConfig:
    payload = dict(data)
    pop = payload.get("prx_population")
    if isinstance(pop, dict):
        payload["prx_population"] = PRXPopulationConfig(**pop)
    return ScanConfig(**payload)


def _failed_audit_row(m_dm: float, eps: float, reason: str, *, wallclock_timeout: bool = False) -> dict:
    metrics = mathieu_metrics(float(m_dm), float(eps), ScanConfig())
    row = {
        "m_dm_kg": float(m_dm),
        "eps": float(eps),
        "mathieu_stable": bool(metrics.get("mathieu_stable", True)),
        "pseudopotential_valid": True,
        "q_max": float(metrics.get("q_max", np.nan)),
        "secular_over_rf": float(metrics.get("secular_over_rf", np.nan)),
        "reference_class": int(metrics.get("reference_class", CLASS_PSEUDO_VALID)),
        "effective_samples": 0,
        "adaptive_stop_reason": reason,
        "phonon_rate_s": np.nan,
        "event_rate_ge1_s": np.nan,
        "event_rate_exact_M_s": np.nan,
        "phonon_mc_se_s": np.nan,
        "ge1_mc_se_s": np.nan,
        "exact_M_mc_se_s": np.nan,
        "phonon_mc_rel_se": np.nan,
        "ge1_mc_rel_se": np.nan,
        "exact_M_mc_rel_se": np.nan,
        "mc_rel_se": np.nan,
        "event_mc_rel_se": np.nan,
        "event_exact_M_mc_rel_se": np.nan,
        "mean_nfev": np.nan,
        "timeout_fraction": 1.0 if wallclock_timeout else np.nan,
        "analytic_shortcut_fraction": np.nan,
        "timeout_pass": False,
        "secondary_audit_skipped": True,
        "b_tail_clipped": False,
        "wallclock_timeout": bool(wallclock_timeout),
        "worker_failure": not bool(wallclock_timeout),
        "reliability_failures": reason,
    }
    for label in ("phonon", "ge1", "exact_M"):
        row[f"mc_{label}_pass"] = False
        row[f"b_tail_{label}_pass"] = False
        row[f"numerical_{label}_pass"] = False
        row[f"reliable_{label}"] = False
        row[f"b_tail_{label}_fraction_upper"] = np.nan
    row["reliable_all"] = False
    return row


def _last_nonempty_line(path: Path) -> str:
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 8192), os.SEEK_SET)
            text = fh.read().decode("utf-8", errors="replace")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else ""
    except Exception:
        return ""


def _row_point_key(row) -> tuple[float, float] | None:
    """Stable log-space key for one positive (mass, epsilon) parameter pair."""
    try:
        m_dm = float(row.get("m_dm_kg", np.nan))
        eps = float(row.get("eps", np.nan))
    except (AttributeError, TypeError, ValueError):
        return None
    if not (np.isfinite(m_dm) and np.isfinite(eps) and m_dm > 0.0 and eps > 0.0):
        return None
    return (round(math.log10(m_dm), 9), round(math.log10(eps), 9))


def _has_finite_central_estimate(row) -> bool:
    """True when all three central observables were actually evaluated.

    Zero is a valid finite estimate.  This deliberately does not require the
    secondary convergence audits to have passed; those audits determine the
    quality class, whereas resume completeness for the rate heatmaps is based
    on whether a central estimate exists.
    """
    keys = ("phonon_rate_s", "event_rate_ge1_s", "event_rate_exact_M_s")
    try:
        return all(np.isfinite(float(row.get(k, np.nan))) for k in keys)
    except (AttributeError, TypeError, ValueError):
        return False


def _has_current_adaptive_bmax_audit(row, conv: ConvergenceConfig) -> bool:
    try:
        version = int(float(row.get("adaptive_bmax_version", 0) or 0))
    except Exception:
        version = 0
    complete = _safe_bool(row.get("adaptive_bmax_complete", False), False) if hasattr(row, "get") else False
    return bool(version == int(conv.adaptive_bmax_version) and complete)


def _can_metadata_migrate_support(row, conv: ConvergenceConfig) -> bool:
    """Old converged support audits remain valid without new trajectories."""
    if not hasattr(row, "get") or not _has_finite_central_estimate(row):
        return False
    try: old_version = int(float(row.get("adaptive_bmax_version", 0) or 0))
    except Exception: old_version = 0
    if old_version <= 0 or old_version >= int(conv.adaptive_bmax_version):
        return False
    if not _safe_bool(row.get("adaptive_bmax_complete", False), False):
        return False
    if str(row.get("adaptive_bmax_status", "")) != "converged":
        return False
    return all(_safe_bool(row.get(f"b_tail_{lab}_pass", False), False) for lab in ("phonon","ge1","exact_M"))


def _metadata_migrate_support(row: dict, conv: ConvergenceConfig) -> dict:
    out = dict(row)
    out["adaptive_bmax_version"] = int(conv.adaptive_bmax_version)
    out["adaptive_bmax_reused_previous"] = True
    out["support_converged"] = True
    out["support_unresolved"] = False
    out["support_unresolved_reason"] = ""
    out["support_migration_only"] = True
    return out

def _checkpoint_row_score(row, order: int) -> tuple:
    """Rank duplicate checkpoint rows, preferring useful/newer evaluations."""
    finite_rates = sum(
        np.isfinite(float(row.get(k, np.nan)))
        for k in ("phonon_rate_s", "event_rate_ge1_s", "event_rate_exact_M_s")
    )
    full_central = int(finite_rates == 3)
    nonfatal = int(not _safe_bool(row.get("worker_failure", False), False) and not _safe_bool(row.get("wallclock_timeout", False), False))
    secondary_done = int(not _safe_bool(row.get("secondary_audit_skipped", True), True))
    try:
        n = int(float(row.get("effective_samples", 0) or 0))
    except (TypeError, ValueError):
        n = 0
    # Row order is the final tie-breaker, so a successful later rerun replaces
    # an older copy without letting a later empty timeout beat a finite result.
    return (full_central, finite_rates, nonfatal, secondary_done, n, int(order))


def _canonicalize_checkpoint(old: pd.DataFrame, valid: pd.DataFrame) -> tuple[list[dict], dict]:
    """Keep exactly one best row for each point on the CURRENT valid grid.

    Stale rows from older grids are ignored. Duplicate rows are collapsed.
    The returned diagnostics make resume behavior explicit instead of treating
    a raw CSV row count as a count of completed current-grid points.
    """
    valid_records = valid.to_dict("records")
    current_keys = {_row_point_key(r) for r in valid_records}
    current_keys.discard(None)
    best: dict[tuple[float, float], tuple[tuple, dict]] = {}
    stale = 0
    malformed = 0
    for order, row in enumerate(old.to_dict("records")):
        key = _row_point_key(row)
        if key is None:
            malformed += 1
            continue
        if key not in current_keys:
            stale += 1
            continue
        score = _checkpoint_row_score(row, order)
        if key not in best or score > best[key][0]:
            best[key] = (score, dict(row))
    rows = [best[k][1] for k in best]
    finite_central = sum(_has_finite_central_estimate(r) for r in rows)
    diagnostics = {
        "checkpoint_rows": int(len(old)),
        "current_valid_points": int(len(valid)),
        "matched_unique_points": int(len(best)),
        "finite_central_points": int(finite_central),
        "duplicate_current_rows": int(max(0, len(old) - stale - malformed - len(best))),
        "stale_rows": int(stale),
        "malformed_rows": int(malformed),
    }
    return rows, diagnostics


def _incomplete_audit_from_baseline(
    baseline: dict, reason: str, *, wallclock_timeout: bool = False, worker_failure: bool = False
) -> dict:
    """Preserve a finished central estimate when a later audit is interrupted."""
    row = dict(baseline)
    row["secondary_audit_skipped"] = True
    row["wallclock_timeout"] = bool(wallclock_timeout)
    row["worker_failure"] = bool(worker_failure)
    row["timeout_pass"] = bool(
        np.isfinite(float(row.get("timeout_fraction", np.nan)))
        and float(row.get("timeout_fraction", np.inf)) <= 0.01
    )
    # Preserve any support audit that had already completed before the later
    # numerical/worker interruption. Older versions erased these flags, causing
    # a timeout during a numerical audit to throw away valid tail work.
    support_done = _safe_bool(row.get("adaptive_bmax_complete", False), False)
    if not support_done:
        row["b_tail_clipped"] = False
    for label in ("phonon", "ge1", "exact_M"):
        row[f"mc_{label}_pass"] = False
        if not support_done:
            row[f"b_tail_{label}_pass"] = False
            row[f"b_tail_{label}_fraction_upper"] = np.nan
        row[f"numerical_{label}_pass"] = False
        row[f"reliable_{label}"] = False
    row["reliable_all"] = False
    row["reliability_failures"] = str(reason)
    row["audit_incomplete_reason"] = str(reason)
    return row


def _run_worker_job(job_path: str | Path) -> int:
    """Standalone worker entry point for central runs and checkpoint upgrades."""
    job_path = Path(job_path)
    job = json.loads(job_path.read_text())
    result_path = Path(job["result_path"])
    cfg = _scanconfig_from_json_dict(job["cfg"])
    ground = CopperGroundPlane(**job["ground"])
    conv = ConvergenceConfig(**job["conv"])
    ordinal = int(job["ordinal"])
    total = int(job["total"])
    m_dm = float(job["m_dm"])
    eps = float(job["eps"])
    job_mode = str(job.get("job_mode", "central"))
    t0 = time.perf_counter()

    def write_payload(payload: dict) -> None:
        tmp = result_path.with_suffix(result_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, allow_nan=True))
        os.replace(tmp, result_path)

    print(
        f"[worker {os.getpid()}] {job_mode} point {ordinal}/{total}: m={m_dm:.6e}, eps={eps:.6e}",
        flush=True,
    )
    baseline = None
    try:
        if not mathieu_metrics(m_dm, eps, cfg)["pseudopotential_valid"]:
            raise ValueError("Refusing to audit a pseudopotential-invalid point")
        seed0 = cfg.seed
        if job_mode != "central":
            baseline = dict(job["baseline"])
            print("    reusing checkpoint central estimate; no central MC rerun", flush=True)
            partial = _incomplete_audit_from_baseline(
                baseline, "adaptive-bmax-upgrade-not-yet-complete", wallclock_timeout=False, worker_failure=False
            )
            partial["central_estimate_reused"] = True
            partial["worker_elapsed_s"] = float(time.perf_counter() - t0)
            partial["worker_pid"] = int(os.getpid())
            partial["module_version"] = MODULE_VERSION
            partial["point_ordinal"] = ordinal
            write_payload({"ok": False, "partial": True, "phase": "central_reused", "result": partial})
            def support_checkpoint(snapshot):
                snap = dict(snapshot)
                snap["central_estimate_reused"] = True
                snap["worker_elapsed_s"] = float(time.perf_counter() - t0)
                snap["worker_pid"] = int(os.getpid())
                snap["module_version"] = MODULE_VERSION
                snap["point_ordinal"] = ordinal
                snap["job_mode"] = job_mode
                write_payload({"ok": False, "partial": True, "phase": "support_progress", "result": snap})
            if job_mode == "asymptotic_only":
                audited = dict(baseline)
                audited["adaptive_bmax_version"] = int(conv.adaptive_bmax_version)
                audited["adaptive_bmax_status"] = "max_factor_reached"
                audited.update(_asymptotic_tail_diagnostics(audited, conv))
                audited["support_repair_action"] = "asymptotic_only"
                print("    asymptotic annulus analysis from checkpoint history; zero new trajectories", flush=True)
            else:
                audited = upgrade_existing_audit_point(
                    m_dm, eps, cfg, ground, conv, baseline, seed=seed0, progress=True,
                    checkpoint_callback=support_checkpoint,
                )
                audited["support_repair_action"] = job_mode
        else:
            print("    central adaptive estimate", flush=True)
            baseline = central_point_estimate_auto(
                m_dm, eps, cfg, ground, conv, seed=seed0, progress=True
            )
            partial = dict(baseline)
            partial["worker_elapsed_s"] = float(time.perf_counter() - t0)
            partial["worker_pid"] = int(os.getpid())
            partial["wallclock_timeout"] = False
            partial["worker_failure"] = False
            partial["module_version"] = MODULE_VERSION
            partial["point_ordinal"] = ordinal
            partial = _incomplete_audit_from_baseline(
                partial, "secondary-audit-not-yet-complete", wallclock_timeout=False, worker_failure=False
            )
            write_payload({"ok": False, "partial": True, "phase": "central_complete", "result": partial})
            audited = _complete_convergence_audit_from_baseline(
                m_dm, eps, cfg, ground, conv, baseline, seed0=seed0, progress=True
            )

        audited["worker_elapsed_s"] = float(time.perf_counter() - t0)
        audited["worker_pid"] = int(os.getpid())
        audited["wallclock_timeout"] = False
        audited["worker_failure"] = False
        audited["module_version"] = MODULE_VERSION
        audited["point_ordinal"] = ordinal
        audited["job_mode"] = job_mode
        write_payload({"ok": True, "phase": "complete", "result": audited})
        return 0
    except Exception as exc:
        traceback.print_exc()
        if baseline is not None and _has_finite_central_estimate(baseline):
            partial = _incomplete_audit_from_baseline(
                baseline, f"worker-exception-after-central:{type(exc).__name__}: {exc}",
                worker_failure=True,
            )
            partial["central_estimate_reused"] = bool(job_mode != "central")
            partial["worker_elapsed_s"] = float(time.perf_counter() - t0)
            partial["worker_pid"] = int(os.getpid())
            partial["module_version"] = MODULE_VERSION
            partial["point_ordinal"] = ordinal
            partial["job_mode"] = job_mode
            payload = {
                "ok": False, "partial": True, "phase": "secondary_failed",
                "result": partial, "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "elapsed_s": float(time.perf_counter() - t0),
            }
        else:
            payload = {
                "ok": False, "partial": False, "phase": "central_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "elapsed_s": float(time.perf_counter() - t0),
            }
        write_payload(payload)
        return 2



def _numerical_audit_is_current(row: dict, conv: ConvergenceConfig) -> bool:
    if not _has_complete_numerical_audit(row):
        return False
    try:
        sf = float(row.get("adaptive_bmax_factor", 1.0) or 1.0)
        nf = float(row.get("numerical_audit_b_support_factor", np.nan))
    except Exception:
        return False
    return bool(np.isfinite(nf) and math.isclose(sf, nf, rel_tol=1e-12, abs_tol=1e-15)
                and not _safe_bool(row.get("secondary_audit_skipped", True), True))


def _needs_numerical_upgrade(row: dict, conv: ConvergenceConfig) -> bool:
    if not all(_safe_bool(row.get(f"b_tail_{lab}_pass", False), False) for lab in ("phonon", "ge1", "exact_M")):
        return False
    candidates = []
    for lab in ("phonon", "ge1", "exact_M"):
        try: rel = float(row.get(f"{lab}_mc_rel_se", np.nan))
        except Exception: rel = np.nan
        candidates.append(np.isfinite(rel) and rel <= conv.well_converged_rel_se)
    return bool(any(candidates) and not _numerical_audit_is_current(row, conv))


def _support_repair_mode(row: dict, conv: ConvergenceConfig) -> str | None:
    """Choose the cheapest next action for a finite checkpoint row."""
    status = str(row.get("adaptive_bmax_status", ""))
    complete = _safe_bool(row.get("adaptive_bmax_complete", False), False)
    if status == "support_unresolved_max_factor":
        status = "max_factor_reached"
    if complete and status == "converged":
        return "numerical_upgrade" if _needs_numerical_upgrade(row, conv) else None
    if complete and status == "tail_noise_limited":
        # Upgrade legacy 1024-sample tails to the new ceiling, but do not
        # resubmit a current v3 row that has already exhausted that ceiling.
        try: n_tail = min(int(float(row.get("b_tail1_n", 0) or 0)), int(float(row.get("b_tail2_n", 0) or 0)))
        except Exception: n_tail = 0
        return "tail_refine" if n_tail < int(conv.adaptive_bmax_tail_max_samples) else None
    if complete and status == "max_factor_reached":
        if int(float(row.get("asymptotic_tail_analysis_version", 0) or 0)) >= 1:
            return None
        # If a lower confidence bound is not clearly above tolerance, more tail
        # statistics are still worthwhile. Otherwise use existing annulus history
        # for asymptotic triage with zero new trajectories.
        clear = False
        for lab in ("phonon", "ge1", "exact_M"):
            try: lo = float(row.get(f"b_tail_{lab}_fraction_lower", np.nan))
            except Exception: lo = np.nan
            clear = clear or (np.isfinite(lo) and lo > conv.b_tail_fraction_tol)
        return "asymptotic_only" if clear else "tail_refine"
    if complete and status == "geometry_limited":
        try: used_R = float(row.get("adaptive_bmax_tail_outer_radius_m", 0.0) or 0.0)
        except Exception: used_R = 0.0
        if used_R >= float(conv.adaptive_bmax_outer_radius_max_m) * (1.0 - 1e-12):
            return None
        return "geometry_repair"
    if not complete or status in ("in_progress", ""):
        return "support_resume"
    return "support_upgrade"

def run_converged_valid_grid(
    cfg: ScanConfig,
    ground: CopperGroundPlane | None = None,
    conv: ConvergenceConfig | None = None,
    *,
    progress: bool = True,
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
    max_workers: int = 1,
    parallel_backend: str = "subprocess",
    point_timeout_s: float | None = 1800.0,
    heartbeat_s: float = 30.0,
    poll_s: float = 1.0,
) -> pd.DataFrame:
    """Audit all pseudopotential-valid points with incremental checkpoint reuse.

    Version 2026-08-14.1 is status-aware and reuses prior work aggressively.

    Missing central estimates use the direct/multifidelity production estimator.
    Finite checkpoint rows are scheduled only for their cheapest missing task:
    tail-noise refinement, geometry repair, support resume, asymptotic-only
    analysis, or paired numerical audit. Previously converged support is migrated
    without new trajectories. No finite central estimate is rerun merely because
    the tail-repair algorithm was upgraded.
    """
    ground = ground or CopperGroundPlane()
    conv = conv or ConvergenceConfig()
    valid = valid_parameter_points(cfg)
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    rows: list[dict] = []
    resume_diag = {
        "checkpoint_rows": 0, "current_valid_points": len(valid),
        "matched_unique_points": 0, "finite_central_points": 0,
        "duplicate_current_rows": 0, "stale_rows": 0, "malformed_rows": 0,
    }
    if resume and checkpoint is not None and checkpoint.exists():
        try:
            old = pd.read_csv(checkpoint)
            if {"m_dm_kg", "eps"}.issubset(old.columns):
                rows, resume_diag = _canonicalize_checkpoint(old, valid)
                if progress:
                    print(f"Checkpoint: {checkpoint}", flush=True)
                    print(
                        f"  raw rows={resume_diag['checkpoint_rows']}; "
                        f"matched unique current-grid points={resume_diag['matched_unique_points']}/{len(valid)}; "
                        f"finite central estimates={resume_diag['finite_central_points']}/{len(valid)}",
                        flush=True,
                    )
                    print(
                        f"  duplicates collapsed={resume_diag['duplicate_current_rows']}; "
                        f"stale rows ignored={resume_diag['stale_rows']}; "
                        f"malformed rows ignored={resume_diag['malformed_rows']}",
                        flush=True,
                    )
        except Exception as exc:
            warnings.warn(f"Could not resume checkpoint {checkpoint}: {exc}")
            rows = []

    by_key = {_row_point_key(r): dict(r) for r in rows if _row_point_key(r) is not None}
    total = len(valid)
    pending = []
    mode_counts = {}
    n_migrated = 0
    for ordinal, rec in enumerate(valid.to_dict("records"), 1):
        m_dm = float(rec["m_dm_kg"]); eps = float(rec["eps"])
        key = _row_point_key(rec)
        existing = by_key.get(key)
        if existing is None or not _has_finite_central_estimate(existing):
            mode = "central"; baseline = None
        else:
            baseline = existing
            if conv.adaptive_bmax_enabled and not _has_current_adaptive_bmax_audit(existing, conv) and _can_metadata_migrate_support(existing, conv):
                migrated = _metadata_migrate_support(existing, conv)
                migrated.update(_asymptotic_tail_diagnostics(migrated, conv))
                n_migrated += 1
                by_key[key] = migrated; baseline = migrated
                for i, rr in enumerate(rows):
                    if _row_point_key(rr) == key:
                        rows[i] = migrated; break
            mode = _support_repair_mode(baseline, conv) if conv.adaptive_bmax_enabled else ("numerical_upgrade" if _needs_numerical_upgrade(baseline, conv) else None)
        if mode is not None:
            pending.append((ordinal, m_dm, eps, key, mode, baseline))
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

    workers = max(1, int(max_workers))
    start = time.perf_counter()
    if progress:
        mode_text = ", ".join(f"{k}={v}" for k, v in sorted(mode_counts.items())) or "none"
        print(
            f"Pending work: {len(pending)} of {total} valid points ({mode_text}); "
            f"workers={workers}; backend={parallel_backend}; point_timeout={point_timeout_s}s; "
            f"segment_timeout={cfg.ode_segment_walltime_s}s.", flush=True,
        )
        if n_migrated:
            print(f"  Metadata-migrated {n_migrated} previously converged support audits with zero new trajectories.", flush=True)
        if any(k != "central" for k in mode_counts):
            print(
                "  Repair jobs reuse finite central estimates and prior adaptive-b history; only the missing "
                "tail / geometry / numerical work is performed.", flush=True,
            )

    def upsert_row(row: dict, key) -> None:
        for i, existing in enumerate(rows):
            if _row_point_key(existing) == key:
                if _has_finite_central_estimate(row) or not _has_finite_central_estimate(existing):
                    rows[i] = row
                return
        rows.append(row)

    def save_checkpoint() -> None:
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(checkpoint, index=False)

    def run_item(item, *, show_progress: bool):
        ordinal, m_dm, eps, key, mode, baseline = item
        t0 = time.perf_counter()
        if mode == "central":
            audited = convergence_audit_point(
                m_dm, eps, cfg, ground, conv, seed=cfg.seed, progress=show_progress
            )
        elif mode == "asymptotic_only":
            audited = dict(baseline)
            # Promote the metadata version because v3's max-factor action is a
            # zero-trajectory diagnostic using already measured octave history.
            audited["adaptive_bmax_version"] = int(conv.adaptive_bmax_version)
            audited["adaptive_bmax_status"] = "max_factor_reached"
            audited.update(_asymptotic_tail_diagnostics(audited, conv))
            audited["support_repair_action"] = "asymptotic_only"
        else:
            audited = upgrade_existing_audit_point(
                m_dm, eps, cfg, ground, conv, baseline,
                seed=cfg.seed, progress=show_progress,
            )
            audited["support_repair_action"] = mode
        audited["worker_elapsed_s"] = float(time.perf_counter() - t0)
        audited["wallclock_timeout"] = False
        audited["worker_failure"] = False
        audited["module_version"] = MODULE_VERSION
        audited["point_ordinal"] = ordinal
        audited["job_mode"] = mode
        return audited

    if workers == 1:
        for item in pending:
            ordinal, m_dm, eps, key, mode, baseline = item
            if progress:
                verb = f"{mode} for" if mode != "central" else "simulate"
                print(
                    f"\n[converged staged] {verb} valid point {ordinal}/{total}: "
                    f"m={m_dm:.6e} kg, eps={eps:.6e}", flush=True,
                )
            try:
                audited = run_item(item, show_progress=progress)
            except Exception as exc:
                if mode != "central" and baseline is not None:
                    audited = _incomplete_audit_from_baseline(
                        baseline, f"support-upgrade-failure:{type(exc).__name__}: {exc}", worker_failure=True
                    )
                else:
                    audited = _failed_audit_row(m_dm, eps, f"serial-worker-failure:{type(exc).__name__}")
                audited["module_version"] = MODULE_VERSION
                audited["point_ordinal"] = ordinal
                audited["job_mode"] = mode
            upsert_row(audited, key); save_checkpoint()
            if progress:
                print(
                    f"    done | mode={mode} | b support={audited.get('adaptive_bmax_factor', np.nan)} B | "
                    f"reliable phonon={audited.get('reliable_phonon', False)} "
                    f"Gamma>=1={audited.get('reliable_ge1', False)} "
                    f"Gamma_M={audited.get('reliable_exact_M', False)}", flush=True,
                )
    elif parallel_backend.lower() == "thread":
        warnings.warn(
            "Thread backend cannot hard-kill a stuck solve_ivp call. "
            "Use parallel_backend='subprocess' for production runs."
        )
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mcp-point") as pool:
            futures = {pool.submit(run_item, item, show_progress=False): item for item in pending}
            done_now = 0
            for future in as_completed(futures):
                item = futures[future]
                ordinal, m_dm, eps, key, mode, baseline = item
                try:
                    audited = future.result()
                except Exception as exc:
                    if mode != "central" and baseline is not None:
                        audited = _incomplete_audit_from_baseline(
                            baseline, f"thread-support-upgrade-failure:{type(exc).__name__}: {exc}", worker_failure=True
                        )
                    else:
                        audited = _failed_audit_row(m_dm, eps, f"thread-worker-failure:{type(exc).__name__}")
                    audited["module_version"] = MODULE_VERSION
                    audited["point_ordinal"] = ordinal
                    audited["job_mode"] = mode
                upsert_row(audited, key); done_now += 1; save_checkpoint()
                if progress:
                    print(
                        f"[thread done {done_now}/{len(pending)}] {mode} point {ordinal}/{total} | "
                        f"b support={audited.get('adaptive_bmax_factor', np.nan)} B",
                        flush=True,
                    )
    elif parallel_backend.lower() == "subprocess":
        module_path = Path(__file__).resolve()
        work_root = (checkpoint.parent if checkpoint is not None else module_path.parent / "outputs") / ".mcp_point_workers"
        work_root.mkdir(parents=True, exist_ok=True)
        queue = deque(pending)
        active: dict[int, dict] = {}
        done_now = 0
        last_heartbeat = 0.0

        def launch(item):
            ordinal, m_dm, eps, key, mode, baseline = item
            stem = f"point_{ordinal:03d}_{mode}_{os.getpid()}_{int(time.time()*1000)}"
            job_path = work_root / f"{stem}.job.json"
            result_path = work_root / f"{stem}.result.json"
            log_path = work_root / f"{stem}.log"
            job = {
                "ordinal": ordinal, "total": total, "m_dm": m_dm, "eps": eps,
                "cfg": asdict(cfg), "ground": asdict(ground), "conv": asdict(conv),
                "result_path": str(result_path), "job_mode": mode,
            }
            if mode != "central":
                job["baseline"] = baseline
            job_path.write_text(json.dumps(job, allow_nan=True))
            log_fh = log_path.open("wb")
            env = os.environ.copy()
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[name] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(
                [sys.executable, "-u", str(module_path), "--worker-job", str(job_path)],
                cwd=str(module_path.parent), stdout=log_fh, stderr=subprocess.STDOUT,
                env=env,
            )
            active[proc.pid] = {
                "proc": proc, "ordinal": ordinal, "m_dm": m_dm, "eps": eps, "key": key,
                "mode": mode, "baseline": baseline,
                "start": time.perf_counter(), "job_path": job_path, "result_path": result_path,
                "log_path": log_path, "log_fh": log_fh,
            }
            if progress:
                print(
                    f"  started {mode} point {ordinal}/{total} [pid {proc.pid}]: "
                    f"m={m_dm:.3e}, eps={eps:.3e}", flush=True,
                )

        while queue or active:
            while queue and len(active) < workers:
                launch(queue.popleft())
            time.sleep(max(float(poll_s), 0.05))
            now = time.perf_counter()
            for pid, meta in list(active.items()):
                proc = meta["proc"]
                elapsed = now - meta["start"]
                timed_out = bool(point_timeout_s is not None and point_timeout_s > 0 and elapsed > float(point_timeout_s))
                rc = proc.poll()
                if rc is None and not timed_out:
                    continue
                if timed_out and rc is None:
                    if progress:
                        print(
                            f"[WATCHDOG] {meta['mode']} point {meta['ordinal']}/{total} exceeded "
                            f"{_format_duration(float(point_timeout_s))}; terminating pid {pid} and continuing.",
                            flush=True,
                        )
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill(); proc.wait(timeout=5)
                    rc = proc.returncode
                meta["log_fh"].close()

                payload = None
                try:
                    if meta["result_path"].exists():
                        payload = json.loads(meta["result_path"].read_text())
                except Exception:
                    payload = None

                if timed_out:
                    preserved = None
                    if payload and payload.get("result") and _has_finite_central_estimate(payload["result"]):
                        preserved = payload["result"]
                    elif meta["mode"] != "central" and meta["baseline"] is not None:
                        preserved = meta["baseline"]
                    if preserved is not None:
                        audited = _incomplete_audit_from_baseline(
                            preserved,
                            "point-wallclock-timeout-during-support-audit" if meta["mode"] != "central" else "point-wallclock-timeout-after-central",
                            wallclock_timeout=True, worker_failure=False,
                        )
                        audited["central_estimate_recovered"] = True
                    else:
                        audited = _failed_audit_row(
                            meta["m_dm"], meta["eps"], "point-wallclock-timeout-before-central", wallclock_timeout=True
                        )
                        audited["central_estimate_recovered"] = False
                    audited["worker_elapsed_s"] = float(elapsed)
                    audited["worker_pid"] = int(pid)
                    audited["module_version"] = MODULE_VERSION
                    audited["point_ordinal"] = int(meta["ordinal"])
                    audited["job_mode"] = meta["mode"]
                    audited["worker_log"] = str(meta["log_path"])
                else:
                    if payload and payload.get("ok"):
                        audited = payload["result"]
                        audited["central_estimate_recovered"] = False
                        audited["worker_log"] = str(meta["log_path"])
                    elif payload and payload.get("result") and _has_finite_central_estimate(payload["result"]):
                        reason = "worker-failure-after-central"
                        if payload.get("error"):
                            reason += ":" + str(payload["error"])
                        audited = _incomplete_audit_from_baseline(
                            payload["result"], reason, worker_failure=True
                        )
                        audited["central_estimate_recovered"] = True
                        audited["worker_elapsed_s"] = float(elapsed)
                        audited["worker_pid"] = int(pid)
                        audited["module_version"] = MODULE_VERSION
                        audited["point_ordinal"] = int(meta["ordinal"])
                        audited["job_mode"] = meta["mode"]
                        audited["worker_log"] = str(meta["log_path"])
                    else:
                        reason = "worker-failure"
                        if payload and payload.get("error"):
                            reason += ":" + str(payload["error"])
                        elif rc is not None:
                            reason += f":returncode={rc}"
                        if meta["mode"] != "central" and meta["baseline"] is not None:
                            audited = _incomplete_audit_from_baseline(meta["baseline"], reason, worker_failure=True)
                        else:
                            audited = _failed_audit_row(meta["m_dm"], meta["eps"], reason)
                        audited["central_estimate_recovered"] = bool(meta["mode"] != "central")
                        audited["worker_elapsed_s"] = float(elapsed)
                        audited["worker_pid"] = int(pid)
                        audited["module_version"] = MODULE_VERSION
                        audited["point_ordinal"] = int(meta["ordinal"])
                        audited["job_mode"] = meta["mode"]
                        audited["worker_log"] = str(meta["log_path"])

                upsert_row(audited, meta["key"])
                done_now += 1
                save_checkpoint()
                if progress:
                    print(
                        f"[process done {done_now}/{len(pending)}] {meta['mode']} point {meta['ordinal']}/{total} "
                        f"in {_format_duration(elapsed)} | b support={audited.get('adaptive_bmax_factor', np.nan)} B | "
                        f"reliable ph={audited.get('reliable_phonon', False)} "
                        f"G>=1={audited.get('reliable_ge1', False)} G_M={audited.get('reliable_exact_M', False)} | "
                        f"timeout={audited.get('wallclock_timeout', False)} failure={audited.get('worker_failure', False)}",
                        flush=True,
                    )
                for pp in (meta["job_path"], meta["result_path"]):
                    try:
                        pp.unlink(missing_ok=True)
                    except Exception:
                        pass
                del active[pid]

            if progress and active and (now - last_heartbeat >= max(float(heartbeat_s), 1.0)):
                last_heartbeat = now
                print(f"[heartbeat] {len(active)} workers running; {len(queue)} jobs queued.", flush=True)
                for meta in sorted(active.values(), key=lambda x: x["ordinal"]):
                    elapsed = now - meta["start"]
                    line = _last_nonempty_line(meta["log_path"])
                    suffix = f" | {line}" if line else ""
                    print(
                        f"    {meta['mode']} point {meta['ordinal']}/{total}: {_format_duration(elapsed)} elapsed{suffix}",
                        flush=True,
                    )
    else:
        raise ValueError("parallel_backend must be 'subprocess' or 'thread'")

    temp = pd.DataFrame(rows)
    canonical_rows, final_diag = _canonicalize_checkpoint(temp, valid) if len(temp) else ([], {})
    by_key = {_row_point_key(r): dict(r) for r in canonical_rows}
    ordered_rows = []
    for ordinal, rec in enumerate(valid.to_dict("records"), 1):
        key = _row_point_key(rec)
        row = by_key.get(key)
        if row is None:
            row = _failed_audit_row(float(rec["m_dm_kg"]), float(rec["eps"]), "missing-current-grid-result")
        row = dict(row)
        row["m_dm_kg"] = float(rec["m_dm_kg"])
        row["eps"] = float(rec["eps"])
        row["point_ordinal"] = int(ordinal)
        ordered_rows.append(row)
    result = pd.DataFrame(ordered_rows)
    result = add_asymptotic_tail_diagnostics(result, conv)
    result = add_result_classifications(result, conv)
    central_complete = int(sum(_has_finite_central_estimate(r) for r in ordered_rows))
    adaptive_complete = int(sum(_has_current_adaptive_bmax_audit(r, conv) for r in ordered_rows))
    if checkpoint is not None:
        result.to_csv(checkpoint, index=False)
    if progress:
        print(
            f"\nCurrent-grid table: {len(result)}/{total} unique valid points; "
            f"finite central estimates={central_complete}/{total}; "
            f"current adaptive-b audits={adaptive_complete}/{total}; "
            f"session runtime={_format_duration(time.perf_counter()-start)}.", flush=True,
        )
    return result


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker-job":
        raise SystemExit(_run_worker_job(sys.argv[2]))

