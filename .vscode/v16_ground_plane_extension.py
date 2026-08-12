"""Ground-structure extension for the single-point v16 dark-matter notebook.

The module does not replace Tests 0--2.5.  It consumes the Test-1.5 weighted
representatives and evaluates each representative with a branch-aware,
full-potential trajectory state machine.  Outside R_switch the DM follows the
full v16 DC+pseudopotential force while the ion evolves freely in its harmonic
trap.  Inside R_switch the ion and DM are propagated as a coupled system.  A
finite axis-aligned metal prism can reflect, promptly transmit, or thermalize
and re-emit a positive MCP.  Repeated ground encounters and repeated returns to
R_switch are supported up to user-set caps.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Iterable
import json
import math
import time

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

PI = math.pi
KB = 1.380649e-23
HBAR = 1.054571817e-34
EV = 1.602176634e-19
ME = 9.1093837015e-31
AMU = 1.66053906660e-27
EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19


@dataclass(frozen=True)
class MetalPrism:
    name: str = "upper_copper_ground"
    x_min_m: float = -17.5e-3
    x_max_m: float = 17.5e-3
    y_min_m: float = -17.5e-3
    y_max_m: float = 17.5e-3
    z_min_m: float = 5.0e-3
    z_max_m: float = 6.0e-3
    material: str = "copper"
    double_layer_eV: float = 3.19
    density_kg_m3: float = 8960.0
    atomic_mass_u: float = 63.546
    atomic_number: int = 29
    temperature_K: float = 300.0
    enable_delayed_reemission: bool = True

    @property
    def lo(self) -> np.ndarray:
        return np.array([self.x_min_m, self.y_min_m, self.z_min_m], dtype=float)

    @property
    def hi(self) -> np.ndarray:
        return np.array([self.x_max_m, self.y_max_m, self.z_max_m], dtype=float)

    @property
    def atom_density_m3(self) -> float:
        return self.density_kg_m3 / (self.atomic_mass_u * AMU)


@dataclass(frozen=True)
class GroundRateConfig:
    density_m3: float = 1.0e9
    target_mode: int = 2
    exact_phonon_number: int = 3
    branch_replicas: int = 8
    max_rows: int | None = None
    random_seed: int = 20260806
    outer_radius_m: float = 40.0e-3
    max_ground_interactions: int = 6
    max_core_passes: int = 4
    n_threads: int = 1
    outside_rtol: float = 1.0e-8
    outside_atol: float = 1.0e-11
    outside_max_step_s: float = 5.0e-7
    outside_time_factor: float = 8.0
    outside_min_time_s: float = 100.0e-6
    core_rtol: float = 1.0e-9
    core_atol: float = 1.0e-12
    core_max_step_s: float = 1.0e-8
    core_time_factor: float = 12.0
    core_min_time_s: float = 20.0e-6
    full_escape_factor: float = 1.25
    coulomb_softening_m: float = 1.0e-12
    surface_nudge_m: float = 5.0e-10
    metal_transport_steps: int = 96
    unresolved_policy: str = "zero"  # zero or nan
    save_trajectory_rows: bool = True


@dataclass
class V16Context:
    m_dm_kg: float
    eps: float
    m_ion_kg: float
    ion_charge_number: float
    coulomb_constant: float
    elementary_charge_c: float
    omega_modes_rad_s: np.ndarray
    trap_force: Callable[[np.ndarray, float, float], np.ndarray]
    invalid_z_floor_m: float | None = None


def context_from_notebook(ns: dict[str, Any]) -> V16Context:
    required = ["m_dm", "eps", "m_ion", "Z_ion", "K", "e", "omega_vec", "trap_force"]
    missing = [name for name in required if name not in ns]
    if missing:
        raise RuntimeError("Run the original v16 potential and Test-2 definition cells first. Missing: " + ", ".join(missing))
    invalid = None
    resolver = ns.get("resolve_invalid_z_floor")
    if callable(resolver):
        invalid = resolver()
    return V16Context(
        m_dm_kg=float(ns["m_dm"]),
        eps=float(ns["eps"]),
        m_ion_kg=float(ns["m_ion"]),
        ion_charge_number=float(ns["Z_ion"]),
        coulomb_constant=float(ns["K"]),
        elementary_charge_c=float(ns["e"]),
        omega_modes_rad_s=np.asarray(ns["omega_vec"], dtype=float).reshape(3),
        trap_force=ns["trap_force"],
        invalid_z_floor_m=invalid,
    )


def direction_basis(theta: float, alpha: float, psi: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact v16 convention."""
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
    u /= max(np.linalg.norm(u), np.finfo(float).tiny)
    b_hat /= max(np.linalg.norm(b_hat), np.finfo(float).tiny)
    return u, b_hat


def initial_state_v16(b_m: float, radius_m: float, theta: float, alpha: float, psi: float, speed_m_s: float) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 <= b_m < radius_m):
        raise ValueError(f"Require 0 <= b < R; got b={b_m:.6e}, R={radius_m:.6e}")
    u, b_hat = direction_basis(theta, alpha, psi)
    s = math.sqrt(max(radius_m * radius_m - b_m * b_m, 0.0))
    return b_m * b_hat - s * u, speed_m_s * u


def box_sdf(point: np.ndarray, prism: MetalPrism) -> float:
    p = np.asarray(point, dtype=float)
    center = 0.5 * (prism.lo + prism.hi)
    half = 0.5 * (prism.hi - prism.lo)
    q = np.abs(p - center) - half
    outside = np.linalg.norm(np.maximum(q, 0.0))
    inside = min(float(np.max(q)), 0.0)
    return float(outside + inside)


def face_normal(point: np.ndarray, prism: MetalPrism) -> np.ndarray:
    p = np.asarray(point, dtype=float)
    distances = np.concatenate([np.abs(p - prism.lo), np.abs(p - prism.hi)])
    index = int(np.argmin(distances))
    normal = np.zeros(3, dtype=float)
    if index < 3:
        normal[index] = -1.0
    else:
        normal[index - 3] = 1.0
    return normal


def ray_box_segment(origin: np.ndarray, direction_unit: np.ndarray, prism: MetalPrism) -> tuple[float, float] | None:
    o = np.asarray(origin, dtype=float)
    d = np.asarray(direction_unit, dtype=float)
    t_min, t_max = -np.inf, np.inf
    for axis, (lo, hi) in enumerate(zip(prism.lo, prism.hi)):
        if abs(d[axis]) < 1.0e-18:
            if o[axis] < lo or o[axis] > hi:
                return None
            continue
        a = (lo - o[axis]) / d[axis]
        b = (hi - o[axis]) / d[axis]
        if a > b:
            a, b = b, a
        t_min = max(t_min, a)
        t_max = min(t_max, b)
        if t_max < t_min:
            return None
    if t_max < 0.0:
        return None
    return max(float(t_min), 0.0), float(t_max)


def _screened_transport(m_dm: float, eps: float, kinetic_j: float, metal: MetalPrism, k_coulomb: float) -> dict[str, float]:
    """Analytic clean-metal baseline for transport mean free path and stopping.

    This is an effective model, not a measured positive-MCP coefficient.
    """
    energy = max(float(kinetic_j), 1.0e-40)
    speed = math.sqrt(2.0 * energy / m_dm)
    n_atom = metal.atom_density_m3
    m_nucleus = metal.atomic_mass_u * AMU
    mu = m_dm * m_nucleus / (m_dm + m_nucleus)
    z = float(metal.atomic_number)
    a_screen = 0.88534 * 5.29177210903e-11 / max(z ** (1.0 / 3.0), 1.0)
    kappa = 1.0 / a_screen
    k_wave = mu * speed / HBAR
    coupling = k_coulomb * abs(eps) * z * E_CHARGE**2
    pref = 4.0 * PI * (mu * coupling / HBAR**2) ** 2
    denom = max(kappa * kappa * (kappa * kappa + 4.0 * k_wave * k_wave), 1.0e-300)
    sigma_total = pref / denom
    x = 4.0 * k_wave * k_wave / max(kappa * kappa, 1.0e-300)
    transport_ratio = 2.0 * ((1.0 + x) * math.log1p(x) - x) / max(x * x, 1.0e-300)
    sigma_tr = max(sigma_total * transport_ratio, 1.0e-300)
    lambda_tr = 1.0 / max(n_atom * sigma_tr, 1.0e-300)
    max_transfer_fraction = 4.0 * m_dm * m_nucleus / (m_dm + m_nucleus) ** 2
    nuclear_stopping = n_atom * sigma_tr * max_transfer_fraction * energy

    n_e = metal.atomic_number * n_atom
    omega_p = math.sqrt(n_e * E_CHARGE**2 / (ME * EPS0))
    v_reg = math.sqrt(speed * speed + (HBAR * omega_p / ME) ** 2)
    log_l = max(math.log1p((2.0 * ME * v_reg * v_reg / max(HBAR * omega_p, 1.0e-40)) ** 2), 0.0)
    electronic_stopping = 4.0 * PI * n_e * (k_coulomb * abs(eps) * E_CHARGE**2) ** 2 * log_l / max(ME * v_reg * v_reg, 1.0e-300)
    return {
        "speed_m_s": speed,
        "lambda_tr_m": lambda_tr,
        "S_n_J_m": nuclear_stopping,
        "S_e_J_m": electronic_stopping,
        "S_total_J_m": nuclear_stopping + electronic_stopping,
    }


def _integrate_metal_transport(m_dm: float, eps: float, energy_in_j: float, length_m: float, metal: MetalPrism, k_coulomb: float, steps: int) -> dict[str, float]:
    length = max(float(length_m), 0.0)
    if length == 0.0:
        return {"energy_out_j": energy_in_j, "tau_tr": 0.0, "thermalization_depth_m": 0.0, "lambda_thermal_m": np.nan}
    n = max(int(steps), 8)
    dx = length / n
    energy = max(float(energy_in_j), 0.0)
    tau = 0.0
    thermal_energy = 1.5 * KB * metal.temperature_K
    thermal_depth = length
    lambda_thermal = np.nan
    for index in range(n):
        if energy <= thermal_energy:
            thermal_depth = min(thermal_depth, index * dx)
            tr = _screened_transport(m_dm, eps, max(thermal_energy, 1.0e-40), metal, k_coulomb)
            lambda_thermal = tr["lambda_tr_m"]
            energy = max(energy, thermal_energy)
            break
        tr = _screened_transport(m_dm, eps, energy, metal, k_coulomb)
        tau += dx / max(tr["lambda_tr_m"], 1.0e-300)
        energy = max(energy - tr["S_total_J_m"] * dx, 0.0)
    if not np.isfinite(lambda_thermal):
        tr = _screened_transport(m_dm, eps, max(energy, thermal_energy), metal, k_coulomb)
        lambda_thermal = tr["lambda_tr_m"]
    return {"energy_out_j": energy, "tau_tr": tau, "thermalization_depth_m": thermal_depth, "lambda_thermal_m": lambda_thermal}


def _sample_effusive_velocity(normal: np.ndarray, m_dm: float, temperature_k: float, barrier_j: float, rng: np.random.Generator) -> np.ndarray:
    n = np.asarray(normal, dtype=float)
    n /= max(np.linalg.norm(n), np.finfo(float).tiny)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, helper)
    e1 /= max(np.linalg.norm(e1), np.finfo(float).tiny)
    e2 = np.cross(n, e1)
    e_n = -KB * temperature_k * math.log(max(float(rng.random()), 1.0e-15))
    e_t = -KB * temperature_k * math.log(max(float(rng.random()), 1.0e-15))
    phi = 2.0 * PI * float(rng.random())
    v_n = math.sqrt(2.0 * (e_n + barrier_j) / m_dm)
    v_t = math.sqrt(2.0 * e_t / m_dm)
    return v_n * n + v_t * (math.cos(phi) * e1 + math.sin(phi) * e2)


def interact_with_prism(position: np.ndarray, velocity: np.ndarray, context: V16Context, prism: MetalPrism, config: GroundRateConfig, rng: np.random.Generator) -> dict[str, Any]:
    p = np.asarray(position, dtype=float)
    v = np.asarray(velocity, dtype=float)
    n_entry = face_normal(p, prism)
    v_n = float(np.dot(v, n_entry))
    if v_n >= 0.0:
        return {"kind": "grazing_numerical", "position": p + config.surface_nudge_m * n_entry, "velocity": v.copy(), "delay_s": 0.0, "prompt_probability": 0.0}
    barrier = abs(context.eps) * prism.double_layer_eV * EV
    normal_energy = 0.5 * context.m_dm_kg * v_n * v_n
    if normal_energy < barrier:
        reflected = v - 2.0 * v_n * n_entry
        return {"kind": "reflection", "position": p + config.surface_nudge_m * n_entry, "velocity": reflected, "delay_s": 0.0, "prompt_probability": 0.0}

    tangent = v - v_n * n_entry
    v_n_inside = -math.sqrt(max(v_n * v_n - 2.0 * barrier / context.m_dm_kg, 0.0))
    v_inside = tangent + v_n_inside * n_entry
    speed_inside = float(np.linalg.norm(v_inside))
    if speed_inside <= 0.0:
        reflected = v - 2.0 * v_n * n_entry
        return {"kind": "reflection_threshold", "position": p + config.surface_nudge_m * n_entry, "velocity": reflected, "delay_s": 0.0, "prompt_probability": 0.0}
    direction = v_inside / speed_inside
    p_inside = p - config.surface_nudge_m * n_entry
    segment = ray_box_segment(p_inside, direction, prism)
    if segment is None:
        return {"kind": "metal_geometry_unresolved", "position": p + config.surface_nudge_m * n_entry, "velocity": v - 2.0 * v_n * n_entry, "delay_s": 0.0, "prompt_probability": 0.0}
    _, length = segment
    length = max(length, config.surface_nudge_m)
    p_exit = p_inside + length * direction
    n_exit = face_normal(p_exit, prism)
    energy_inside = 0.5 * context.m_dm_kg * speed_inside**2
    transport = _integrate_metal_transport(context.m_dm_kg, context.eps, energy_inside, length, prism, context.coulomb_constant, config.metal_transport_steps)
    p_prompt = math.exp(-min(transport["tau_tr"], 700.0)) if transport["energy_out_j"] > 1.5 * KB * prism.temperature_K else 0.0

    if float(rng.random()) < p_prompt:
        speed_out_inside = math.sqrt(2.0 * transport["energy_out_j"] / context.m_dm_kg)
        v_pre = speed_out_inside * direction
        vn_pre = float(np.dot(v_pre, n_exit))
        tangent_pre = v_pre - vn_pre * n_exit
        vn_post = math.sqrt(max(vn_pre * vn_pre + 2.0 * barrier / context.m_dm_kg, 0.0))
        return {"kind": "prompt_transmission", "position": p_exit + config.surface_nudge_m * n_exit, "velocity": tangent_pre + vn_post * n_exit, "delay_s": length / max(speed_inside, 1.0e-30), "prompt_probability": p_prompt}

    if not prism.enable_delayed_reemission:
        return {"kind": "metal_absorbed_model", "position": p_inside, "velocity": np.zeros(3), "delay_s": np.inf, "prompt_probability": p_prompt}

    x_th = min(max(transport["thermalization_depth_m"], 0.0), length)
    p_forward = min(max(x_th / length, 0.0), 1.0)
    if float(rng.random()) < p_forward:
        exit_point, exit_normal, kind = p_exit, n_exit, "delayed_forward"
    else:
        exit_point, exit_normal, kind = p, n_entry, "delayed_return"
    thermal_speed = math.sqrt(8.0 * KB * prism.temperature_K / (PI * context.m_dm_kg))
    diffusion = max(thermal_speed * transport["lambda_thermal_m"] / 3.0, 1.0e-30)
    delay = x_th * max(length - x_th, 0.0) / max(2.0 * diffusion, 1.0e-30)
    v_emit = _sample_effusive_velocity(exit_normal, context.m_dm_kg, prism.temperature_K, barrier, rng)
    return {"kind": kind, "position": exit_point + config.surface_nudge_m * exit_normal, "velocity": v_emit, "delay_s": delay, "prompt_probability": p_prompt}


def _advance_harmonic(x: np.ndarray, v: np.ndarray, omega: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    if dt <= 0.0:
        return x.copy(), v.copy()
    phase = omega * dt
    c = np.cos(phase)
    s = np.sin(phase)
    x_new = x * c + np.divide(v, omega, out=np.zeros_like(v), where=omega != 0.0) * s
    v_new = v * c - omega * x * s
    return x_new, v_new


def _coulomb_force_on_ion(x_ion: np.ndarray, x_dm: np.ndarray, ctx: V16Context, softening_m: float) -> np.ndarray:
    delta = np.asarray(x_ion) - np.asarray(x_dm)
    r2 = max(float(np.dot(delta, delta)) + softening_m**2, np.finfo(float).tiny)
    return ctx.coulomb_constant * ctx.ion_charge_number * ctx.eps * ctx.elementary_charge_c**2 * delta * r2 ** (-1.5)


def _outside_rhs(ctx: V16Context, config: GroundRateConfig) -> Callable[[float, np.ndarray], np.ndarray]:
    omega = ctx.omega_modes_rad_s
    def rhs(t: float, state: np.ndarray) -> np.ndarray:
        xi, xd = state[0:3], state[3:6]
        vi, vd = state[6:9], state[9:12]
        ai = -(omega**2) * xi
        ad = np.asarray(ctx.trap_force(xd[None, :], ctx.m_dm_kg, ctx.eps), dtype=float)[0] / ctx.m_dm_kg
        force = _coulomb_force_on_ion(xi, xd, ctx, config.coulomb_softening_m)
        phase = omega * t
        return np.concatenate([vi, vd, ai, ad, force * np.cos(phase), force * np.sin(phase)])
    return rhs


def _core_rhs(ctx: V16Context, config: GroundRateConfig) -> Callable[[float, np.ndarray], np.ndarray]:
    omega = ctx.omega_modes_rad_s
    def rhs(t: float, state: np.ndarray) -> np.ndarray:
        xi, xd = state[0:3], state[3:6]
        vi, vd = state[6:9], state[9:12]
        force = _coulomb_force_on_ion(xi, xd, ctx, config.coulomb_softening_m)
        ai = -(omega**2) * xi + force / ctx.m_ion_kg
        ad = np.asarray(ctx.trap_force(xd[None, :], ctx.m_dm_kg, ctx.eps), dtype=float)[0] / ctx.m_dm_kg - force / ctx.m_dm_kg
        phase = omega * t
        return np.concatenate([vi, vd, ai, ad, force * np.cos(phase), force * np.sin(phase)])
    return rhs


def _relative_radial_velocity(state: np.ndarray) -> float:
    r = state[3:6] - state[0:3]
    v = state[9:12] - state[6:9]
    norm = max(float(np.linalg.norm(r)), 1.0e-300)
    return float(np.dot(r, v) / norm)


def _integrate_outside(state: np.ndarray, t0: float, r_switch: float, r_outer: float, ctx: V16Context, prisms: list[MetalPrism], config: GroundRateConfig) -> tuple[str, int | None, np.ndarray, float, Any]:
    speed = max(float(np.linalg.norm(state[9:12])), 1.0e-12)
    t_span = max(config.outside_time_factor * 2.0 * r_outer / speed, config.outside_min_time_s)

    def switch_event(_t: float, y: np.ndarray) -> float:
        return float(np.linalg.norm(y[3:6] - y[0:3]) - r_switch)
    switch_event.terminal = True
    switch_event.direction = -1

    def outer_event(_t: float, y: np.ndarray) -> float:
        radius = float(np.linalg.norm(y[3:6]))
        radial = float(np.dot(y[3:6], y[9:12])) / max(radius, 1.0e-300)
        if radial <= 0.0:
            return -abs(radius - r_outer) - 1.0e-30
        return radius - r_outer
    outer_event.terminal = True
    outer_event.direction = 1

    events: list[Callable] = [switch_event, outer_event]
    for prism in prisms:
        def ground_event(_t: float, y: np.ndarray, prism=prism) -> float:
            return box_sdf(y[3:6], prism)
        ground_event.terminal = True
        ground_event.direction = -1
        events.append(ground_event)
    if ctx.invalid_z_floor_m is not None:
        def invalid_event(_t: float, y: np.ndarray) -> float:
            return float(min(y[2], y[5]) - ctx.invalid_z_floor_m)
        invalid_event.terminal = True
        invalid_event.direction = -1
        events.append(invalid_event)

    sol = solve_ivp(
        _outside_rhs(ctx, config),
        (t0, t0 + t_span),
        state,
        method="DOP853",
        rtol=config.outside_rtol,
        atol=config.outside_atol,
        max_step=config.outside_max_step_s,
        events=events,
    )
    event_index = None
    for index, times in enumerate(sol.t_events):
        if len(times):
            event_index = index
            break
    final_state = np.asarray(sol.y[:, -1], dtype=float)
    final_time = float(sol.t[-1])
    if event_index == 0:
        return "switch", None, final_state, final_time, sol
    if event_index == 1:
        return "outer", None, final_state, final_time, sol
    if event_index is not None and 2 <= event_index < 2 + len(prisms):
        return "ground", event_index - 2, final_state, final_time, sol
    if event_index is not None:
        return "invalid", None, final_state, final_time, sol
    return ("timeout" if sol.success else "solver_failure"), None, final_state, final_time, sol


def _integrate_core(state: np.ndarray, t0: float, r_switch: float, ctx: V16Context, config: GroundRateConfig) -> tuple[str, np.ndarray, float, float, Any]:
    escape_radius = config.full_escape_factor * r_switch
    speed = max(float(np.linalg.norm(state[9:12] - state[6:9])), 1.0e-12)
    t_span = max(config.core_time_factor * r_switch / speed, config.core_min_time_s)

    def escape_event(_t: float, y: np.ndarray) -> float:
        radius = float(np.linalg.norm(y[3:6] - y[0:3]))
        if _relative_radial_velocity(y) <= 0.0:
            return -abs(radius - escape_radius) - 1.0e-30
        return radius - escape_radius
    escape_event.terminal = True
    escape_event.direction = 1

    events: list[Callable] = [escape_event]
    if ctx.invalid_z_floor_m is not None:
        def invalid_event(_t: float, y: np.ndarray) -> float:
            return float(min(y[2], y[5]) - ctx.invalid_z_floor_m)
        invalid_event.terminal = True
        invalid_event.direction = -1
        events.append(invalid_event)

    sol = solve_ivp(
        _core_rhs(ctx, config),
        (t0, t0 + t_span),
        state,
        method="DOP853",
        rtol=config.core_rtol,
        atol=config.core_atol,
        max_step=config.core_max_step_s,
        events=events,
    )
    separation = np.linalg.norm(sol.y[3:6] - sol.y[0:3], axis=0)
    d_min = float(np.min(separation)) if separation.size else np.nan
    final_state = np.asarray(sol.y[:, -1], dtype=float)
    final_time = float(sol.t[-1])
    if len(sol.t_events[0]):
        return "escaped_core", final_state, final_time, d_min, sol
    if len(events) > 1 and len(sol.t_events[1]):
        return "invalid", final_state, final_time, d_min, sol
    return ("core_timeout" if sol.success else "core_solver_failure"), final_state, final_time, d_min, sol


def simulate_history(row: pd.Series | dict[str, Any], ctx: V16Context, prisms: list[MetalPrism], config: GroundRateConfig, seed: int) -> dict[str, Any]:
    record = dict(row)
    speed = float(record["v_inf_m_s"])
    b = float(record["b_m"])
    theta = float(record["theta_rad"])
    alpha = float(record["alpha_rad"])
    psi = float(record["psi_rad"])
    r_switch = float(record["R_switch_m"])
    required_outer = max([config.outer_radius_m] + [float(np.linalg.norm(p.hi)) + 1.0e-3 for p in prisms])
    if b >= required_outer:
        return {"resolved": False, "status": "b_exceeds_outer_radius", "mean_phonons": np.nan}
    x_dm, v_dm = initial_state_v16(b, required_outer, theta, alpha, psi, speed)
    state = np.zeros(18, dtype=float)
    state[3:6] = x_dm
    state[9:12] = v_dm
    t = 0.0
    rng = np.random.default_rng(seed)
    ground_count = 0
    core_count = 0
    min_separation = float(np.linalg.norm(x_dm))
    ground_kinds: list[str] = []
    status = "unresolved"
    resolved = False

    while True:
        mode, prism_index, state, t, _sol = _integrate_outside(state, t, r_switch, required_outer, ctx, prisms, config)
        min_separation = min(min_separation, float(np.linalg.norm(state[3:6] - state[0:3])))
        if mode == "outer":
            status = "escaped_outer"
            resolved = True
            break
        if mode == "switch":
            if core_count >= config.max_core_passes:
                status = "max_core_passes"
                break
            core_count += 1
            core_status, state, t, d_min, _ = _integrate_core(state, t, r_switch, ctx, config)
            min_separation = min(min_separation, d_min)
            if core_status != "escaped_core":
                status = core_status
                break
            continue
        if mode == "ground":
            if ground_count >= config.max_ground_interactions:
                status = "max_ground_interactions"
                break
            ground_count += 1
            prism = prisms[int(prism_index)]
            interaction = interact_with_prism(state[3:6], state[9:12], ctx, prism, config, rng)
            ground_kinds.append(f"{prism.name}:{interaction['kind']}")
            if not np.isfinite(interaction["delay_s"]):
                status = "metal_absorbed_model"
                resolved = True
                break
            delay = float(interaction["delay_s"])
            state[0:3], state[6:9] = _advance_harmonic(state[0:3], state[6:9], ctx.omega_modes_rad_s, delay)
            t += delay
            state[3:6] = np.asarray(interaction["position"], dtype=float)
            state[9:12] = np.asarray(interaction["velocity"], dtype=float)
            continue
        status = mode
        break

    cos_q = state[12:15]
    sin_q = state[15:18]
    mode_energy = (cos_q**2 + sin_q**2) / (2.0 * ctx.m_ion_kg)
    mean_phonons_modes = mode_energy / (HBAR * ctx.omega_modes_rad_s)
    target = int(config.target_mode)
    mean_phonons = float(mean_phonons_modes[target]) if resolved else (0.0 if config.unresolved_policy == "zero" else np.nan)
    p_ge1 = float(-math.expm1(-max(mean_phonons, 0.0))) if np.isfinite(mean_phonons) else np.nan
    M = int(config.exact_phonon_number)
    p_M = float(math.exp(-mean_phonons) * mean_phonons**M / math.factorial(M)) if np.isfinite(mean_phonons) and mean_phonons >= 0.0 else np.nan
    return {
        "resolved": resolved,
        "status": status,
        "mean_phonons": mean_phonons,
        "p_ge1": p_ge1,
        "p_exact_M": p_M,
        "target_mode": target,
        "mode_energy_J": float(mode_energy[target]),
        "min_separation_m": min_separation,
        "ground_interactions": ground_count,
        "core_passes": core_count,
        "ground_history": "|".join(ground_kinds),
        "final_time_s": t,
        "outer_radius_m": required_outer,
    }


def _resolve_pilot_path(ns: dict[str, Any]) -> Path:
    for name in ("PILOT_OUTPUT_CSV", "TEST1P5_TEMPLATE_CSV"):
        value = ns.get(name)
        if value is not None and Path(value).is_file():
            return Path(value)
    prefix = ns.get("TEST1P5_OUTPUT_PREFIX")
    if prefix:
        candidate = Path(f"{prefix}_test2_input.csv")
        if candidate.is_file():
            return candidate
    run_dir = Path(ns.get("RUN_DIRECTORY", "."))
    candidates = sorted(run_dir.glob("*test1p5*_test2_input.csv")) + sorted(run_dir.glob("*test2_input.csv"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError("Could not locate the Test-1.5 weighted pilot table. Run through Test 1.5 first.")


def _row_rate_weight(row: pd.Series, density_m3: float) -> float:
    for name in ("test2_analysis_weight_unconditional", "population_weight_unconditional"):
        if name in row and np.isfinite(pd.to_numeric(row[name], errors="coerce")):
            probability_weight = float(row[name])
            break
    else:
        probability_weight = float(row.get("impact_weight", 0.0))
    b_scan = float(row.get("b_scan_max_m", np.nan))
    if not np.isfinite(b_scan):
        b_scan = float(globals().get("b_max", 4.0e-3))
    speed = float(row["v_inf_m_s"])
    return density_m3 * speed * PI * b_scan**2 * max(probability_weight, 0.0)


def run_event_rate_from_test1p5(ns: dict[str, Any], prisms: Iterable[MetalPrism], config: GroundRateConfig, output_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ctx = context_from_notebook(ns)
    prism_list = list(prisms)
    pilot_path = _resolve_pilot_path(ns)
    table = pd.read_csv(pilot_path)
    if config.max_rows is not None:
        table = table.head(int(config.max_rows)).copy()
    if len(table) == 0:
        raise RuntimeError("Test-1.5 pilot table is empty")
    required = ["v_inf_m_s", "b_m", "theta_rad", "alpha_rad", "psi_rad", "R_switch_m"]
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise KeyError("Test-1.5 table is missing required columns: " + ", ".join(missing))

    output = Path(output_dir or ns.get("RUN_DIRECTORY", ".")) / "ground_plane_event_rate"
    output.mkdir(parents=True, exist_ok=True)
    base_rng = np.random.default_rng(config.random_seed)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    total_jobs = len(table) * max(int(config.branch_replicas), 1)
    job = 0
    for source_index, source in table.iterrows():
        rate_weight = _row_rate_weight(source, config.density_m3)
        for replica in range(max(int(config.branch_replicas), 1)):
            seed = int(base_rng.integers(0, 2**32 - 1, dtype=np.uint32))
            result = simulate_history(source, ctx, prism_list, config, seed)
            record = {
                "source_index": int(source_index),
                "trajectory_id": str(source.get("trajectory_id", source_index)),
                "branch_replica": replica,
                "branch_seed": seed,
                "rate_weight_s_inv": rate_weight,
                "v_inf_m_s": float(source["v_inf_m_s"]),
                "b_m": float(source["b_m"]),
                "theta_rad": float(source["theta_rad"]),
                "alpha_rad": float(source["alpha_rad"]),
                "psi_rad": float(source["psi_rad"]),
                "R_switch_m": float(source["R_switch_m"]),
                **result,
            }
            rows.append(record)
            job += 1
            if job % 25 == 0 or job == total_jobs:
                print(f"[ground-rate] {job}/{total_jobs}", flush=True)
    results = pd.DataFrame(rows)
    replicas = max(int(config.branch_replicas), 1)
    results["phonon_rate_contribution_s_inv"] = results["rate_weight_s_inv"] * results["mean_phonons"] / replicas
    results["ge1_rate_contribution_s_inv"] = results["rate_weight_s_inv"] * results["p_ge1"] / replicas
    results["exact_M_rate_contribution_s_inv"] = results["rate_weight_s_inv"] * results["p_exact_M"] / replicas

    grouped = results.groupby("source_index", dropna=False).agg(
        phonon_rate=("phonon_rate_contribution_s_inv", "sum"),
        ge1_rate=("ge1_rate_contribution_s_inv", "sum"),
        exact_M_rate=("exact_M_rate_contribution_s_inv", "sum"),
        resolved_fraction=("resolved", "mean"),
    )
    summary = pd.DataFrame([{
        "m_dm_kg": ctx.m_dm_kg,
        "eps": ctx.eps,
        "density_m3": config.density_m3,
        "target_mode": config.target_mode,
        "exact_phonon_number": config.exact_phonon_number,
        "n_test1p5_rows": len(table),
        "branch_replicas": replicas,
        "n_history_runs": len(results),
        "resolved_fraction": float(results["resolved"].mean()),
        "selected_mode_phonon_rate_s_inv": float(results["phonon_rate_contribution_s_inv"].sum()),
        "event_rate_ge1_s_inv": float(results["ge1_rate_contribution_s_inv"].sum()),
        "event_rate_exact_M_s_inv": float(results["exact_M_rate_contribution_s_inv"].sum()),
        "between_row_phonon_rate_se_s_inv": float(grouped["phonon_rate"].std(ddof=1) / math.sqrt(len(grouped))) if len(grouped) > 1 else np.nan,
        "runtime_s": time.perf_counter() - start,
        "pilot_input": str(pilot_path),
        "unresolved_policy": config.unresolved_policy,
    }])

    if config.save_trajectory_rows:
        results.to_csv(output / "ground_plane_event_rate_histories.csv", index=False)
    summary.to_csv(output / "ground_plane_event_rate_summary.csv", index=False)
    (output / "ground_plane_event_rate_config.json").write_text(json.dumps({"config": asdict(config), "prisms": [asdict(p) for p in prism_list]}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    return results, summary
