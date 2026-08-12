"""Exact coupled-Mathieu/Floquet validity classifier used by the event-rate notebooks.

This implements the same reference A and Q matrices and House Fourier-hierarchy
criteria as the user's coupled stability/pseudopotential map.  The expensive
calculation is one-dimensional in rho=epsilon/m.  Results are cached and then
mapped to any epsilon-mass grid with straight boundaries m=epsilon/rho.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings
import numpy as np
from scipy.integrate import solve_ivp

A_REFERENCE = np.array([
    [1.59747084e-03, -1.71433182e-19, 1.04230051e-18],
    [-1.71433182e-19, -5.36175342e-04, 2.47333621e-19],
    [1.04230051e-18, 2.47333621e-19, -1.06129549e-03],
], dtype=float)
Q_REFERENCE = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.23775125, 0.0],
    [0.0, 0.0, -0.23775125],
], dtype=float)

# v16 constants: reference matrices apply to epsilon=1 and m=171 u.
REFERENCE_EPSILON = 1.0
REFERENCE_MASS_KG = 171.0 * 1.66053907e-27
REFERENCE_RATIO = REFERENCE_EPSILON / REFERENCE_MASS_KG

GROWTH_RATE_TOL = 2.0e-5
MAX_B1_OVER_B0 = 0.10
MAX_B2_OVER_B1 = 0.10
USE_BETA_CUTOFF = False
MAX_ABS_BETA = 0.30

FLOQUET_RTOL = 2.0e-9
FLOQUET_ATOL = 2.0e-11
FLOQUET_MAX_STEP = np.pi / 120.0
FLOQUET_MIN_SEGMENTS = 24
FLOQUET_MAX_SEGMENTS = 512
FLOQUET_TARGET_EXPONENT_PER_SEGMENT = 6.0
FLOQUET_RESCALE_HIGH = 1.0e80
FLOQUET_RESCALE_LOW = 1.0e-80
FLOQUET_SOLVER_METHODS = ("DOP853", "Radau")
FAILED_GROWTH_SENTINEL = 1.0e6

FOURIER_SAMPLES_PER_PERIOD = 256
FOURIER_MAX_STEP = np.pi / 160.0
FOURIER_RTOL = 2.0e-9
FOURIER_ATOL = 2.0e-11
B0_NORM_FLOOR = 1.0e-13
B1_NORM_FLOOR_RELATIVE_TO_B0 = 1.0e-13

CLASS_UNSTABLE = 0
CLASS_STABLE_NON_PSEUDO = 1
CLASS_PSEUDO_VALID = 2

@dataclass(frozen=True)
class Boundary:
    ratio: float
    kind: str
    left_class: int
    right_class: int


def first_order_matrix(tau: float, A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    z = np.zeros((n, n), dtype=float)
    I = np.eye(n, dtype=float)
    stiffness = A + 2.0 * Q * np.cos(2.0 * tau)
    return np.block([[z, I], [-stiffness, z]])


def _segment_count(A: np.ndarray, Q: np.ndarray) -> int:
    scale = max(1.0, float(np.linalg.norm(A, 2) + 2.0 * np.linalg.norm(Q, 2)))
    requested = int(np.ceil(np.pi * np.sqrt(scale) / FLOQUET_TARGET_EXPONENT_PER_SEGMENT))
    return int(np.clip(requested, FLOQUET_MIN_SEGMENTS, FLOQUET_MAX_SEGMENTS))


def scaled_monodromy(A: np.ndarray, Q: np.ndarray):
    n = A.shape[0]
    d = 2 * n
    fundamental = np.eye(d, dtype=float)
    log_scale = 0.0
    edges = np.linspace(0.0, np.pi, _segment_count(A, Q) + 1)

    def rhs(tau, flat):
        M = flat.reshape(d, d)
        return (first_order_matrix(tau, A, Q) @ M).ravel()

    last = ""
    for t0, t1 in zip(edges[:-1], edges[1:]):
        result = None
        length = float(t1 - t0)
        for method in FLOQUET_SOLVER_METHODS:
            cand = solve_ivp(
                rhs, (float(t0), float(t1)), fundamental.ravel(), method=method,
                rtol=FLOQUET_RTOL, atol=FLOQUET_ATOL,
                max_step=min(FLOQUET_MAX_STEP, length / 4.0),
            )
            last = cand.message
            if cand.success and np.all(np.isfinite(cand.y[:, -1])):
                result = cand
                break
        if result is None:
            return fundamental, log_scale, False, last
        fundamental = result.y[:, -1].reshape(d, d)
        scale = float(np.max(np.abs(fundamental)))
        if not np.isfinite(scale) or scale == 0.0:
            return fundamental, log_scale, False, "non-finite fundamental matrix"
        if scale > FLOQUET_RESCALE_HIGH or scale < FLOQUET_RESCALE_LOW:
            fundamental /= scale
            log_scale += float(np.log(scale))
    return fundamental, log_scale, True, "success"


def floquet_spectrum(A: np.ndarray, Q: np.ndarray):
    scaled, log_scale, success, message = scaled_monodromy(A, Q)
    if not success:
        return FAILED_GROWTH_SENTINEL, None, None, False, message
    try:
        multipliers, eigenvectors = np.linalg.eig(scaled)
    except np.linalg.LinAlgError as exc:
        return FAILED_GROWTH_SENTINEL, None, None, False, str(exc)
    mags = np.maximum(np.abs(multipliers), np.finfo(float).tiny)
    growth = float(np.max(np.log(mags) + log_scale) / np.pi)
    if not np.isfinite(growth):
        growth = FAILED_GROWTH_SENTINEL
    elif abs(growth) < 1e-13:
        growth = 0.0
    return growth, multipliers, eigenvectors, True, "success"


def _select_modes(multipliers: np.ndarray, n: int) -> np.ndarray:
    phases = np.angle(multipliers)
    positive = np.flatnonzero(phases > 1e-9)
    if positive.size == n:
        return positive
    unit = np.exp(1j * phases)
    remaining = list(range(unit.size))
    selected = []
    while remaining and len(selected) < n:
        i = remaining.pop(0)
        if not remaining:
            selected.append(i)
            break
        target = np.conjugate(unit[i])
        pos = int(np.argmin([abs(unit[j] - target) for j in remaining]))
        j = remaining.pop(pos)
        pi, pj = phases[i], phases[j]
        if pi > 1e-9 and pj <= 1e-9:
            selected.append(i)
        elif pj > 1e-9 and pi <= 1e-9:
            selected.append(j)
        else:
            selected.append(i if pi >= pj else j)
    if len(selected) != n:
        raise RuntimeError("could not select independent Floquet modes")
    return np.asarray(selected, dtype=int)


def _ratios(b0: float, b1: float, b2: float):
    if not np.isfinite(b0) or b0 <= B0_NORM_FLOOR:
        return np.inf, np.inf, np.inf
    r10 = b1 / b0
    r20 = b2 / b0
    if b1 <= B1_NORM_FLOOR_RELATIVE_TO_B0 * b0:
        r21 = 0.0 if b2 <= B1_NORM_FLOOR_RELATIVE_TO_B0 * b0 else np.inf
    else:
        r21 = b2 / b1
    return float(r10), float(r21), float(r20)


def fourier_metrics(A: np.ndarray, Q: np.ndarray, multipliers: np.ndarray, eigenvectors: np.ndarray):
    n = A.shape[0]
    d = 2 * n
    try:
        idx = _select_modes(multipliers, n)
    except RuntimeError:
        return False, np.inf, np.inf, np.inf, np.inf
    modes0 = np.asarray(eigenvectors[:, idx], dtype=complex)
    for j in range(n):
        norm = float(np.linalg.norm(modes0[:, j]))
        if not np.isfinite(norm) or norm == 0.0:
            return False, np.inf, np.inf, np.inf, np.inf
        modes0[:, j] /= norm

    def rhs(tau, flat):
        modes = flat.reshape(d, n)
        return (first_order_matrix(tau, A, Q) @ modes).ravel()

    tau_ep = np.linspace(0.0, np.pi, FOURIER_SAMPLES_PER_PERIOD + 1)
    sol = solve_ivp(
        rhs, (0.0, np.pi), modes0.ravel(), method="DOP853", t_eval=tau_ep,
        rtol=FOURIER_RTOL, atol=FOURIER_ATOL, max_step=FOURIER_MAX_STEP,
    )
    if not sol.success or not np.all(np.isfinite(sol.y)):
        return False, np.inf, np.inf, np.inf, np.inf
    tau = tau_ep[:-1]
    states = sol.y[:, :-1].reshape(d, n, FOURIER_SAMPLES_PER_PERIOD)
    positions = np.transpose(states[:n, :, :], (2, 0, 1))
    phases = np.angle(multipliers[idx])
    beta = phases / np.pi
    periodic = np.empty_like(positions, dtype=complex)
    for j in range(n):
        periodic[:, :, j] = positions[:, :, j] * np.exp(-1j * beta[j] * tau)[:, None]
    r10s, r21s, r20s = [], [], []
    for j in range(n):
        norms = {}
        pm = periodic[:, :, j]
        for harmonic in (-2, -1, 0, 1, 2):
            coefficient = np.mean(pm * np.exp(-2j * harmonic * tau)[:, None], axis=0)
            norms[harmonic] = float(np.linalg.norm(coefficient))
        r10, r21, r20 = _ratios(norms[0], max(norms[-1], norms[1]), max(norms[-2], norms[2]))
        r10s.append(r10); r21s.append(r21); r20s.append(r20)
    return True, float(max(r10s)), float(max(r21s)), float(max(r20s)), float(np.max(np.abs(beta)))


def classify_ratio_exact(ratio: float):
    scale = float(ratio) / REFERENCE_RATIO
    A = scale * A_REFERENCE
    Q = scale * Q_REFERENCE
    growth, multipliers, eigenvectors, success, _ = floquet_spectrum(A, Q)
    stable = bool(success and growth <= GROWTH_RATE_TOL)
    if not stable:
        return CLASS_UNSTABLE, growth, np.nan, np.nan, np.nan, np.nan
    ok, r10, r21, r20, beta = fourier_metrics(A, Q, multipliers, eigenvectors)
    pseudo = bool(ok and r10 <= MAX_B1_OVER_B0 and r21 <= MAX_B2_OVER_B1)
    if USE_BETA_CUTOFF:
        pseudo = pseudo and beta <= MAX_ABS_BETA
    cls = CLASS_PSEUDO_VALID if pseudo else CLASS_STABLE_NON_PSEUDO
    return cls, growth, r10, r21, r20, beta


def _interp_log_zero(x0, y0, x1, y1):
    lx0, lx1 = np.log10(x0), np.log10(x1)
    if not np.isfinite(y0) or not np.isfinite(y1) or np.isclose(y0, y1):
        return float(np.sqrt(x0 * x1))
    f = float(np.clip(-y0 / (y1 - y0), 0.0, 1.0))
    return float(10.0 ** (lx0 + f * (lx1 - lx0)))


def build_scan_cache(path: str | Path, ratio_min=1e8, ratio_max=1e30, n=900, progress=True):
    path = Path(path)
    ratio = np.geomspace(float(ratio_min), float(ratio_max), int(n))
    classification = np.empty(ratio.size, dtype=np.int8)
    growth = np.empty(ratio.size, dtype=float)
    r10 = np.full(ratio.size, np.nan)
    r21 = np.full(ratio.size, np.nan)
    r20 = np.full(ratio.size, np.nan)
    beta = np.full(ratio.size, np.nan)
    for i, rho in enumerate(ratio):
        cls, g, a, b, cc, d = classify_ratio_exact(float(rho))
        classification[i] = cls; growth[i] = g; r10[i] = a; r21[i] = b; r20[i] = cc; beta[i] = d
        if progress and (i == 0 or (i + 1) % 25 == 0 or i + 1 == ratio.size):
            print(f"{i+1:4d}/{ratio.size}: rho={rho:.6e}, class={int(cls)}, growth={g:.3e}", flush=True)
    transitions = np.flatnonzero(classification[:-1] != classification[1:])
    boundaries = []
    score = np.maximum(r10 / MAX_B1_OVER_B0, r21 / MAX_B2_OVER_B1)
    for i in transitions:
        left, right = int(classification[i]), int(classification[i + 1])
        stability_changed = (left == CLASS_UNSTABLE) != (right == CLASS_UNSTABLE)
        pseudo_changed = (left == CLASS_PSEUDO_VALID) != (right == CLASS_PSEUDO_VALID)
        if stability_changed:
            br = _interp_log_zero(ratio[i], growth[i] - GROWTH_RATE_TOL, ratio[i+1], growth[i+1] - GROWTH_RATE_TOL)
            kind = "stability+pseudopotential" if pseudo_changed else "stability"
        else:
            br = _interp_log_zero(ratio[i], score[i] - 1.0, ratio[i+1], score[i+1] - 1.0)
            kind = "pseudopotential"
        boundaries.append((br, kind, left, right))
    np.savez_compressed(
        path,
        ratio=ratio, classification=classification, growth=growth,
        b1_over_b0=r10, b2_over_b1=r21, b2_over_b0=r20, max_abs_beta=beta,
        boundary_ratio=np.array([x[0] for x in boundaries], float),
        boundary_kind=np.array([x[1] for x in boundaries], dtype="U32"),
        boundary_left=np.array([x[2] for x in boundaries], np.int8),
        boundary_right=np.array([x[3] for x in boundaries], np.int8),
    )
    return load_scan_cache(path)


def load_scan_cache(path: str | Path):
    data = np.load(Path(path), allow_pickle=False)
    return {k: data[k] for k in data.files}


def classify_from_cache(ratio_values, cache):
    ratio_values = np.asarray(ratio_values, dtype=float)
    scan_ratio = np.asarray(cache["ratio"], dtype=float)
    classes = np.asarray(cache["classification"], dtype=np.int8)
    # Piecewise-constant classification in logarithmic-ratio intervals. Use the
    # nearest scanned point; boundaries are separately plotted from interpolated
    # transitions.
    idx = np.searchsorted(scan_ratio, ratio_values)
    idx = np.clip(idx, 1, scan_ratio.size - 1)
    left = scan_ratio[idx - 1]
    right = scan_ratio[idx]
    choose_right = np.abs(np.log(ratio_values / right)) < np.abs(np.log(ratio_values / left))
    nearest = np.where(choose_right, idx, idx - 1)
    return classes[nearest]


def boundaries_from_cache(cache):
    return [
        Boundary(float(r), str(k), int(l), int(rr))
        for r, k, l, rr in zip(
            cache["boundary_ratio"], cache["boundary_kind"],
            cache["boundary_left"], cache["boundary_right"],
        )
    ]
