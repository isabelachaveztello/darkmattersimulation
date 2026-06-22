import numpy as np
import math

# ============================================================
# Physical scales (yours)
# ============================================================
omega   = 2*math.pi*36e6          # rad/s, trap frequency
T_trap  = 2*math.pi/omega         # s, trap period  (~27.8 ns)
d_chain = 3.7066438742e-06        # m, ion-ion equilibrium spacing

# Impact parameter: fixed across all three cases, per your choice
b = d_chain   # closest approach ~ ion spacing; adjust if you want tighter/looser

print(f"T_trap = {T_trap:.6e} s")
print(f"b      = {b:.6e} m")


def dm_initial_conditions(v_dm_mag, b, approach_periods=15.0,
                           axis_travel='z', axis_offset='y'):
    """
    Build x0_dm, v0_dm for a DM particle approaching the ion chain
    (centered at origin) along `axis_travel`, offset by impact parameter
    `b` along `axis_offset`, starting far enough away that the Coulomb
    force is negligible at t=0.

    `approach_periods` controls how many trap periods of "free flight"
    the DM gets before reaching closest approach -- i.e. how far back
    we place the starting point. This should be large enough that the
    starting Coulomb force on the ions is negligible, but not so large
    that you waste huge amounts of integration time on free flight for
    the fast cases.

    Returns x0_dm (3,), v0_dm (3,), and t_closest_approach (the time,
    relative to t=0, at which the DM crosses the chain's transverse
    plane -- useful for centering plots).
    """
    axes = {'x': 0, 'y': 1, 'z': 2}
    i_travel = axes[axis_travel]
    i_offset = axes[axis_offset]

    # distance to travel before reaching closest approach
    start_distance = v_dm_mag * approach_periods * T_trap

    x0_dm = np.zeros(3)
    x0_dm[i_travel] = -start_distance
    x0_dm[i_offset] = b

    v0_dm = np.zeros(3)
    v0_dm[i_travel] = v_dm_mag

    t_closest_approach = start_distance / v_dm_mag  # == approach_periods*T_trap by construction

    return x0_dm, v0_dm, t_closest_approach


def build_case(v_dm_mag, label, b=b, n_periods_total=40.0, n_points=20000,
               approach_periods=15.0):
    """
    Build full initial conditions + time span for one test case.
    """
    x0_dm, v0_dm, t_close = dm_initial_conditions(
        v_dm_mag, b, approach_periods=approach_periods
    )

    x0_ions = np.array([
        [-d_chain, 0, 0],
        [0, 0, 0],
        [d_chain, 0, 0]
    ], dtype=float)
    v0_ions = np.zeros_like(x0_ions)

    t_total = n_periods_total * T_trap
    t_span = (0.0, t_total)
    dt = t_total / n_points

    tau = b / v_dm_mag
    tau_omega = tau * omega

    print(f"\n--- {label} ---")
    print(f"v_dm           = {v_dm_mag:.6e} m/s")
    print(f"tau = b/v_dm   = {tau:.6e} s")
    print(f"tau * omega    = {tau_omega:.6e}   "
          f"(>>1 adiabatic, ~1 resonant, <<1 impulsive)")
    print(f"x0_dm          = {x0_dm}")
    print(f"v0_dm          = {v0_dm}")
    print(f"t_closest_approach (from t=0) = {t_close:.4e} s "
          f"= {t_close/T_trap:.2f} trap periods")
    print(f"t_span         = {t_span}  ({n_periods_total} trap periods)")
    print(f"dt             = {dt:.4e} s")

    return dict(x0_dm=x0_dm, v0_dm=v0_dm, x0_ions=x0_ions, v0_ions=v0_ions,
                t_span=t_span, dt=dt, tau_omega=tau_omega, t_close=t_close,
                label=label)


# ============================================================
# Case 1: Adiabatic / quasi-static
#   tau*omega >> 1  -> ions should show NO excitation, just a
#   smoothly shifting equilibrium position that relaxes back after
#   the DM passes. Use a longer approach so the "slow turn-on" is
#   actually resolved.
# ============================================================
v_dm_adiabatic = b * omega / 50.0   # tau*omega = 50
case1 = build_case(v_dm_adiabatic, "Case 1: Adiabatic (tau*omega=50)",
                    n_periods_total=10.0,      # slow DM -> fewer trap periods needed
                                                # to span the whole encounter; increase
                                                # if DM hasn't fully passed by t_span[1]
                    approach_periods=2.0,
                    n_points=20000)

# ============================================================
# Case 2: Impulsive / Rutherford-like
#   tau*omega << 1 -> the kick is much shorter than the trap period.
#   Ions should barely move during the kick and the encounter should
#   look like elastic 2-body scattering, independent of omega.
#   Need MANY trap periods after the kick to see the resulting free
#   oscillation clearly (kick is instantaneous on the trap timescale).
# ============================================================
v_dm_impulsive = b * omega * 50.0   # tau*omega = 1/50
case2 = build_case(v_dm_impulsive, "Case 2: Impulsive (tau*omega=1/50)",
                    n_periods_total=40.0,
                    approach_periods=15.0,
                    n_points=40000)

# ============================================================
# Case 3: Resonant
#   tau*omega ~ 1 -> max residual oscillation amplitude. The exact
#   peak isn't guaranteed to be precisely at tau*omega=1, so we scan
#   a range around it and you should run all of them and plot
#   resulting ion oscillation amplitude vs tau*omega to find the peak.
# ============================================================
resonant_scan_factors = [0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0]
case3_scan = []
for f in resonant_scan_factors:
    v_dm = b * omega / f
    case = build_case(v_dm, f"Case 3 scan: tau*omega={f}",
                       n_periods_total=30.0,
                       approach_periods=8.0,
                       n_points=30000)
    case3_scan.append(case)

print("\n\nAll cases built. Next: feed case['x0_dm'], case['v0_dm'], "
      "case['x0_ions'], case['v0_ions'], case['t_span'], case['dt'] "
      "into run_simulation(...) for each case, then analyze:")
print(" - Case 1: plot ion x-position vs t -> expect smooth shift, no ringing after DM leaves")
print(" - Case 2: plot ion velocity kick (delta-v) vs impact parameter/v_dm -> "
      "compare to Rutherford scattering formula")
print(" - Case 3 scan: for each run, measure post-encounter oscillation amplitude "
      "of an ion (e.g. max|x_ion - x_ion_equilibrium| after t_close), "
      "plot amplitude vs tau*omega -> find the peak")