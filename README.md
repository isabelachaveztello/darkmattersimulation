DM SImulation.ipynb - Itai's code

[Ion_Trap_Electric_Potential.pdf](https://github.com/user-attachments/files/21148406/Ion_Trap_Electric_Potential.pdf)

contants.py - contains all the constants and paramters used in the calculations

Trap potential:

dc_potential.ipynb, rf_potential.ipynb - calculate the DC and RF potentials, then store them in 3D arrays

ion_positions.ipynb - calculate the ion positions, eigenfrequencies, and normal modes of the 3-ion chain

Potential.py - class for the functions called in ion_positions.ipynb

verify_potential.ipynb - analyzes and verifies the potentials computed

electric_force.ipynb - computes the DC and RF electric forces by analytically taking the gradient of the potential

graph_potentials.ipynb - displays contour maps of the potentials in 2D space

forces.ipynb - correct trap force functions where z=0 is the location of the ions

Trajectory simulation:

full_simulation_updated.ipynb - most recent updates on the full trajectory simulation code; includes full-staged structure aware simulation code used in the importance sampling files

full_simulation_test_cases.ipynb - includes test cases for the behavior of a single trajectory at different energy collision regimes (adiabatic,trap-sensitive,rutherford) achieved through adjusting the speed

full_simulation_v2.ipynb (unused) - old file containing code for full trajectory simulation

DM_simulation_original.ipynb (unused) - oldest file containing simulation code

ODE.ipynb, combined_full_simulation.ipynb, diff_eq_part_1.ipynb, diff_eq_part_2.ipynb, diff_eq_part_3.ipynb, diff_eq_simulation.ipynb - SHO, full trap ion-DM, and rutherford simulations

Rutherford Scattering.ipynb - simulates rutherford scattering, adapted for parameter truncation

(m_dm,eps) parameter scan:

DM_through_materials.ipynb - quick scan for probability for MCP with (m_dm,eps) to overcome the surface potential barrier of copper and transmit through it upon contact

truncate_dm_parameters_potential.ipynb, truncate_dm_parameters_potential_full_range_combined_priority.ipynb - estimate the probability of MCP with (m_dm,eps) to overcome the potential barrier at r_min_rutherford

truncate_dm_parameters_rutherford.ipynb, truncate_dm_parameters_rutherford_full_range_coupled.ipynb - estimate the probability of MCP with (m_dm,eps) to deposit enough energy (>1e-27 J) to the ion in Rutherford scattering

escape_point_trajectories.ipynb - calculate the probability of MCP with (m_dm,eps) to exceed the escape point potential barrier based on the Maxwell-Boltzmann distribution; simulates straight line trajectories along z-axis through the escape point

matthieu_equation.ipynb - computes the matthieu equation stability and pseudopotential-valid regions in the (m_dm,eps) space

matthieu_floquet_reference.py, mathieu_floquet_reference_scan.npz, mathieu_stability_a_q.png, paul_trap_stability_and_pseudopotential.png, mathieu_stability_pseudopotential_ratio_scan.csv - matthieu stability simulation results referenced by the importance sampling files

Importance sampling:

MCP_Event_Rate_Full_Staged_v16_Coordinates.ipynb, MCP_Event_Rate_Full_Staged_v16_Problem_Points - full staged simulation for the event rates of (m_dm,eps)

MCP_Event_Rate_Screening_v16_Coordinates.ipynb - quick analytic scan for the event rates of (m_dm,eps)

dark_matter_trajectory_all_tests_connected_bowl_dc_rf_v16_hessian_fixed_ground_plane - single point (m_dm,eps) computation of event rate; conducts full trajectory parameter truncation

truncate_trajectory_parameters_{...}.ipynb - several versions of the code for truncating parameter space for a single (m_dm,eps), the newest version is in the actual single-point notebook

mcp_v16_event_rate.py, mcp_v16_event_rate_convergence.py - helper functions for the above notebooks

paul_trap_stability_and_pseudopotential.png - pseudopotential and stability map for (m_dm, eps)

v16_ground_plane_extension.py, upper_copper_punctured_geometry.png - helper functions for simulating MCPs through the ground plane, used by the single point simulation

Other files:

DM_distribution.ipynb (unused) - code for simulating a flux of many MCPs and the ion's steady state energy