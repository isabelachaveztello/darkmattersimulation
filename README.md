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

Importance sampling:

MCP_Event_Rate_Full_Staged_v16_Coordinates.ipynb - full staged simulation for the event rates of (m_dm,eps)

MCP_Event_Rate_Screening_v16_Coordinates.ipynb - quick analytic scan for the event rates of (m_dm,eps)

dark_matter_trajectory_all_tests_connected_bowl_dc_rf_v16_hessian_fixed_ground_plane - single point (m_dm,eps) computation of event rate

mcp_v16_event_rate.py, mcp_v16_event_rate_convergence.py - helper functions for the above notebooks

paul_trap_stability_and_pseudopotential.png - pseudopotential and stability map for (m_dm, eps)