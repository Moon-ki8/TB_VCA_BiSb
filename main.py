#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Moon-ki Choi
Description: This script calculates the tight-binding model for Bi and Sb.
Reference: Electronic structure of the semimetals Bi and Sb, Physical Review B (1995)
"""

from utils import *
import numpy as np
import matplotlib.pyplot as plt
import ase.io

############################
# Parameter Definitions
############################

params_Bi = {
    # Base scalar parameters for Bi
    "a": 4.5332,         # Lattice constant (Å)
    "c": 11.7967,        # Lattice constant (Å)
    "alpha": 57.19,      # α (degrees)
    "g": 1.3861,         # g (Å)
    "mu": 0.2341,        # μ Internal displacement
    # Energies / distances / integrals for Bi
    "d1": 3.062,         # Nearest-neighbor distance (Å)
    "d2": 3.512,         # Next-nearest-neighbor distance (Å)
    "E_s": -10.906,      # On-site energy for s orbital (eV)
    "E_p": -0.486,       # On-site energy for p orbital (eV)
    "V_ss_sigma":  -0.608, 
    "V_sp_sigma":   1.320, 
    "V_pp_sigma":   1.854,
    "V_pp_pi":     -0.600,
    "V_ss_sigma_1": -0.384,
    "V_sp_sigma_1":  0.433,
    "V_pp_sigma_1":  1.396,
    "V_pp_pi_1":    -0.344,
    "V_ss_sigma_2":  0.0,
    "V_sp_sigma_2":  0.0,
    "V_pp_sigma_2":  0.156,
    "V_pp_pi_2":     0.0,
    "lambda": 1.5
}

params_Sb = {
    # Base scalar parameters for Sb
    "a": 4.3007,       # Lattice constant (Å)
    "c": 11.2221,      # Lattice constant (Å)
    "alpha": 57.14,    # α (degrees)
    "g": 1.4610,       # g (Å)
    "mu": 0.2336,      # μ internal displacement
    # Distances, on-site energies, and Slater–Koster integrals for Sb
    "d1": 2.902,       # Nearest-neighbor distance (Å)
    "d2": 3.343,       # Next-nearest-neighbor distance (Å)
    "E_s": -10.068,    # On-site energy for s orbital (eV)
    "E_p": -0.926,     # On-site energy for p orbital (eV)
    # Nearest-neighbor Slater–Koster integrals
    "V_ss_sigma": -0.694,
    "V_sp_sigma":  1.554,
    "V_pp_sigma":  2.342,
    "V_pp_pi":    -0.582,
    # Next-nearest-neighbor Slater–Koster integrals (single prime)
    "V_ss_sigma_1": -0.366,
    "V_sp_sigma_1":  0.478,
    "V_pp_sigma_1":  1.418,
    "V_pp_pi_1":    -0.393,
    # Next-nearest-neighbor Slater–Koster integrals (double prime)
    "V_ss_sigma_2": 0.0,
    "V_sp_sigma_2": 0.0,
    "V_pp_sigma_2": 0.352,
    "V_pp_pi_2":    0.0,
    "lambda": 0.6
}

# Common parameter: alphas array (adjust as needed)
alphas = np.array([-1.07, 0.0, 0.0, 0.0, 0.02, -0.02, 0.0, 0.0])


############################
# Function Definitions
############################

def calculate_single_tb(Sb_concent,k_point):
    """Perform a single tight-binding model calculation for given parameters."""
    #Sb_concent = 0  # Sb concentration for this calculation
    params = vca_params(params_Bi, params_Sb, Sb_concent, alphas)
    #
    eigenvalues, eigenvectors = solve_H(params, k_point)
    return eigenvalues, eigenvectors

def compute_band_evolution():
    """
    Compute the evolution of L_s, L_a, and T bands as a function of Sb concentration.
    Returns:
        concen_list (np.ndarray): Array of Sb concentrations.
        L_a_plot (np.ndarray): L_a band (swapped post-crossing for plotting).
        L_s_plot (np.ndarray): L_s band (swapped post-crossing for plotting).
        T_band (np.ndarray): T band.
    """
    # Define k-points for L and T bands
    k_L = [0.5, 0.0, 0.0]
    k_T = [0.5, 0.5, 0.5]
    concen_list = np.linspace(0, 0.2, 80)

    eigenvalues_L_list = []
    eigenvalues_T_list = []
    
    for Sb_concent in concen_list:
        params = vca_params(params_Bi, params_Sb, Sb_concent, alphas)
        eigenvalues_L, _ = solve_H(params, k_L)
        eigenvalues_T, _ = solve_H(params, k_T)
        eigenvalues_L_list.append(eigenvalues_L)
        eigenvalues_T_list.append(eigenvalues_T)
    
    eigenvalues_L_list = np.array(eigenvalues_L_list)
    eigenvalues_T_list = np.array(eigenvalues_T_list)
    
    # Extract original L_a and L_s from eigenvalues (assumes indices 8 and 10)
    L_a_orig = eigenvalues_L_list[:, 8]
    L_s_orig = eigenvalues_L_list[:, 10]
    T_band = eigenvalues_T_list[:, 8]  # Assumed to be the T45 band
    
    # Determine crossing index and swap bands for a smooth plot
    diff = np.abs(L_a_orig - L_s_orig)
    crossing_index = np.argmin(diff)
    L_a_plot = np.copy(L_a_orig)
    L_s_plot = np.copy(L_s_orig)
    L_a_plot[crossing_index:] = L_s_orig[crossing_index:]
    L_s_plot[crossing_index:] = L_a_orig[crossing_index:]
    
    return concen_list, L_a_plot, L_s_plot, T_band

def plot_band_evolution(concen_list, L_a, L_s, T_band):
    """Plot the band evolution over Sb concentration."""
    # Set global font
    plt.rcParams["font.family"] = "Arial"
    plt.figure()
    plt.plot(concen_list * 100, L_a * 1000, label=r"L$_a$", color='red')
    plt.plot(concen_list * 100, L_s * 1000, label=r"L$_s$", color='blue')
    plt.plot(concen_list * 100, T_band * 1000, label=r"T$_{45}^+$", color='black')
    plt.xlabel("Sb concentration (x)", fontsize=15)
    plt.ylabel("Energy (meV)", fontsize=15)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig("line.png", transparent=False, dpi=300)
    plt.show()

def calculate_band_structure(Sb_concent):
    """
    Calculate and plot the band structure for Bi (Sb concentration = 0).
    Reads the Bi.cif file and computes eigenvalues along a high-symmetry k-path.
    """
    atoms = ase.io.read('Bi.cif', format='cif')
    lat = atoms.cell.get_bravais_lattice()
    
    # Define k-path (modify the path string and number of points as needed)
    kpath = lat.bandpath('GLZ', npoints=150)
    k_values = kpath.get_linear_kpoint_axis()[0]  # linear axis
    special_ticks = kpath.get_linear_kpoint_axis()[1]
    special_labels = kpath.get_linear_kpoint_axis()[2]
    
    eigenvalues_list = []
    for k in kpath.kpts:
        params = vca_params(params_Bi, params_Sb, Sb_concent, alphas)
        eigenvalues, _ = solve_H(params, k)
        eigenvalues_list.append(eigenvalues)
    
    eigenvalues_arr = np.array(eigenvalues_list)  # shape: (num_k_points, num_bands)
    
    # Plotting the band structure
    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(8, 6))
    num_bands = eigenvalues_arr.shape[1]
    for band in range(num_bands):
        plt.plot(k_values, eigenvalues_arr[:, band], color='b', marker='o',
                 lw=1, markersize=1.0)
    
    # Mark special k-points and add Fermi energy line
    for x in special_ticks:
        plt.axvline(x=x, color='k', linestyle='--', linewidth=1)
    plt.xticks(special_ticks, special_labels, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(special_ticks[0], special_ticks[-1])
    plt.ylim(-3, 3)
    plt.ylabel(r"$E$ (eV)", fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.title('Band structure of Bi (x = 0)')
    plt.savefig("band_structure.png", transparent=False, dpi=300)
    plt.show()

######## MAIN ##########
# 1. Single TB model calculation
Sb_concent,k_point = 0.0, [0.5, 0.0, 0.0]
eigenvalues_single, eigenvectors_single = calculate_single_tb(Sb_concent,k_point)

# 2. Band evolution with Sb concentration
concen_list, L_a_plot, L_s_plot, T_band = compute_band_evolution()
plot_band_evolution(concen_list, L_a_plot, L_s_plot, T_band)

# 3. Band structure calculation for Bi
calculate_band_structure(Sb_concent)