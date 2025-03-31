import numpy as np
def compute_g1_to_g25(params,k):
    """
    Computes g1-g12
    """
    # Extract parameters from the dictionary
    a1    = params["a1"]    # vector
    a2    = params["a2"]    # vector
    a3    = params["a3"]    # vector
    d     = params["d"]     # vector
    v     = a2-d
    v_norm = np.linalg.norm(v)
    cos_alpha = v[0]/v_norm
    cos_beta = v[1]/v_norm
    cos_gamma = v[2]/v_norm
    sin_gamma = np.sqrt(1-cos_gamma**2)

    # Precompute the exponentials to avoid repeating calculations
    E1 = np.exp(1j * np.dot(k, (a1 - d)))
    E2 = np.exp(1j * np.dot(k, (a2 - d)))
    E3 = np.exp(1j * np.dot(k, (a3 - d)))
    
    # Compute the various g values as given
    g0  = E1 + E2 + E3
    g1  = (E2 - E1) * cos_alpha
    g2  = (E1 + E2 - 2 * E3) * cos_beta
    g3  = g0 * cos_gamma
    g4  = (E1 + E2) * cos_alpha**2
    g5  = g0 - g4
    g6  = g1 * cos_gamma
    g7  = (E1 + E2 + 4 * E3) * cos_beta**2
    g8  = g0 - g7
    g9  = g0 * cos_gamma**2
    g10 = g0 * sin_gamma**2
    g11 = g2 * cos_gamma
    g12 = g1 * cos_beta
    
    # Return all computed quantities in a dictionary

    """
    Computes g13-g25
    """
    v     = a1+a3-d
    v_norm = np.linalg.norm(v)
    cos_alpha = v[0]/v_norm
    cos_beta = v[1]/v_norm
    cos_gamma = v[2]/v_norm
    sin_gamma = np.sqrt(1-cos_gamma**2)

    # Precompute the exponentials to avoid repeating calculations
    E1 = np.exp(1j * np.dot(k, (a2+a3 - d)))
    E2 = np.exp(1j * np.dot(k, (a1+a3 - d)))
    E3 = np.exp(1j * np.dot(k, (a2+a1 - d)))
    
    # Compute the various g values as given
    g13  = E1 + E2 + E3
    g14  = (E2 - E1) * cos_alpha
    g15  = (E1 + E2 - 2 * E3) * cos_beta
    g16  = g13 * cos_gamma
    g17  = (E1 + E2) * cos_alpha**2
    g18  = g13 - g17
    g19  = g14 * cos_gamma
    g20  = (E1 + E2 + 4 * E3) * cos_beta**2
    g21  = g13 - g20
    g22  = g13 * cos_gamma**2
    g23  = g13 * sin_gamma**2
    g24  = g15 * cos_gamma
    g25  = g14 * cos_beta

    return {
        "g0": g0,  "g1": g1,  "g2": g2,  "g3": g3,  "g4": g4,  "g5": g5,  "g6": g6,  "g7": g7,  "g8": g8,  "g9": g9,  "g10": g10,  "g11": g11,  "g12": g12, "g13": g13,  "g14": g14,  "g15": g15,  "g16": g16,  "g17": g17,  "g18": g18,  "g19": g19,  "g20": g20,  "g21": g21,  "g22": g22,  "g23": g23,  "g24": g24,  "g25": g25
    }


def compute_g26_to_g31(params,k):
    """
    Computes g26, g27, g28, g29, g30, and g31 based on:
    """

    a1 = params["a1"]
    a2 = params["a2"]
    a3 = params["a3"]
    
    # If k is a vector, use np.dot(k, (a1 - a2)) instead of k*(a1 - a2).
    # For clarity, we'll show the dot product form below:
    E12 = np.exp(1j * np.dot(k, (a1 - a2)))
    E21 = np.exp(1j * np.dot(k, (a2 - a1)))
    E13 = np.exp(1j * np.dot(k, (a1 - a3)))
    E31 = np.exp(1j * np.dot(k, (a3 - a1)))
    E23 = np.exp(1j * np.dot(k, (a2 - a3)))
    E32 = np.exp(1j * np.dot(k, (a3 - a2)))
    
    g26 = E12 + E21 + E23 + E32 + E13 + E31
    g27 = (E21 - E12) + 0.5*(E23 - E32) + 0.5*(E31 - E13)
    g28 = 0.5 * np.sqrt(3) * (E31 - E13) + 0.5 * np.sqrt(3) * (E32 - E23)
    g29 = 0.25 * (E13 + E31) + 0.25 * (E23 + E32) + (E21 + E12)
    g30 = 3/4 * (E13 + E31) + 3/4*(E23 + E32)
    g31 = 0.25 * np.sqrt(3) * (E31 + E13 - E32 - E23)
    
    return {
        "g26": g26,
        "g27": g27,
        "g28": g28,
        "g29": g29,
        "g30": g30,
        "g31": g31
    }
    
import numpy as np

def make_H11(params, k=None):
    """
    Construct the 8x8 block H_{11} (and similarly one might do H_{22}),
    following the form of Table IX (s, p) with spin-orbit terms.
    
    Basis assumed as:
      index 0 -> s (spin up)
      index 1 -> s (spin down)
      index 2 -> p_x (spin up)
      index 3 -> p_y (spin up)
      index 4 -> p_z (spin up)
      index 5 -> p_x (spin down)
      index 6 -> p_y (spin down)
      index 7 -> p_z (spin down)
      
    The 'params' dictionary should contain (example names):
      E_s, E_p              : onsite energies
      V_s_sigma, V_sp_sigma : Slater-Koster s-sigma & s-p-sigma integrals
      V_p_sigma, V_p_pi     : Slater-Koster p-p-sigma & p-p-pi integrals
      lambda_so             : spin-orbit parameter (λ)
      g26, g27, g28, g29,
      g30, g31              : the geometric prefactors g_{ij} from Table IX
      etc.
      
    Returns:
      H11: (8,8) complex numpy array
    """
    # Short names for parameters (adjust to match your own param keys)
    E_s     = params['E_s']
    E_p     = params['E_p']
    V_ss    = params['V_ss_sigma_2']   # sometimes called  V_{ssσ}
    V_sp    = params['V_sp_sigma_2']  #               or  V_{spσ}
    V_ppσ   = params['V_pp_sigma_2']   #               or  V_{ppσ}
    V_ppπ   = params['V_pp_pi_2']      #               or  V_{ppπ}
    lamb   = params['lambda']
    
    g26     = params['g26']
    g27     = params['g27']
    g28     = params['g28']
    g29     = params['g29']
    g30     = params['g30']
    g31     = params['g31']
    
    H11 = np.zeros((8, 8), dtype=complex)

    #
    # For convenience, define an index map:
    #
    s_up  = 0;  s_dn  = 1
    px_up = 2;  py_up = 3;  pz_up = 4
    px_dn = 5;  py_dn = 6;  pz_dn = 7

    #------------------------------------------------------------------
    # Diagonal terms from Table IX
    #------------------------------------------------------------------
    H11[s_up,  s_up ] = E_s + g26*V_ss
    H11[s_dn,  s_dn ] = E_s + g26*V_ss
    
    # p_x up
    H11[px_up, px_up] = E_p + g29*V_ppσ + g30*V_ppπ
    # p_y up
    H11[py_up, py_up] = E_p + g30*V_ppσ + g29*V_ppπ
    # p_z up
    H11[pz_up, pz_up] = E_p + g26*V_ppπ  # per the table's geometry factors
    
    # p_x down
    H11[px_dn, px_dn] = E_p + g29*V_ppσ + g30*V_ppπ
    # p_y down
    H11[py_dn, py_dn] = E_p + g30*V_ppσ + g29*V_ppπ
    # p_z down
    H11[pz_dn, pz_dn] = E_p + g26*V_ppπ

    #------------------------------------------------------------------
    # Off-diagonal (s-p) blocks (no spin flip)
    #  from Table IX: the g27*V_spσ etc.
    #------------------------------------------------------------------
    # s_up -> p_x_up
    H11[s_up, px_up] = g27*V_sp
    H11[s_up, py_up] = g28*V_sp
    H11[s_up, pz_up] = 0.0
    # s_dn -> p_x_dn
    H11[s_dn, px_dn] = g27*V_sp
    H11[s_dn, py_dn] = g28*V_sp
    H11[s_dn, pz_dn] = 0.0

    # (Hermitian counterpart)
    H11[px_up, s_up] = np.conjugate(H11[s_up, px_up])
    H11[py_up, s_up] = np.conjugate(H11[s_up, py_up])
    H11[pz_up, s_up] = np.conjugate(H11[s_up, pz_up])
    H11[px_dn, s_dn] = np.conjugate(H11[s_dn, px_dn])
    H11[py_dn, s_dn] = np.conjugate(H11[s_dn, py_dn])
    H11[pz_dn, s_dn] = np.conjugate(H11[s_dn, pz_dn])

    #------------------------------------------------------------------
    # p-p off-diagonal (same spin), including the (V_ppσ - V_ppπ) piece
    #  plus the spin-orbit terms ~ i(±1/3)*lambda.
    #  From the table, one sees entries like  g31*(V''_{ppσ}-V''_{ppπ})
    #  plus or minus i(1/3)*λ for the p_x–p_y, p_y–p_z, p_z–p_x couplings.
    #------------------------------------------------------------------

    #
    # -- For spin UP block (indices 2,3,4) --
    #
    # x_up <-> y_up
    H11[px_up, py_up] = g31*(V_ppσ - V_ppπ) - 1j*(lamb/3.0)
    H11[py_up, px_up] = np.conjugate(H11[px_up, py_up])  # Hermitian
    
    # x_up <-> z_up
    #  Table typically shows 0 for x_up–z_up direct, so:
    H11[px_up, pz_up] = 0.0
    H11[pz_up, px_up] = 0.0

    # y_up <-> z_up
    #  likewise often 0 for direct p_y–p_z, check if any g31 factor is nonzero:
    H11[py_up, pz_up] = 0.0
    H11[pz_up, py_up] = 0.0


    #
    # -- For spin DOWN block (indices 5,6,7) --
    #   (similar structure, but note the spin‐orbit i sign flips may differ)
    #
    H11[px_dn, py_dn] = g31*(V_ppσ - V_ppπ) + 1j*(lamb/3.0)
    H11[py_dn, px_dn] = np.conjugate(H11[px_dn, py_dn])

    H11[px_dn, pz_dn] = 0.0
    H11[pz_dn, px_dn] = 0.0

    H11[py_dn, pz_dn] = 0.0
    H11[pz_dn, py_dn] = 0.0

    #------------------------------------------------------------------
    # Spin-flip terms among the p-orbitals (the "L·S" cross-couplings),
    # typically appear as ±(1/3)*λ, possibly times i, in x_up<->z_down, etc.
    # From the table:
    #   x↑ <-> z↓ = + (1/3)*λ
    #   y↑ <-> z↓ = - i(1/3)*λ
    #   z↑ <-> x↓ = - (1/3)*λ
    #   z↑ <-> y↓ = + i(1/3)*λ
    #  ...and so on.  Fill them carefully:
    #------------------------------------------------------------------

    # x_up -> z_dn
    H11[px_up, pz_dn] = (lamb/3.0)
    # z_up -> x_dn
    H11[pz_up, px_dn] = -(lamb/3.0)

    # y_up -> z_dn
    H11[py_up, pz_dn] = -1j*(lamb/3.0)
    # z_up -> y_dn
    H11[pz_up, py_dn] =  1j*(lamb/3.0)

    # Hermitian conjugates
    H11[pz_dn, px_up] = np.conjugate(H11[px_up, pz_dn])
    H11[px_dn, pz_up] = np.conjugate(H11[pz_up, px_dn])
    H11[pz_dn, py_up] = np.conjugate(H11[py_up, pz_dn])
    H11[py_dn, pz_up] = np.conjugate(H11[pz_up, py_dn])

    return H11


def make_H12(params):
    """
    Construct the 8x8 off-diagonal block H_{12} (and hence H_{21}).
    Based on Table X in your reference.

    Returns:
      H11: (8,8) complex numpy array
    """
    # Short names for parameters (adjust to match your own param keys)
    E_s     = params['E_s']
    E_p     = params['E_p']
    V_ssσ    = params['V_ss_sigma']  
    V_ssσ1    = params['V_ss_sigma_1']  
    V_ssσ2    = params['V_ss_sigma_2']  
    V_spσ    = params['V_sp_sigma']  
    V_spσ1    = params['V_sp_sigma_1']  
    V_spσ2    = params['V_sp_sigma_2']  
    V_ppσ    = params['V_pp_sigma']  
    V_ppσ1    = params['V_pp_sigma_1']  
    V_ppσ2    = params['V_pp_sigma_2']  
    V_ppπ    = params['V_pp_pi']  
    V_ppπ1    = params['V_pp_pi_1']  
    V_ppπ2    = params['V_pp_pi_2']  
    g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15, g16, g17, g18, g19, g20, g21, g22, g23, g24, g25, g26, g27, g28, g29, g30, g31 = params['g0'], params['g1'], params['g2'], params['g3'], params['g4'], params['g5'], params['g6'], params['g7'], params['g8'], params['g9'], params['g10'], params['g11'], params['g12'], params['g13'], params['g14'], params['g15'], params['g16'], params['g17'], params['g18'], params['g19'], params['g20'], params['g21'], params['g22'], params['g23'], params['g24'], params['g25'], params['g26'], params['g27'], params['g28'], params['g29'], params['g30'], params['g31']

    # For convenience, define an index map:
    s_up  = 0;  s_dn  = 1
    px_up = 2;  py_up = 3;  pz_up = 4
    px_dn = 5;  py_dn = 6;  pz_dn = 7

    H12 = np.zeros((8, 8), dtype=complex)

    # Diagonal terms

    H12[s_up, s_up] = g0*V_ssσ+g13*V_ssσ1
    H12[s_dn, s_dn] = g0*V_ssσ+g13*V_ssσ1
    H12[px_up, px_up] = g4*V_ppσ+g5*V_ppπ+g17*V_ppσ1+g18*V_ppπ1
    H12[py_up, py_up] = g7*V_ppσ+g8*V_ppπ+g20*V_ppσ1+g21*V_ppπ1
    H12[pz_up, pz_up] = g9*V_ppσ+g10*V_ppπ+g22*V_ppσ1+g23*V_ppπ1    
    H12[px_dn, px_dn] = g4*V_ppσ+g5*V_ppπ+g17*V_ppσ1+g18*V_ppπ1
    H12[py_dn, py_dn] = g7*V_ppσ+g8*V_ppπ+g20*V_ppσ1+g21*V_ppπ1
    H12[pz_dn, pz_dn] = g9*+V_ppσ+g10*V_ppπ+g22*V_ppσ1+g23*V_ppπ1

    # Off-diagonal terms
    H12[px_up,s_up] = -g1*V_spσ-g14*V_spσ1
    H12[py_up,s_up] = -g2*V_spσ-g15*V_spσ1
    H12[pz_up,s_up] = -g3*V_spσ-g16*V_spσ1

    H12[px_dn,s_dn] = -g1*V_spσ-g14*V_spσ1
    H12[py_dn,s_dn] = -g2*V_spσ-g15*V_spσ1
    H12[pz_dn,s_dn] = -g3*V_spσ-g16*V_spσ1

    H12[s_up,px_up]  = g1*V_spσ+g14*V_spσ1
    H12[py_up,px_up] = g12*(V_ppσ-V_ppπ)+g25*(V_ppσ1-V_ppπ1)
    H12[pz_up,px_up] = g6*(V_ppσ-V_ppπ)+g19*(V_ppσ1-V_ppπ1)

    H12[s_up,py_up]  = g2*V_spσ+g15*V_spσ1
    H12[px_up,py_up] = g12*(V_ppσ-V_ppπ)+g25*(V_ppσ1-V_ppπ1)
    H12[pz_up,py_up] = g11*(V_ppσ-V_ppπ)+g24*(V_ppσ1-V_ppπ1)

    H12[s_up,pz_up]  = g3*V_spσ+g16*V_spσ1
    H12[px_up,pz_up] = g6*(V_ppσ-V_ppπ)+g19*(V_ppσ1-V_ppπ1)
    H12[py_up,pz_up] = g11*(V_ppσ-V_ppπ)+g24*(V_ppσ1-V_ppπ1)

    H12[s_dn,px_dn] = g1*V_spσ+g14*V_spσ1
    H12[py_dn,px_dn] = g12*(V_ppσ-V_ppπ)+g25*(V_ppσ1-V_ppπ1)
    H12[pz_dn,px_dn] = g6*(V_ppσ-V_ppπ)+g19*(V_ppσ1-V_ppπ1)

    H12[s_dn,py_dn] = g2*V_spσ+g15*V_spσ1
    H12[px_dn,py_dn] = g12*(V_ppσ-V_ppπ)+g25*(V_ppσ1-V_ppπ1)
    H12[pz_dn,py_dn] = g11*(V_ppσ-V_ppπ)+g24*(V_ppσ1-V_ppπ1)

    H12[s_dn,pz_dn] = g3*V_spσ+g16*V_spσ1
    H12[px_dn,pz_dn] = g6*(V_ppσ-V_ppπ)+g19*(V_ppσ1-V_ppπ1)
    H12[py_dn,pz_dn] = g11*(V_ppσ-V_ppπ)+g24*(V_ppσ1-V_ppπ1)

    return H12


def vca_params(params_Bi, params_Sb, x,alphas):
    """
    Returns a dictionary of VCA-mixed parameters:
        params_VCA = (1 - x)*params_Bi + x*params_Sb
    For simplicity, we do a direct entrywise linear combination for keys
    that are numeric scalars or NumPy arrays.
    """

    def linear_mix(a, b, x):
        return (1.0 - x)*a + x*b
    def non_linear_mix(a, b, x):
        return (1.0 - x**2)*a + (x)*b
    def fitting_mix(a, b, x, key):
        alp = 0.0
        if key == "V_ss_sigma_1":
            alp = alphas[0]
        elif key == "V_ss_sigma_2":
            alp = alphas[1]
        elif key == "V_sp_sigma_1":
            alp = alphas[2]
        elif key == "V_sp_sigma_2":
            alp = alphas[3] 
        elif key == "V_pp_sigma_1":
            alp = alphas[4]
        elif key == "V_pp_sigma_2":
            alp = alphas[5]
        elif key == "V_pp_pi_1":
            alp = alphas[6]
        elif key == "V_pp_pi_2":
            alp = alphas[7]
        elif key == "E_s" or key == "E_p":
            alp = alphas[8]
        return (1.0 - x**2)*a + x*b + alp*x*(1-x)

    params_vca = {}
    # Onsite, lattice keys 
    linear_key = ['a','c','alpha','g','mu','d1','d2','E_s','E_p','lambda']
    # Hopping keys
    hopping_key = ['V_ss_sigma','V_sp_sigma','V_pp_sigma','V_pp_pi','V_ss_sigma_1','V_sp_sigma_1','V_pp_sigma_1','V_pp_pi_1','V_ss_sigma_2','V_sp_sigma_2','V_pp_sigma_2','V_pp_pi_2']

    for key in params_Bi.keys():
        bi_val = params_Bi[key]
        sb_val = params_Sb[key]
        ## Linear or non-linear mixing
        ##Lattice param, Core 
        #if key in linear_key:    
        #   params_vca[key] = linear_mix(bi_val, sb_val, x)
        #elif key in hopping_key:
        #   params_vca[key] = non_linear_mix(bi_val, sb_val, x)
        
        ## Fine-tuning mixing
        if key in linear_key:    
           params_vca[key] = linear_mix(bi_val, sb_val, x)
        elif key in hopping_key:
           params_vca[key] = fitting_mix(bi_val, sb_val, x, key)

        if x == 0.0:
            #print(f"Bi: {key} = {bi_val}, {params_vca[key]}")
            if bi_val != params_vca[key]:
                print(f"Bi: {key} = {bi_val}, {params_vca[key]}")


    # Now store derived parameters directly into the dictionary
    params_vca["b"] = params_vca["a"] / params_vca["c"]

    # Lattice vectors a1, a2, a3
    params_vca["a1"] = np.array([
        -0.5*params_vca["a"], 
        -np.sqrt(3)/6*params_vca["a"], 
        (1.0/3)*params_vca["c"]
    ])
    params_vca["a2"] = np.array([
        0.5*params_vca["a"], 
        -np.sqrt(3)/6*params_vca["a"], 
        (1.0/3)*params_vca["c"]
    ])
    params_vca["a3"] = np.array([
        0.0, 
        (np.sqrt(3)/3)*params_vca["a"], 
        (1.0/3)*params_vca["c"]
    ])

    # Reciprocal-like vectors b1, b2, b3 (scaled by “g”)
    params_vca["b1"] = np.array([
        -1.0, 
        -np.sqrt(3)/3, 
        params_vca["b"]
    ]) * params_vca["g"]

    params_vca["b2"] = np.array([
        1.0,
        -np.sqrt(3)/3, 
        params_vca["b"]
    ]) * params_vca["g"]

    params_vca["b3"] = np.array([
        0.0,
        (2*np.sqrt(3))/3,
        params_vca["b"]
    ]) * params_vca["g"]

    # Displacement vector d
    params_vca["d"] = np.array([
        0.0, 
        0.0, 
        2.0*params_vca["mu"]*params_vca["c"]
    ])
    return params_vca

def solve_H(params,k):
    k_vec = k[0]*np.array(params["b1"]) + k[1]*np.array(params["b2"]) + k[2]*np.array(params["b3"])
    # Compute dot product between d and k_vec
    d = params["d"]
    k_dot_d = np.dot(k_vec,d)
    params_g1 = compute_g1_to_g25(params,k_vec)
    params_g2 = compute_g26_to_g31(params,k_vec)
    params_all = {**params, **params_g1, **params_g2}
    H11 = make_H11(params_all)
    H22 = H11.copy()  
    H12 = make_H12(params_all)
    H21 = H12.conjugate().T 
    H = np.block([
        [H11, H12],
        [H21, H22]
    ])
    # Solve eigen value of H 
    eigenvalues, eigenvectors_single = np.linalg.eigh(H)
    return eigenvalues, eigenvectors_single