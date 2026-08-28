import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pprint
import math
from scipy.integrate import odeint
from scipy.signal import fftconvolve
from scipy.integrate import solve_ivp

import plots as pl
import BAG as bag
matplotlib.use('Agg')

def get_dVA(dt, L, channel_radius, dr_dt_diss):
    """Change in volume of the dissolving solid nineral A.
    """
    dx = L / (len(channel_radius) - 1)
    return -dt * np.trapezoid(2.0 * np.pi * channel_radius * dr_dt_diss, dx=dx)

def get_dVE(dt, L, channel_radius, dr_dt_precip):
    """Change in volume of the precipitating solid mineral E
    """
    dx = L / (len(channel_radius) - 1)
    return -dt * np.trapezoid(2.0 * np.pi * channel_radius * dr_dt_precip, dx=dx)

def get_volume_change(dt, L, r, dr_dt_diss, dr_dt_precip):
    """ Change in volume of A and E
    """
    # padded dr/dt vectors for volume change
    full_dr_dt_diss = np.zeros(len(dr_dt_diss)+1)
    full_dr_dt_diss[1:] = dr_dt_diss
    full_dr_dt_precip = np.zeros(len(dr_dt_precip)+1)
    full_dr_dt_precip[1:] = dr_dt_precip
    Delta_VA = get_dVA(dt, L, r, full_dr_dt_diss)
    Delta_VE = get_dVE(dt, L, r, full_dr_dt_precip)
    return Delta_VA, Delta_VE

def get_nucleation_rate(cC, A=10, c_sat=1/1000):
    """ Nucleation rate from CNT: J ~ A exp(-B/ln^2 S)
        Nucleation is only possible when supersaturated S > 1.
    Parameters
    --------
    cC : np.array
        Spatial concentration profile at the current time. 
    """
    S = cC / c_sat
    J_nucl = np.zeros_like(S)
    S_greater_1 = S > (1.0 + 1e-9)
    if np.any(S_greater_1):
        log_S = np.log(S[S_greater_1])
        J_nucl[S_greater_1] = A * np.exp(-1. / (log_S**2))
    return J_nucl

def simple_pipe(params, prefix, show=True, save_snapshots=False):

    # Parameters
    A = params['A'] 
    c_sat = params['c_sat'] 
    q = params['q']
    Gamma = params['Gamma']
    K = params['K']
    G1 = params['G1']
    Da_eff = params['Da_eff']
    G2 = G1 * K
    Da1_0 = Da_eff * (1 + G1)
    Da2_0 = Da1_0 * K
    
    # Grid setup 
    L = 1.0     
    T = params['T']
    nx = 200
    dx = L / (nx - 1)
    dt = 0.3 * dx   # dt < dx for upwind
    nt = int(T / dt) + 1

    # Concentration and radius initial conditions 
    cB, cC, r = np.zeros(nx), np.zeros(nx), np.ones(nx) 
    cB_new, cC_new, r_new = np.zeros(nx), np.zeros(nx), np.zeros(nx)
    
    # Grid
    x = np.linspace(0, L, nx)
    y = np.linspace(0, 2*np.pi*r[0], nx)
    t = np.linspace(0, T, nt)

    # Volume of A in a given slice
    VolA = np.pi * ((r+0.1)**2 - r**2) * dx
    VolE = np.zeros_like(VolA)

    # Data to record
    t_hist = []
    cB_hist, cC_hist, r_hist = [], [], []
    VolA_total, VolE_total = 0, 0
    VolA_history, VolE_history = [], []
    cc_effluent, cb_effluent = [], []
    f1_xt, f2_xt = [], []
    
    # History method
    J_history = np.empty((nt, nx), dtype=np.float64)
    v_history = np.empty((nt, nx), dtype=np.float64)

    # 
    save_interval = 0.25
    next_save_time = 0.0

    classicalBAG = bag.BAGProcess(x, dx, dt, L, A, c_sat) 
    classicalBAG_backward = bag.BAGProcess(x, dx, dt, L, A, c_sat) 
    for n in range(nt):
        current_t = t[n]
            
        # Constnat supply of B at inlet
        cB_new[0], cC_new[0], r_new[0] = 1.0, 0.0, r[0]
        
        # Update transformed fraction
        growth_velocity = Gamma * K * cC / (1.0 + G2 * r)

        J_nucl = get_nucleation_rate(cC, A, c_sat)
        f_vals_classicalBAG = classicalBAG.get_f_fraction(cC, r, growth_velocity, J_nucl)[0:-1]
        #f_vals = classicalBAG_stochastic.get_f_fraction_stochastic(cC, r, growth_velocity, J_nucl)[0:-1]
        #f_vals_classicalBAG = classicalBAG.get_f_fraction_discrete(cC, r, growth_velocity, J_nucl)[0:-1]
        f_vals = f_vals_classicalBAG # not comparing

        J_history[n,:] = J_nucl
        v_history[n,:] = growth_velocity

        # Spatial derivatives 
        dcB_dx = (cB[1:] - cB[:-1]) / dx
        dcC_dx = (cC[1:] - cC[:-1]) / dx
        
        # dr/dt
        cB_i, cC_i, r_i = cB[1:], cC[1:], r[1:]
        dr_dt_diss = ((1.0 - f_vals) / (1.0 + G1 * r_i)) * cB_i # NO VOLUME TRACKING
        dr_dt_precip = - (Gamma * K * f_vals / (1.0 + G2 * r_i)) * cC_i

        # FINITE VOLUMES
        # \Delta V = 2 * pi * r * \Delta r * dx => \Delta r_max = VolA / (2 * pi * r * dx)
        dr_dt_diss_max = VolA[1:] / (2.0 * np.pi * r_i * dx * dt)
        dr_dt_diss_act = np.minimum(dr_dt_diss, dr_dt_diss_max)
        # Subtract the exact volume that was just dissolved
        VolA[1:] -= 2.0 * np.pi * r_i * dr_dt_diss_act * dx * dt
        VolE[1:] -= 2.0 * np.pi * r_i * dr_dt_precip * dx * dt
        rhs_r = dr_dt_diss_act + dr_dt_precip

        # Accumulate the total solid volumes
        dva, dve = get_volume_change(dt, L, r, dr_dt_diss_act, dr_dt_precip)
        VolA_total += dva
        VolE_total += dve

        # Reaction terms - dcB/dx, dcC/dx
        term1 = (Da1_0 * r_i / q) * dr_dt_diss_act
        term2 = Da2_0 * (r_i / (1.0 + G2 * r_i)) * (cC_i / q)
        rhs_cB = -(1.0 - f_vals) * term1
        rhs_cC =  (1.0 - f_vals) * term1 - f_vals * term2
        
        # Euler update
        v_adv = q / (r_i**2)
        epsilon = 1. # tau_diss / tau_adv
        #v_adv = 1
        cB_new[1:] = cB_i - dt * epsilon * v_adv * dcB_dx + dt * rhs_cB
        cC_new[1:] = cC_i - dt * epsilon * v_adv * dcC_dx + dt * rhs_cC
        r_new[1:]  = r_i + dt * rhs_r
        cB[:], cC[:], r[:] = cB_new[:], cC_new[:], r_new[:]

        # Save current state for plotting
        VolA_history.append(VolA_total)
        VolE_history.append(VolE_total)
        cb_effluent.append(cB[-1])
        cc_effluent.append(cC[-1])
        f2_xt.append(np.mean(f_vals_classicalBAG))
        f1_xt.append(np.mean(f_vals))

        if save_snapshots and current_t >= next_save_time:
            print(f"Recording {i}-{j} at {current_t:.2f}")
            next_save_time += save_interval

            # SNAPSHOT DATA PLOT
            title_data = (
                f"$t = $ {current_t:.2f}, $G =$ {G1:.2f}, "
                f"$Da =$ {Da1_0:.2f}, $K =$ {K:.2f}, "
                f"$\\Gamma =$ {Gamma:.2f}, $A = $ {A:.2f}"
            )
            snap_fname = f"snapshots/{prefix}_snapshot-{n}.png"
            pl.plot_snapshot(snap_fname, title_data, c_sat, x, t, cB, cC, f_vals)

            # PANEL DATA PLOT
            panel_fname = f"snapshots/{prefix}_panel-{n}.png"
            xc_vec = [0.1, 0.3, 0.4]
            pl.plot_panel_data(panel_fname, title_data, x, t, cB, cC, f_vals, r, VolA_history, VolE_history, \
                    VolA, VolE, f2_xt, f1_xt, xc_vec, n, v_history, classicalBAG, c_sat, cb_effluent, cc_effluent)

            cone_fname = f"snapshots/{prefix}_cone-{n}.png"
            #xc_vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            xc_vec = [0.1, 0.25, 0.4]
            #plot_cones(cone_fname, xc_vec, x, t, n, v_history, J_history)

            cone_snap_fname = f"snapshots/{prefix}_conesnap-{n}.png"
            pl.plot_snapshot_cone(cone_snap_fname, title_data, c_sat, x, \
                    t, cB, cC, f_vals, v_history, xc_vec, n)


    return t, cb_effluent, cc_effluent, VolA_history, VolE_history, f1_xt, f2_xt
    

# ==========================
# VARIATIONS IN DA_EFF AND A
# ==========================
Da_eff_ = [5, 1, 0.1]
A_ = [1e5, 1e4, 1e3]
Gamma, K, G1 = 1, 0.5, 0.0
T = 2
c_sat = 1/10000
idx = 0

print("     Parameters of this study:")
for i in range(len(A_)):
    for j in range(len(Da_eff_)):
        loop_params = { 'q'    : 1.0,
                  'Gamma' : Gamma,
                  'K'     : K,
                  'G1'    : G1,
                  'Da_eff': Da_eff_[j],
                  'T'     : T,
                  'A'     : A_[i],
                  'c_sat' : c_sat}
        #print(f"{idx} : {i}-{j} : Gamma {Gamma} : K {K} : G1 {G1} : Da_eff {Da_eff_[j]} : A {A_[i]} : T {T} : c_sat {c_sat}")
        print(f"{idx:<4} : {i:<2}-{j:<2} : Gamma {Gamma:<6} : K {K:<6} : G1 {G1:<6} : Da_eff {Da_eff_[j]:<10} : A {A_[i]:<10} : T {T:<6} : c_sat {c_sat:<6}")

# Effluent concentration plot
fig_eff, ax_eff = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True, layout="constrained")
fig_eff.suptitle(f'$\Gamma =$ {Gamma:.3f}, $K =$ {K}, $G = $ {G1}, $c_s = $ {c_sat}, $T = $ {T}')
axs_eff = ax_eff.flatten() 

# Final volumes 
fig_vol, ax_vol = plt.subplots(3, 3, figsize=(9, 9), sharex=True,layout="constrained")
fig_vol.suptitle(f'$\Gamma =$ {Gamma:.3f}, $K =$ {K}, $G = $ {G1}, $c_s = $ {c_sat}, $T = $ {T}')
axs_vol = ax_vol.flatten() 

# Average of f(x,t) over simulation time
fig_fxt, ax_fxt = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True,layout="constrained")
fig_fxt.suptitle(f'$\Gamma =$ {Gamma:.3f}, $K =$ {K}, $G = $ {G1}, $c_s = $ {c_sat}, $T = $ {T}')
axs_fxt = ax_fxt.flatten() 

#fig_snp, ax_snp = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)
#fig_snp.suptitle(f'$\Gamma =$ {Gamma:.3f}, $K =$ {K}, $G = $ {G1}, $c_s = $ {c_sat}, $T = $ {T}')
#axs_snp = ax_snp.flatten() 
for i in range(len(A_)):
    for j in range(len(Da_eff_)):
        print(f"Running {idx}: {i}-{j}")
        loop_params = { 'q'    : 1.0,
                  'Gamma' : Gamma,
                  'K'     : K,
                  'G1'    : G1,
                  'Da_eff': Da_eff_[j],
                  'T'     : T,
                  'A'     : A_[i],
                  'c_sat' : c_sat}
        t, cb_eff, cc_eff, va, ve, f1_xt, f2_xt = \
                simple_pipe(loop_params, prefix=f"{i}-{j}", show=False, save_snapshots=True)

        # PLOT SEQUENCE ---- effluents
        axs_eff[idx].set_title(f"A={A_[i]}, Da={Da_eff_[j]}")
        axs_eff[idx].plot(t, cb_eff, label='$c_B$', color='blue')
        axs_eff[idx].plot(t, cc_eff, label='$c_C$', color='red')
        axs_eff[idx].legend(loc='upper left')
        axs_eff[idx].grid(True)
        if j == 0:
            axs_eff[idx].set_ylabel('c$(x=L,t)$')
        if i == 2:
            axs_eff[idx].set_xlabel('t')

        # PLOT SEQUENCE ---- volumes
        axs_vol[idx].set_title(f"A={A_[i]}, Da={Da_eff_[j]}")
        axs_vol[idx].plot(t, np.abs(va), label='$V_A$', color='blue')
        axs_vol[idx].plot(t, ve, label='$V_E$', color='red')
        axs_vol[idx].legend(loc='upper left')
        axs_vol[idx].grid(True)
        if j == 0:
            axs_vol[idx].set_ylabel('Volume')
        if i == 2:
            axs_vol[idx].set_xlabel('t')

        # PLOT SEQUENCE ---- comparison of differently computed f(x,t)'s
        axs_fxt[idx].set_title(f"A={A_[i]}, Da={Da_eff_[j]}")
        axs_fxt[idx].set_ylim([0,1])
        axs_fxt[idx].plot(t, f1_xt, label=r'$f_{stoch}$', color='blue')
        axs_fxt[idx].plot(t, f2_xt, label=r'$f_{classicalBAG}$', color='red')
        axs_fxt[idx].legend(loc='upper left')
        axs_fxt[idx].grid(True)
        if j == 0:
            axs_fxt[idx].set_ylabel('f')
        if i == 2:
            axs_fxt[idx].set_xlabel('t')

        # PLOT SEQUENCE ---- 1 or 2 SNAPSHOT OF CONCENTRATION propagationg
        idx += 1
    fig_eff.savefig(f"effluent_param_sweep.png", bbox_inches="tight", dpi=300)
    fig_vol.savefig(f"volume_param_sweep.png", bbox_inches="tight", dpi=300)
    fig_fxt.savefig(f"fxt_param_sweep.png", bbox_inches="tight", dpi=300)

