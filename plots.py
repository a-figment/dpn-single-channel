import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def plot_panel_data(fname, title_data, x, t, cB, cC, f_vals, radius, VolA_history, VolE_history, \
        VolA_local, VolE_local, f1_xt, f2_xt, xc_vec, n, v_history, BAG, c_sat, cb_eff, cc_eff):
    """ 3x3 panel plot of different observables (up to the current time)
    Parameters
    -----------
        BAG : class 
            Birth and growth class object
    """

    fig, ax = plt.subplots(3, 3, figsize=(12, 12),layout="constrained")
    axs = ax.flatten() #if total_plots > 1 else [ax]
    
    fig.suptitle(title_data)
    plot_cb_cc_f(x, cB, cC, f_vals, c_sat, axs[0])
    plot_channel_radius(x, radius, ax=axs[1])
    sl = slice(0,len(VolA_history))
    plot_radii_distribution(BAG, num_bins=50, ax=axs[2])
    plot_nuclei_spatial_pdf(BAG, ax=axs[3])
    plot_total_volume(t[sl], VolA_history, VolE_history, ax=axs[4])
    plot_f_xt(t[sl], f1_xt, f2_xt=f2_xt, ax=axs[5])
    plot_effluents(t[sl], cb_eff, cc_eff, ax=axs[6])
    plot_local_volume(x, VolA_local, VolE_local, ax=axs[7])
    #plot_time_cone(xc_vec, BAG.x_grid, t[:n+1], v_history[:n+1].T, ax=axs[8])
    fig.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_cb_cc_f(x, cB, cC, f_vals, c_sat, ax=None):
    ax.hlines(y=c_sat, xmin=0, xmax=x[-1], color='gray', linestyle='--', alpha=0.3)
    ax.plot(x, cB, label='${c}_B$', color='blue')
    ax.plot(x, cC, label='${c}_C$', color='red')
    ax.plot(x[0:-1], f_vals, label='$f(x,t)$', color='magenta')
    ax.set_xlabel('$x$')
    ax.legend()
    ax.grid(True)
    return ax

def plot_effluents(t, cbeff, cceff, ax=None):
    ax.plot(t, cbeff, label='$c_B(L,t)$', color='blue')
    ax.plot(t, cceff, label='$c_C(L,t)$', color='red')
    ax.set_xlabel('$t$')
    ax.set_ylabel('Concentration')
    ax.legend()
    ax.grid(True)
    return ax

def plot_channel_radius(x, radius, ax=None):
    ax.plot(x, radius, color='black', linestyle='--')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$r(x,t)$')
    ax.grid(True)
    return ax

def plot_total_volume(t, VolA_history, VolE_history, ax=None):
    ax.plot(t, np.abs(VolA_history), label='$V_A$', color='black')
    ax.plot(t, VolE_history, label='$V_E$', color='orange')
    ax.set_xlabel('$t$')
    ax.set_ylabel('Volume')
    ax.legend()
    ax.grid(True)
    return ax

def plot_local_volume(x, VolA_local, VolE_local, ax=None):
    ax.plot(x[1:], VolA_local[1:]/np.max(VolA_local), label='$V_A(x)$', color='blue')
    ax.plot(x[1:], VolE_local[1:]/np.max(VolE_local), label='$V_E(x)$', color='red')
    ax.set_xlabel('$x$')
    ax.set_ylabel('Fraction of total volume')
    ax.legend()
    ax.grid(True)
    return ax

def plot_f_xt(t, f1_xt, f2_xt=None, ax=None):
    ax.plot(t, f1_xt, label='$f$', color='black')
    if f2_xt is not None:
        ax.plot(t, f2_xt, label=r'$f_{2}$', color='black', linestyle='--')
    ax.set_xlabel('$t$')
    ax.set_ylabel(r'\overline{f}(x,t)')
    ax.legend()
    ax.grid(True)
    return ax

def plot_radii_distribution(BAG, num_bins=50, ax=None):
    """ Distribution of nuclei radius'
    Parameters
    -----------
        BAG : class 
            Birth and growth class object
    """
    if len(BAG.nuclei_R) == 0:
        print("No nuclei have spawned yet.")
        return
        
    max_R = np.max(BAG.nuclei_R)
    if max_R == 0:
        max_R = 1e-6 
    bins = np.linspace(0, max_R, num_bins + 1)
    # Bine radii, weighted nuclei (expected) number
    expected_counts, bin_edges = np.histogram(BAG.nuclei_R, bins=bins, weights=BAG.nuclei_num)
    cov = 1/np.sqrt(np.sum(expected_counts))
    print(f"CoV % = {cov*100:.3f}") 
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bins[1] - bins[0]
    
    ax.bar(bin_centers, expected_counts, width=bin_width * 0.9, \
            color='teal', alpha=0.8, label=f"CoV = {cov:.4f}")

    ax.set_xlabel("Nuclei radius")
    ax.set_ylabel("Expected number of nuclei")
    ax.set_xlim(0, max_R * 1.05)
    ax.grid(True, linestyle='--', alpha=0.5)

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0)) 
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter)
    #ax.tight_layout()
    return ax

def plot_nuclei_spatial_pdf(BAG, ax=None):
    """ Spatial distribution of the expected number of nuclei 
    Parameters
    -----------
        BAG : class 
            Birth and growth class object
    """
    if len(BAG.nuclei_idx) == 0 or np.sum(BAG.nuclei_num) == 0:
        print("No nuclei have spawned yet.")
        return

    N_expected_x = np.bincount(BAG.nuclei_idx, weights=BAG.nuclei_num, minlength=BAG.Nx)
    N_total = np.sum(N_expected_x)
    dx = BAG.x_grid[1] - BAG.x_grid[0]
    pdf_x = N_expected_x / (N_total * dx)
    ax.plot(BAG.x_grid, pdf_x, color='purple', linewidth=2)
    ax.fill_between(BAG.x_grid, pdf_x, color='purple', alpha=0.3)
    ax.set_xlabel("$x$")
    ax.set_ylabel("Nuclei density")
    ax.set_xlim(BAG.x_grid[0], BAG.x_grid[-1])
    ax.grid(True, linestyle='--', alpha=0.5)
    return ax


def plot_snapshot(fname, title_data, c_sat, x, t, cB, cC, f_vals):
    """ Snapshot of concentration and f(x,t) profiles 
    """
    fig, ax_snapshot = plt.subplots(figsize=(8, 6))#, sharex=True)
    fig.suptitle(title_data)
    ax_snapshot.hlines(y=c_sat, xmin=0, xmax=x[-1], color='gray', linestyle='--', alpha=0.3)
    ax_snapshot.plot(x, cB, label='${c}_B$', color='blue')
    ax_snapshot.plot(x, cC, label='${c}_C$', color='red')
    ax_snapshot.plot(x[0:-1], f_vals, label='$f(x,t)$', color='magenta')
    ax_snapshot.set_xlabel('$x$')
    ax_snapshot.legend()
    ax_snapshot.grid(True)
    fig.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_cones(cone_fname, xc_vec, x, t, n, v_history):
    """ Plots time cone up to the current time t[-1] at the coords xc_vec
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    v_fluid = 1
    for xc in xc_vec: # cone locations
        plot_time_cone(xc, x, t[:n+1], v_history[:n+1].T, ax=ax)
        ax.plot(v_fluid*t[:n+1], t[:n+1], color='blue', linestyle='dashed', linewidth=0.2)
    plt.tight_layout()
    fig.savefig(cone_fname, bbox_inches="tight", dpi=300)
    plt.close(fig)
    #plt.show()

def plot_snapshot_cone(fname, title_data, c_sat, x, t, cB, cC, f_vals, v_history, xc_vec, n):
    """ Combined plot of concentration/passivation profiles (top) and time cones (bottom)
    """
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    fig.suptitle(title_data)
    ax_top.hlines(y=c_sat, xmin=0, xmax=x[-1], color='gray', linestyle='--', alpha=0.3)
    ax_top.plot(x, cB, label='${c}_B$', color='blue')
    ax_top.plot(x, cC, label='${c}_C$', color='red')
    ax_top.plot(x[0:-1], f_vals, label='$f(x,t)$', color='magenta')
    #ax_top.set_xlabel('$x$')
    ax_top.legend()
    ax_top.grid(True)

    v_fluid = 1
    for xc in xc_vec: # cone locations
        plot_time_cone(xc, x, t[:n+1], v_history[:n+1].T, ax=ax_bot)
        #ax_bot.plot(v_fluid*t[:n+1], t[:n+1], color='blue', linestyle='dashed', linewidth=0.2)
    ax_top.set_xlim([0,0.5])
    ax_bot.set_xlim([0,0.5])
    plt.tight_layout()

    fig.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_time_cone(x_c, x_grid, t_grid, v_g_history, ax=None):
    """ Plots the backward time cone at a point x_c along the pore 
        TODO: Could put nucleation rate as a heat map inside the cone

    Parameters:
    -----------
    x_c         : float, 
        Observation point of cone
        time steps up to the current time
    v_g_history : np.array 
        history of growth velocities v_g(x,t) in a given edge
    """
    
    t_current = t_grid[-1]
    dist_to_xc = np.abs(x_grid - x_c)
    
    # Points in the cone
    x_min_vals = []
    x_max_vals = []
    valid_taus = []
    
    for j, tau in enumerate(t_grid):
        if j == len(t_grid) - 1:
            R_x = np.zeros_like(x_grid)
        else: # Integrate v_g(x, s) from s = tau (j) to s = t_current
            R_x = np.trapezoid(v_g_history[:, j:], x=t_grid[j:], axis=1)
            
        # Grain radius must be >= distance to observation point
        inside_cone = R_x >= dist_to_xc
        
        # Cone data
        if np.any(inside_cone):
            x_inside = x_grid[inside_cone]
            x_min_vals.append(x_inside[0])
            x_max_vals.append(x_inside[-1])
            valid_taus.append(tau)

    # Plot the cone
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_betweenx(valid_taus, x_min_vals, x_max_vals, 
                     color='lightgray', alpha=0.5)
    ax.plot(x_min_vals, valid_taus, color='black', linewidth=1.5)
    ax.plot(x_max_vals, valid_taus, color='black', linewidth=1.5)
    ax.scatter([x_c], [t_current], color='red', s=30, zorder=5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    ax.set_xlim(x_grid[0], x_grid[-1])
    ax.set_ylim(t_grid[0], t_grid[-1] + (t_grid[-1]*0.05)) # Add slight top padding
    ax.grid(True, linestyle='--', alpha=0.5)
    return ax


#def plot_many_cones(xc_vec, BAG.x_grid, t_grid, v_g_history, J_history=None):
#
#    fig, ax = plt.subplots(figsize=(8, 6))
#    for x in xc_vec:
#        plot_dynamic_causal_cone(x, BAG.x_grid, t_grid, v_g_history, J_history=None, ax=ax)
#    fig.savefig(f"snapshots/{prefix}_snapshot-{n}.png", bbox_inches="tight", dpi=300)
#    plt.show()

#    # Plot the heatmap of J(x, tau) if provided
#    if J_history is not None:
#        # Mask out data outside the cone so we only see relevant nucleation
#        J_masked = np.ma.masked_where(~cone_mask, J_history)
#        
#        # Extent matches the full grid: [x_min, x_max, tau_min, tau_max]
#        extent = [BAG.x_grid[0], BAG.x_grid[-1], t_grid[0], t_grid[-1]]
#        #im = ax.imshow(J_masked.T, origin='lower', aspect='auto', extent=extent, 
#        #               cmap='viridis', alpha=0.8)
#
#        im = ax.imshow(J_masked.T, origin='lower', aspect='auto', extent=extent, 
#               cmap='viridis', alpha=0.8, 
#               norm=mcolors.LogNorm(vmin=np.min(J_masked[J_masked>0]), vmax=np.max(J_masked)))
#        cbar = fig.colorbar(im, ax=ax)
#        cbar.set_label('Nucleation Rate $J(x,\\tau)$')
