import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pprint
import math
from scipy.integrate import odeint
from scipy.signal import fftconvolve
from scipy.integrate import solve_ivp

class BAGProcess:
    """ Generic class implementing a local birth and growth processes 
        in different ways.
    """
    def __init__(self, x_grid, dx, dt, L, A, c_sat):
        # Channel details
        self.x_grid = x_grid
        self.dx = dx
        self.dt = dt
        self.Nx = len(x_grid)
        self.L = L
        
        # Nucleation params
        self.A = A
        self.c_sat = c_sat
        
        # Nuclei details
        self.nuclei_list = []  # for stoch. method
        self.nuclei_idx = np.array([], dtype=np.int32)
        self.nuclei_R   = np.array([], dtype=np.float64)
        self.nuclei_num = np.array([], dtype=np.float64)
        self.accumulated_nuclei = np.zeros(self.Nx, dtype=np.float64)

    def get_f_fraction(self, cC, channel_radius, v_growth, J_nucl, discrete=False):
        """ Computes the local transformed fraction via forward iteration of a time cone 
        """
        # Update radii of all existing grains
        if len(self.nuclei_idx) > 0:
            self.nuclei_R += v_growth[self.nuclei_idx] * self.dt
    
        # Flux of nuclei
        #J_nucl = get_nucleation_rate(cC, self.A, self.c_sat)
        spawn_mask = J_nucl > 0
        #if discrete:
        #    spawn_mask = np.floor(self.accumulated_nuclei) > np.floor(old_accumulated)

        if np.any(spawn_mask):
            # Expected number of nuclei in a volume slice
            new_indices = np.where(spawn_mask)[0]
            #spawn_counts = np.floor(self.accumulated_nuclei[new_indices]) - np.floor(old_accumulated[new_indices])
            area_i = 2 * np.pi * channel_radius[new_indices] * self.dx
            expected_num = J_nucl[new_indices] * area_i * self.dt
            
            self.nuclei_idx = np.concatenate([self.nuclei_idx, new_indices])
            initial_R = (v_growth[new_indices] * self.dt) / 2.0 
            self.nuclei_R = np.concatenate([self.nuclei_R, initial_R])
            self.nuclei_num = np.concatenate([self.nuclei_num, expected_num])
            #self.nuclei_num = np.concatenate([self.nuclei_num, spawn_counts])
    
        f_e = np.zeros(self.Nx)
        for x_idx, R, expected_num in zip(self.nuclei_idx, self.nuclei_R, self.nuclei_num):
            if R <= 0:
                continue
                
            # Bounds of current grain
            x_center = self.x_grid[x_idx]
            x_min = x_center - R
            x_max = x_center + R
            
            # Convert physical bounds to array indices and clamp to grid edges
            idx_min_float = np.floor((x_min - self.x_grid[0]) / self.dx)
            i_min = max(0, int(idx_min_float))
            # np.ceil ensures we grab the grid node just outside the right edge
            idx_max_float = np.ceil((x_max - self.x_grid[0]) / self.dx)
            i_max = min(self.Nx - 1, int(idx_max_float))

            #i_min = max(0, int((x_min - self.x_grid[0]) / self.dx))
            #i_max = min(self.Nx - 1, int(np.ceil((x_max - self.x_grid[0]) / self.dx)))
            
            # Extract the spatial slice this grain overlaps
            x_slice = self.x_grid[i_min:i_max+1]
            circumference_slice = 2 * np.pi * channel_radius[i_min:i_max+1]
            
            # Chord length needs to preserve PBC
            L_chord = 2 * np.sqrt(np.maximum(0, R**2 - (x_slice - x_center)**2))
            w_eff = np.minimum(L_chord, circumference_slice)
            prob_coverage = w_eff / circumference_slice
            f_e[i_min:i_max+1] += expected_num * prob_coverage
    
        f = 1.0 - np.exp(-f_e)
        return f

    def get_f_fraction_backward(self, current_time_idx, J_history, v_history, channel_radius):
        """ Computes the local transformed fraction from nucleation and growth rate histories
        """
        f_e = np.zeros(self.Nx)
        circumference = 2 * np.pi * channel_radius
        
        # R(x,z,t) = \int_tau^t v(x, s) ds
        R_matrix = np.zeros((current_time_idx, self.Nx))
        for tau in range(current_time_idx):
            R_matrix[tau, :] = np.sum(v_history[tau:current_time_idx, :], axis=0) * self.dt
        area = circumference * self.dx
        expected_num = J_history[:current_time_idx, :] * area * self.dt 

        # For every grid point integrate the J(x_c,t) over the time cone
        for i_c in range(self.Nx):
            x_c = self.x_grid[i_c]
            circ_c = circumference[i_c]
            
            # Nucleus at (tau, x) reaches x_c if R_matrix >= dist
            dist = np.abs(self.x_grid - x_c) 
            cone_mask = R_matrix >= dist 
            if np.any(cone_mask):
                # Filter down to ONLY the grains that successfully reach x_c
                R_in_cone = R_matrix[cone_mask]
                expected_in_cone = expected_num[cone_mask]
                
                dist_2d = np.broadcast_to(dist, (current_time_idx, self.Nx))
                dist_in_cone = dist_2d[cone_mask]
                
                L_chord = 2 * np.sqrt(R_in_cone**2 - dist_in_cone**2)
                w_eff = np.minimum(L_chord, circ_c)
                prob_coverage = w_eff / circ_c
                
                # Summation over both time and space inside the cone
                f_e[i_c] = np.sum(expected_in_cone * prob_coverage)
        f = 1.0 - np.exp(-f_e)
        return f

    def get_f_fraction_stochastic(self, cC, channel_radius, v_growth, J_nucl):
        """ Stochastic realization of the classicalBAG model
        """
        #J_nucl = get_nucleation_rate(cC, self.A, self.c_sat)
        expected_nuclei = J_nucl * self.dx * self.dt * (2.0 * np.pi * channel_radius) 
        actual_spawn_counts = np.random.poisson(expected_nuclei)
        spawn_indices = np.where(actual_spawn_counts > 0)[0]
        
        # birth
        for i in spawn_indices:
            count = actual_spawn_counts[i]
            if count > 0:
                x_rands = self.x_grid[i] + (np.random.rand(count) - 0.5) * self.dx
                y_rands = np.random.rand(count) * (2.0 * np.pi * channel_radius[i])
                self.nuclei_list.extend([[x, y, 0.0] for x, y in zip(x_rands, y_rands)])
                
        # growth
        for nucleus in self.nuclei_list:
            idx_center = int(np.clip(nucleus[0] / self.dx, 0, self.Nx - 1))
            nucleus[2] += v_growth[idx_center] * self.dt 
        dy_eval_target = self.dx / 2.0  
        # Calculate max points needed based on the LARGEST current radius
        max_circumference = 2.0 * np.pi * np.max(channel_radius)
        num_y = int(np.ceil(max_circumference / dy_eval_target))
        num_y = max(num_y, len(self.x_grid)) 
        theta_eval = np.linspace(0, 2.0 * np.pi, num_y)
        X_grid, Theta_grid = np.meshgrid(self.x_grid, theta_eval)
        Y_grid = Theta_grid * channel_radius  

        # Coverage %
        is_covered = np.zeros_like(X_grid, dtype=bool)
        for x_center, y_center, R in self.nuclei_list:
            if R > 0:
                idx_center = int(np.clip(x_center / self.dx, 0, self.Nx - 1))
                W = 2.0 * np.pi * channel_radius[idx_center]
                
                # bounding box in x-dir
                i_min = np.searchsorted(self.x_grid, x_center - R)
                i_max = np.searchsorted(self.x_grid, x_center + R, side='right')
                
                if i_min < i_max:
                    X_sub = X_grid[:, i_min:i_max]
                    Y_sub = Y_grid[:, i_min:i_max]
                    
                    dx_sq = (X_sub - x_center)**2
                    dy = np.abs(Y_sub - y_center)
                    dy_periodic = np.minimum(dy, W - dy)
                    dy_sq = dy_periodic**2
                    mask = (dx_sq + dy_sq) < R**2
                    is_covered[:, i_min:i_max][mask] = True
                
        # Average along the y-axis
        #rs = []
        #for nucleus in self.nuclei_list:
        #    rs.append(nucleus[2])
        #print(np.max(channel_radius), np.array(rs))
        return np.mean(is_covered, axis=0)
