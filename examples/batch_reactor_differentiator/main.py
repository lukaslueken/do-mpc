#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc
import do_mpc.differentiator as differentiator
# from do_mpc.differentiator._nlpdifferentiator import get_do_mpc_nlp_sol, build_sens_sym_struct, assign_num_to_sens_struct

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

import parametric_sensitivities as ps


""" User settings: """
show_animation = False
store_results = False

"""
Get configured do-mpc modules:
"""

model = template_model()
mpc = template_mpc(model)
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)

"""
Set initial state
"""

X_s_0 = 1.0 # This is the initial concentration inside the tank [mol/l]
S_s_0 = 0.5 # This is the controlled variable [mol/l]
P_s_0 = 0.0 #[C]
V_s_0 = 120.0 #[C]
x0 = np.array([X_s_0, S_s_0, P_s_0, V_s_0])


mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

mpc.set_initial_guess()

"""
Setup graphic:
"""

fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data, figsize=(8,5))
plt.ion()

"""
Run MPC main loop:
"""

# run stats
LICQ_status_list = []
SC_status_list = []
residuals_list = []
param_sens_list = []
track_nlp_obj = []
track_nlp_res = []


# nlp_diff = differentiator.NLPDifferentiator.from_optimizer(mpc)
# nlp_diff = differentiator.NLPDifferentiator(mpc)
nlp_diff = differentiator.DoMPCDifferentiatior(mpc)


import cProfile
import pstats

# with cProfile.Profile() as pr:
pr = cProfile.Profile()
pr.enable()

    

for k in range(10):
    

    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)    
    x0 = estimator.make_step(y_next)


    # get_do_mpc_nlp_sol
    # nlp_sol = nlp_diff._get_do_mpc_nlp_sol(mpc)
    # if nlp_diff.flags["reduced_nlp"]:
    #     nlp_sol_red = nlp_diff.reduce_nlp_solution_to_determined(nlp_sol)
    # else:
    #     nlp_sol_red = nlp_sol
    
    # p_num = nlp_sol_red["p"]
    
    # z_num, where_cons_active = nlp_diff._extract_active_primal_dual_solution(nlp_sol_red, tol=1e-6,set_lam_zero=False)
    # z_num, where_cons_active = nlp_diff.extract_active_primal_dual_solution(nlp_sol_red, method_active_set="primal", primal_tol=1e-6,dual_tol=1e-12)

    track_nlp_obj.append(nlp_diff.nlp.copy())
    track_nlp_res.append(nlp_diff._get_do_mpc_nlp_sol(nlp_diff.optimizer).copy())

    tic = time.time()
    print("iteration: ", k)
    # param_sens, residues, LICQ_status = nlp_diff.calculate_sensitivities(z_num, p_num, where_cons_active, check_rank=True, track_residues=True, lstsq_fallback=True)
    # param_sens, residues, LICQ_status = nlp_diff.calculate_sensitivities(z_num, p_num, where_cons_active, check_rank=True, track_residues=True, lstsq_fallback=True)
    dx_dp_num, dlam_dp_num, residuals, LICQ_status, SC_status, where_cons_active = nlp_diff.differentiate()
    toc = time.time()
    print("Time to calculate sensitivities: ", toc-tic)
    assert LICQ_status==True
    assert residuals<=1e-12
    # assert k<5

    LICQ_status_list.append(LICQ_status)
    SC_status_list.append(SC_status)
    residuals_list.append(residuals)
    param_sens_list.append(dx_dp_num)

    # sens_struct = differentiator.build_sens_sym_struct(mpc)    
    # sens_num = differentiator.assign_num_to_sens_struct(sens_struct,dx_dp_num,nlp_diff.undet_sym_idx_dict)

    if True:
        nlp_dict, nlp_bounds = ps.get_do_mpc_nlp(mpc)
        nlp_sol = ps.get_do_mpc_nlp_sol(mpc)
        nlp_dict_red, nlp_bounds_red, nlp_sol_red, det_idx_dict, undet_idx_dict = ps.reduce_nlp_to_determined(nlp_dict, nlp_bounds, nlp_sol)
        
        idx_x_determined, idx_p_determined = det_idx_dict["opt_x"], det_idx_dict["opt_p"]

        dxdp_num_alt_zeros = np.zeros((nlp_dict["x"].shape[0],nlp_dict["p"].shape[0]))

        dxdp_num_alt_full = dxdp_num_alt_zeros.copy()
        dxdp_num_alt_full[idx_x_determined[:,None],idx_p_determined] = ps.solve_nlp_sens(nlp_dict_red, nlp_bounds_red, nlp_sol_red, nlp_sol_red["p"], mode="full")
        
        dxdp_num_alt_as = dxdp_num_alt_zeros.copy()
        # dxdp_num_alt_as = ps.solve_nlp_sens(nlp_dict_red, nlp_bounds_red, nlp_sol_red, nlp_sol_red["p"], mode="active-set")
        dxdp_num_alt_as[idx_x_determined[:,None],idx_p_determined] = ps.solve_nlp_sens(nlp_dict_red, nlp_bounds_red, nlp_sol_red, nlp_sol_red["p"], mode="active-set")

    # rec_nlp = ps.reconstruct_nlp(nlp_standard_full_dict)
    if True:
        rec_nlp = ps.reconstruct_nlp(nlp_diff.nlp_unreduced)
        S = nlpsol("S", "ipopt", rec_nlp)    
        # eval_dict = ps.validate_fd(dx_dp_num, S, nlp_diff.nlp_bounds_unreduced, nlp_sol["p"], nlp_sol["x"], n_eval= 10, step_size= 1e-3)
        abs_env_error = ps.validate_env_theorem(dxdp_num_alt_as, vertcat(nlp_sol["x"],nlp_sol["lam_g"]), nlp_sol["p"], rec_nlp, return_matrices=False)
        # abs_env_error = ps.validate_env_theorem(dx_dp_num, vertcat(nlp_sol["x"],nlp_sol["lam_g"]), nlp_sol["p"], rec_nlp, return_matrices=False)
        assert abs_env_error.max()<1e-8
    
    assert np.abs(dxdp_num_alt_full - dxdp_num_alt_as).max()<1e-6
    assert np.abs(dx_dp_num - dxdp_num_alt_as).max()<1e-6
    assert np.abs(dx_dp_num - dxdp_num_alt_full).max()<1e-6    
    

    if show_animation:
        graphics.plot_results(t_ind=k)
        graphics.plot_predictions(t_ind=k)
        graphics.reset_axes()
        plt.show()
        plt.pause(0.01)

# input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results([mpc, simulator], 'batch_reactor_MPC')


pr.disable()

stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.TIME)
stats.sort_stats("tottime")
stats.print_stats()
# dump stats to readable file
stats.dump_stats("profile_stats_Sens.prof")