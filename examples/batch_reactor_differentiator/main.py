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
show_animation = True
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
residues_list = []
param_sens_list = []


nlp_diff = differentiator.NLPDifferentiator.from_optimizer(mpc)
# nlp_diff = differentiator.NLPDifferentiator(mpc)

# import pdb
# pdb.set_trace()
for k in range(10):
    
   
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)    
    x0 = estimator.make_step(y_next)


    # get_do_mpc_nlp_sol
    nlp_sol = differentiator.get_do_mpc_nlp_sol(mpc)
    if nlp_diff.flags["reduced_nlp"]:
        nlp_sol_red = nlp_diff.reduce_nlp_solution_to_determined(nlp_sol)
    else:
        nlp_sol_red = nlp_sol
    
    p_num = nlp_sol_red["p"]
    
    z_num, where_cons_active = nlp_diff.extract_active_primal_dual_solution(nlp_sol_red, method_active_set="primal", primal_tol=1e-6,dual_tol=1e-12)
    tic = time.time()
    print("iteration: ", k)
    param_sens, residues, LICQ_status = nlp_diff.calculate_sensitivities(z_num, p_num, where_cons_active, check_rank=True, track_residues=True, lstsq_fallback=True)
    toc = time.time()
    print("Time to calculate sensitivities: ", toc-tic)
    # assert k<87
    # assert residues<=1e-12
    dx_dp_num, dlam_dp_num = nlp_diff.map_param_sens(param_sens, where_cons_active)

    LICQ_status_list.append(LICQ_status)
    residues_list.append(residues)
    param_sens_list.append(param_sens)

    # sens_struct = differentiator.build_sens_sym_struct(mpc)    
    # sens_num = differentiator.assign_num_to_sens_struct(sens_struct,dx_dp_num,nlp_diff.undet_sym_idx_dict)


    # nlp_dict, nlp_bounds = ps.get_do_mpc_nlp(mpc)
    # nlp_sol = ps.get_do_mpc_nlp_sol(mpc)
    # nlp_dict_red, nlp_bounds_red, nlp_sol_red, det_idx_dict, undet_idx_dict = ps.reduce_nlp_to_determined(nlp_dict, nlp_bounds, nlp_sol)
    
    # dxdp_num_alt = ps.solve_nlp_sens(nlp_dict_red, nlp_bounds_red, nlp_sol_red, nlp_sol_red["p"], mode="full")
    
    # assert k<81
    # assert np.abs(dx_dp_num-dxdp_num_alt).max()<1e-10
    # print(sens_num["dxdp", indexf["_x",0,0,-1], indexf["_x0"]])
    
    # dudx = sens_num["dxdp", indexf["_u",0,0], indexf["_x0"]]
    
    

    if show_animation:
        graphics.plot_results(t_ind=k)
        graphics.plot_predictions(t_ind=k)
        graphics.reset_axes()
        plt.show()
        plt.pause(0.01)

input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results([mpc, simulator], 'batch_reactor_MPC')