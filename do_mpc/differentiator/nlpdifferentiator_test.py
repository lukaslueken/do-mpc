from _nlpdifferentiator import NLPDifferentiator, setup_NLP_example_1, validate_fd
import casadi as ca
import numpy as np

# setup NLP
nlp, nlp_bounds, nlp_id = setup_NLP_example_1()
nlp_dict = {"nlp": nlp, "nlp_bounds": nlp_bounds}

# instantiate NLPDifferentiator
nlp_diff = NLPDifferentiator(nlp_dict)

# specify solver
def specify_solver(nlp):
    nlp_sol_opts = {}
    ipopt_options = {"fixed_variable_treatment": "make_constraint"}
    nlp_sol_opts["expand"] = False
    ipopt_options["print_level"] = 4
    nlp_sol_opts["ipopt"] = ipopt_options
    # nlp_sol_opts["expand"] = False

    nlp_solver = ca.nlpsol('S', 'ipopt', nlp,nlp_sol_opts) #,nlpsol_options=ipopt_options
    return nlp_solver

nlp_solver = specify_solver(nlp)

# solve NLP
p_num = np.array((1,1))
nlp_sol = nlp_solver(x0=0, p=p_num, **nlp_bounds)

# get_do_mpc_nlp_sol
if nlp_diff.flags["reduced_nlp"]:
    nlp_sol_red = nlp_diff.reduce_nlp_solution_to_determined(nlp_sol)
else:
    nlp_sol_red = nlp_sol
z_num, where_cons_active = nlp_diff.extract_active_primal_dual_solution(nlp_sol_red, method_active_set="primal")
param_sens, residues, LICQ_status = nlp_diff.calculate_sensitivities(z_num, p_num, where_cons_active, lin_solver="scipy", check_LICQ=False, check_rank=False, track_residues=True, lstsq_fallback=True)
dx_dp_num, dlam_dp_num = nlp_diff.map_param_sens(param_sens, where_cons_active)

if True:
    eval_dict = validate_fd(dx_dp_num, nlp_solver, nlp_bounds, p_num, x0=nlp_sol["x"], n_eval = 10, step_size = 1e-3)
    print(eval_dict)