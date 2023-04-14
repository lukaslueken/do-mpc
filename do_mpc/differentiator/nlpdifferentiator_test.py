from _nlpdifferentiator import NLPDifferentiator#, setup_NLP_example_1, validate_fd
import casadi as ca
import numpy as np


def validate_fd(sens_vals, nlp_solver, nlp_bounds, p_num, x0, n_eval = 10, step_size = 1e-3):
    
    # define function to solve nlp for various bounds
    def solve_nlp(S,nlp_bounds, p_num, x0):
        # solve NLP
        if "lbx" in nlp_bounds.keys() and "ubx" in nlp_bounds.keys():
            sol = S(x0=x0, p=p_num, lbx=nlp_bounds["lbx"], ubx=nlp_bounds["ubx"], lbg=nlp_bounds["lbg"], ubg=nlp_bounds["ubg"])
        else:
            sol = S(x0=x0, p=p_num, lbg=nlp_bounds["lbg"], ubg=nlp_bounds["ubg"])
        return sol
    
    n_p = p_num.shape[0]
    n_x = x0.shape[0]
    
    param_sens = sens_vals[0:n_x,0:n_p]
    
    dp_unscaled = 2*(np.random.rand(n_p,n_eval)-0.5)
    dp_len = np.linalg.norm(dp_unscaled,axis=0)
    dp = step_size*dp_unscaled/dp_len
    
    p_fd = p_num.reshape((-1,1))+dp    
    dopt = param_sens@dp
    
    r_ca_old = solve_nlp(nlp_solver, nlp_bounds, p_num, x0)
    r_ca_new = solve_nlp(nlp_solver, nlp_bounds, p_fd, r_ca_old["x"]+dopt)
    # r_ca_new = solve_nlp(nlp_solver, nlp_bounds, p_fd, x0)
    
    x_old = np.array(r_ca_old["x"])
    x_new = np.array(r_ca_new["x"])
    
    diff_x = x_new - x_old
    abs_dev = np.abs(diff_x - dopt)

    eval_dict = {}
    eval_dict["max"] = np.max(abs_dev)
    eval_dict["rel_max"] = np.max(abs_dev)/step_size
    eval_dict["mean"] = np.mean(abs_dev)
    eval_dict["rel_mean"] = np.mean(abs_dev)/step_size
    eval_dict["std"] = np.std(abs_dev)
    eval_dict["rel_std"] = np.std(abs_dev)/step_size
    
    return eval_dict
    
def setup_NLP_example_1():
    # build NLP
    # https://web.casadi.org/blog/nlp_sens/

    nlp_id = "casadi_nlp_sens_adapted"
    
    ## Decision Variables
    x_sym = ca.SX.sym('x',2,1)

    ## Parameters
    p_sym = ca.SX.sym('p',2,1)

    ## Objective Function
    f_sym = (p_sym[0] - x_sym[0])**2 + 0.2*(x_sym[1] - x_sym[0]**2)**2

    ## Constraint Functions
    # ubg = 0 (standard form)

    g_0 = (p_sym[1]**2)/4 - (x_sym[0]+0.5)**2 + x_sym[1]**2
    g_1 = (x_sym[0]+0.5)**2 + x_sym[1]**2- p_sym[1]**2
    g_2 = (x_sym[0]+0.5)

    # concat constraints
    g_sym = ca.vertcat(g_0,g_1,g_2)

    ## setup NLP
    nlp = {'x':x_sym, 'p':p_sym, 'f':f_sym, 'g':g_sym}


    ## setup Bounds
    lbg = -np.inf*np.ones(g_sym.shape)
    ubg = np.zeros(g_sym.shape)
    # ubg[-1] = 1.2
    # lbg[-1] = 1.0

    # lbx = np.zeros(x_sym.shape)
    lbx = -np.inf*np.ones(x_sym.shape)
    ubx = np.inf*np.ones(x_sym.shape)

    # lbx[0] = 0.0
    # ubx[0] = 0.0

    nlp_bounds = {"lbg": lbg,"ubg": ubg,"lbx": lbx,"ubx": ubx}
    
    return nlp, nlp_bounds, nlp_id

# setup NLP
nlp, nlp_bounds, nlp_id = setup_NLP_example_1()
nlp_dict = {"nlp": nlp, "nlp_bounds": nlp_bounds}

# instantiate NLPDifferentiator
nlp_diff = NLPDifferentiator(nlp_dict)

nlp_diff.settings.check_LICQ = True

print(nlp_diff.settings)

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
p_num = np.array((0,1))
nlp_sol = nlp_solver(x0=0, p=p_num, **nlp_bounds)

# get_do_mpc_nlp_sol
if nlp_diff.flags["reduced_nlp"]:
    nlp_sol_red = nlp_diff.reduce_nlp_solution_to_determined(nlp_sol)
else:
    nlp_sol_red = nlp_sol
z_num, where_cons_active = nlp_diff.extract_active_primal_dual_solution(nlp_sol_red, method_active_set="primal")
param_sens, residues, LICQ_status = nlp_diff.calculate_sensitivities(z_num, p_num, where_cons_active, check_rank=True, track_residues=True, lstsq_fallback=True)
dx_dp_num, dlam_dp_num = nlp_diff.map_param_sens(param_sens, where_cons_active)

if True:
    eval_dict = validate_fd(dx_dp_num, nlp_solver, nlp_bounds, p_num, x0=nlp_sol["x"], n_eval = 10, step_size = 1e-3)
    print(eval_dict)