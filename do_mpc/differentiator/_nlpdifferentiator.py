import numpy as np
import scipy as sp
from casadi import *
from casadi.tools import *


from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

class NLPDifferentiator:
    """
    Documentation for NLPDifferentiator.
 
    .. warning::

        This tool is currently not fully implemented and cannot be used.
    """

    def __init__(self, nlp_container):
        
        ## Setup
        self._setup_nlp(nlp_container)

        # TODO: refactor flags
        self.flags = {}
        self.flags['sym_KKT_system'] = False
        self.flags['reduced_nlp'] = False

        ## Preparation
        self._prepare_differentiator()
                

    ### SETUP
    def _setup_nlp(self,nlp_container):
        #TODO: rewrite initialization to be more streamlined
        #TODO: check whether mpc using scaling
        if type(nlp_container)==dict:
            self.nlp, self.nlp_bounds = nlp_container["nlp"].copy(), nlp_container["nlp_bounds"].copy()
        elif hasattr(nlp_container,"opt_x"):
            nlp, nlp_bounds = self._get_do_mpc_nlp(nlp_container)
            self.nlp, self.nlp_bounds = nlp.copy(), nlp_bounds.copy()
        else:
            raise ValueError('nlp_container must be a tuple or a do_mpc object.')

    ### PREPARATION
    def _prepare_differentiator(self):
        # 1. Detect undetermined symbolic variables and reduce NLP
        # if self.flags['reduced_nlp']:
        self._remove_unused_sym_vars()

        # 2. Get size metrics
        self._get_size_metrics()

        # 3. Get symbolic expressions for lagrange multipliers
        self._get_sym_lagrange_multipliers()
        self._stack_primal_dual()

        # 4. Get symbolic expressions for Lagrangian
        self._get_Lagrangian_sym()
        
        # 5. Get symbolic expressions for sensitivity matrices
        self._prepare_sensitivity_matrices()

        # 6. Get LDLT factorization
        # self._get_regularized_LDL_factorization()
        
    def _get_do_mpc_nlp(self,mpc_object):
        """
        This function is used to extract the symbolic expressions and bounds of the underlying NLP of the MPC.
        It is used to initialize the NLPDifferentiator class.
        """

        # 1 get symbolic expressions of NLP
        nlp = {'x': vertcat(mpc_object.opt_x), 'f': mpc_object.nlp_obj, 'g': mpc_object.nlp_cons, 'p': vertcat(mpc_object.opt_p)}

        # 2 extract bounds
        nlp_bounds = {}
        nlp_bounds['lbg'] = mpc_object.nlp_cons_lb.full()#.reshape(-1,1)
        nlp_bounds['ubg'] = mpc_object.nlp_cons_ub.full()#.reshape(-1,1)
        nlp_bounds['lbx'] = vertcat(mpc_object._lb_opt_x).full()#.reshape(-1,1)
        nlp_bounds['ubx'] = vertcat(mpc_object._ub_opt_x).full()#.reshape(-1,1)

        return nlp, nlp_bounds
    
    def _detect_undetermined_sym_var(self, var="x"):
        
        # symbolic expressions
        var_sym = self.nlp[var]        
        # objective function
        f_sym = self.nlp["f"]
        # constraints
        g_sym = self.nlp["g"]

        # boolean expressions on wether a symbolic is contained in the objective function f or the constraints g
        map_f_var = map(lambda x: depends_on(f_sym,x),vertsplit(var_sym))
        map_g_var = map(lambda x: depends_on(g_sym,x),vertsplit(var_sym))

        # combined boolean expressions as list for each symbolic variable in var_sym
        dep_list = [f_dep or g_dep for f_dep,g_dep in zip(map_f_var,map_g_var)]

        # indices of undetermined and determined symbolic variables
        undet_sym_idx = np.where(np.logical_not(dep_list))[0]
        det_sym_idx = np.where(dep_list)[0]

        # example:
        # if undet_sym_idx = [1,3], then the second and fourth symbolic variable in var_sym are undetermined
                
        return undet_sym_idx,det_sym_idx

    def _remove_unused_sym_vars(self):
        """
        Reduces the NLP by removing symbolic variables for x and p that are not contained in the objective function or the constraints.

        """
        # detect undetermined symbolic variables
        undet_opt_x_idx, det_opt_x_idx = self._detect_undetermined_sym_var("x")
        undet_opt_p_idx, det_opt_p_idx = self._detect_undetermined_sym_var("p")
        
        # copy nlp and nlp_bounds
        nlp_red = self.nlp.copy()
        nlp_bounds_red = self.nlp_bounds.copy()

        # adapt nlp
        nlp_red["x"] = self.nlp["x"][det_opt_x_idx]
        nlp_red["p"] = self.nlp["p"][det_opt_p_idx]

        # adapt nlp_bounds
        nlp_bounds_red["lbx"] = self.nlp_bounds["lbx"][det_opt_x_idx]
        nlp_bounds_red["ubx"] = self.nlp_bounds["ubx"][det_opt_x_idx]

        det_sym_idx_dict = {"opt_x":det_opt_x_idx, "opt_p":det_opt_p_idx}
        undet_sym_idx_dict = {"opt_x":undet_opt_x_idx, "opt_p":undet_opt_p_idx}

        N_vars_to_remove = len(undet_sym_idx_dict["opt_x"])+len(undet_sym_idx_dict["opt_p"])
        if N_vars_to_remove > 0:
            self.nlp_unreduced, self.nlp_bounds_unreduced = self.nlp, self.nlp_bounds
            self.nlp, self.nlp_bounds = nlp_red, nlp_bounds_red
            self.det_sym_idx_dict, self.undet_sym_idx_dict = det_sym_idx_dict, undet_sym_idx_dict
            self.flags["reduced_nlp"] = True
        else:
            self.flags["reduced_nlp"] = False
            print("NLP formulation does not contain unused variables.")

    def _get_size_metrics(self):
        """
        Specifies the number of decision variables, nonlinear constraints and parameters of the NLP.
        """
        self.n_x = self.nlp["x"].shape[0]
        self.n_g = self.nlp["g"].shape[0]
        self.n_p = self.nlp["p"].shape[0]

    def _get_sym_lagrange_multipliers(self):
        self.nlp["lam_g"] = SX.sym("lam_g",self.n_g,1)
        self.nlp["lam_x"] = SX.sym("lam_x",self.n_x,1)
        self.nlp["lam"] = vertcat(self.nlp["lam_g"],self.nlp["lam_x"])

    def _stack_primal_dual(self):
        self.nlp["z"] = vertcat(self.nlp["x"],self.nlp["lam"])

    def _get_Lagrangian_sym(self):
        """
        Returns the Lagrangian of the NLP for sensitivity calculation.
        Attention: It is not verified, whether the NLP is in standard form. 

        """
        # TODO: verify if NLP is in standard form to simplify further evaluations
        self.L_sym = self.nlp["f"] + self.nlp['lam_g'].T @ self.nlp['g'] + self.nlp['lam_x'].T @ self.nlp['x']
        # self.flags['get_Lagrangian'] = True

    def _get_A_matrix(self):
        self.A_sym = hessian(self.L_sym,self.nlp["z"])[0]
        self.A_func = Function("A", [self.nlp["z"],self.nlp["p"]], [self.A_sym], ["z_opt", "p_opt"], ["A"])

    def _get_B_matrix(self):
        self.B_sym = jacobian(gradient(self.L_sym,self.nlp["z"]),self.nlp["p"])
        self.B_func = Function("B", [self.nlp["z"],self.nlp["p"]], [self.B_sym], ["z_opt", "p_opt"], ["B"])

    def _prepare_sensitivity_matrices(self):
        # TODO: move to _get_A_matrix and _get_B_matrix
        self._get_A_matrix()
        self._get_B_matrix()
        self.flags['sym_KKT_system'] = True

    def _get_regularized_LDL_factorization(self):
        """
        Returns the regularized LDL factorization of the Hessian of the Lagrangian.
        """
        # TODO: remove (use sparse QR factorization instead)
                
        # generate symbolic variable alpha, which tracks, whether constraints are active or inactive (alpha = 1 if active, alpha = 0 if inactive)
        self.alpha = SX.sym("alpha", self.n_g+self.n_x, 1)
        
        # symbolic regularization factor for diagonal elements of Hessian
        self.rho = SX.sym("rho", 1, 1)

        # generate identity matrix of size n_x+n_g+n_x (n_x for primal variables, n_g for dual variables of nonlinear constraints, n_x for dual variables of linear constraints)
        I_alpha_mult = SX.eye(self.n_x+self.n_g+self.n_x)

        # set diagonal values of n_g+n_x to alpha values
        I_alpha_mult[self.n_x:self.n_x+self.n_g+self.n_x,self.n_x:self.n_x+self.n_g+self.n_x] = diag(self.alpha)

        # generate identity matrix of size n_x+n_g+n_x (n_x for primal variables, n_g for dual variables of nonlinear constraints, n_x for dual variables of linear constraints)
        I_alpha_add = SX.eye(self.n_x+self.n_g+self.n_x)

        # set diagonal values of n_g+n_x to (1-alpha) values
        I_alpha_add[self.n_x:self.n_x+self.n_g+self.n_x,self.n_x:self.n_x+self.n_g+self.n_x] = diag(1-self.alpha)

        # generate regularization matrix
        I_rho_add = self.rho*SX.eye(self.n_x+self.n_g+self.n_x)

        # generate adapted A and B matrices
        self.A_adapted_sym = self.A_sym @ I_alpha_mult + I_alpha_add + I_rho_add
        self.B_adapted_sym = I_alpha_mult @ self.B_sym

        # generate LDL factorization of adapted A matrix
        self.D_sym, self.LT_sym, self.LDL_permutation = ldl(self.A_adapted_sym)

        # add identity to self.LT_sym
        # self.LT_sym = self.LT_sym + SX.eye(self.n_x+self.n_g+self.n_x)

        # set flag for ldl factorization
        self.flags['get_ldl'] = True


        # get symbolic solution of A*x = b with LDL factorization
        # self.x_ldl_sym = ldl_solve(self.B_adapted_sym, self.D_sym, self.LT_sym, self.LDL_permutation)

        # self.ls_sol_reg_ldl_func = Function("dzdp", [self.nlp["z"],self.nlp["p"], self.alpha, self.rho], [self.x_ldl_sym], ["z_opt", "p_opt", "alpha", "rho"], ["dzdp"])
        
        self.ldl_func = Function("ldl_matrices", [self.nlp["z"],self.nlp["p"], self.alpha, self.rho], [self.B_adapted_sym, self.D_sym, self.LT_sym, self.LDL_permutation], ["z_opt", "p_opt", "alpha", "rho"], ["B_adapted_sym", "D_sym", "LT_sym", "LDL_permutation"])
        # self.A_adapted_func = Function("A_adapted", [self.nlp["z"],self.nlp["p"], self.alpha, self.rho], [self.A_adapted_sym], ["z_opt", "p_opt", "alpha", "rho"], ["A_adapted"])

    ### ALGORITHM
    def get_do_mpc_nlp_sol(self,mpc):
        nlp_sol = {}
        nlp_sol["x"] = vertcat(mpc.opt_x_num)
        nlp_sol["x_unscaled"] = vertcat(mpc.opt_x_num_unscaled)
        nlp_sol["g"] = vertcat(mpc.opt_g_num)
        nlp_sol["lam_g"] = vertcat(mpc.lam_g_num)
        nlp_sol["lam_x"] = vertcat(mpc.lam_x_num)
        nlp_sol["p"] = vertcat(mpc.opt_p_num)
        return nlp_sol
    
    def reduce_nlp_solution_to_determined(self,nlp_sol):
        assert self.flags["reduced_nlp"], "NLP is not reduced."

        # adapt nlp_sol
        nlp_sol_red = nlp_sol.copy()
        nlp_sol_red["x"] = nlp_sol["x"][self.det_sym_idx_dict["opt_x"]]
        nlp_sol_red["lam_x"] = nlp_sol["lam_x"][self.det_sym_idx_dict["opt_x"]] 
        nlp_sol_red["p"] = nlp_sol["p"][self.det_sym_idx_dict["opt_p"]]
        
        # backwards compatilibity TODO: remove
        if "x_unscaled" in nlp_sol:
            nlp_sol_red["x_unscaled"] = nlp_sol["x_unscaled"][self.det_sym_idx_dict["opt_x"]]

        return nlp_sol_red

    def extract_active_primal_dual_solution(self, nlp_sol, method_active_set="dual", tol=1e-6):
        """
        This function extracts the active primal and dual solution from the NLP solution and stackes it into a single vector. The active set is determined by the "primal" or "dual" solution.
        Args:
            nlp_sol: dict containing the NLP solution.
            method_active_set: str, either "primal" or "dual". Determines the active set by the primal or dual solution.

        Returns:
            z_num: numpy array containing the active primal and dual solution.
            where_lam_not_zero: numpy array containing the indices of the active constraints.

        Raises:
            KeyError: If the NLP solution does not contain the primal or dual solution.        
        """
        if "x" not in nlp_sol.keys():
            raise KeyError("NLP solution does not contain primal solution.")
        if "lam_g" not in nlp_sol.keys():
            raise KeyError("NLP solution does not contain dual solution to nonlinear constraints.")
        if "lam_x" not in nlp_sol.keys():
            raise KeyError("NLP solution does not contain dual solution to linear constraints.")
        if "g" not in nlp_sol.keys():
            raise KeyError("NLP solution does not contain nonlinear constraints.")
        
        x_num = nlp_sol["x"]
        lam_num = vertcat(nlp_sol["lam_g"],nlp_sol["lam_x"])

        ## determine active set
        if method_active_set == "primal":
            g_num = nlp_sol["g"]
            lbg = self.nlp["lbg"]
            ubg = self.nlp["ubg"]
            lbx = self.nlp["lbx"]
            ubx = self.nlp["ubx"]

            g_delta_lbg = g_num - lbg
            g_delta_ubg = g_num - ubg
            x_delta_lbx = x_num - lbx
            x_delta_ubx = x_num - ubx

            where_g_inactive = np.where(np.abs(g_delta_lbg)>tol & np.abs(g_delta_ubg)>tol)[0]
            where_x_inactive = np.where(np.abs(x_delta_lbx)>tol & np.abs(x_delta_ubx)>tol)[0]            
            where_g_active = np.where(np.abs(g_delta_lbg)<=tol | np.abs(g_delta_ubg)<=tol)[0]
            where_x_active = np.where(np.abs(x_delta_lbx)<=tol | np.abs(x_delta_ubx)<=tol)[0]

            where_cons_active = np.concatenate((where_g_active,where_x_active+self.n_g))
            where_cons_inactive = np.concatenate((where_g_inactive,where_x_inactive+self.n_g))
        
        elif method_active_set == "dual":
            where_cons_active = np.where(np.abs(lam_num)>tol)[0]
            where_cons_inactive = np.where(np.abs(lam_num)<=tol)[0]
        else:
            raise ValueError("Unknown method for determining active set.")

        # set lagrange multipliers of inactive constraints to zero
        lam_num[where_cons_inactive] = 0
        
        # stack primal and dual solution
        z_num = vertcat(x_num,lam_num)

        return z_num, where_cons_active


    def calculate_sensitivities(self, z_num, p_num, where_lam_not_zero, lin_solver="scipy", check_rank=False, track_residues=False):
        """
        Calculates the sensitivities of the NLP solution.
        Args:
            nlp_sol: dict containing the NLP solution.
            method_active_set: str, either "primal" or "dual". Determines the active set by the primal or dual solution.
            tol: float, tolerance for determining the active set.
        Returns:
        """
        
        A_num, B_num = self._get_sensitivity_matrices(z_num, p_num)
        A_num, B_num = self._reduce_sensitivity_matrices(A_num, B_num, where_lam_not_zero)
        if check_rank:
            self._check_rank(A_num)

        # solve LSE to get parametric sensitivities
        param_sens = self.solve_linear_system(A_num,B_num, lin_solver=lin_solver)

        if track_residues:
            residues = self._track_residues(A_num, B_num)
            return param_sens, residues
        else:
            return param_sens, None
    
    def _get_sensitivity_matrices(self, z_num, p_num):
        """
        Returns the sensitivity matrix A and the sensitivity vector B of the NLP.
        """
        if self.flags['sym_KKT_system'] is False:
            raise RuntimeError('No symbolic expression for sensitivitiy system computed yet.')        
        A_num = self.A_func(z_num, p_num)
        B_num = self.B_func(z_num, p_num)
        return A_num, B_num
    
    def _reduce_sensitivity_matrices(self, A_num, B_num, where_cons_active):
        """
        Reduces the sensitivity matrix A and the sensitivity vector B of the NLP such that only the rows and columns corresponding to non-zero dual variables are kept.
        """
        where_keep_idx = [i for i in range(self.n_x)]+list(where_cons_active+self.n_x)
        A_num = A_num[where_keep_idx,where_keep_idx].full().copy()
        B_num = B_num[where_keep_idx,:].full().copy() #TODO: remove .full()
        return A_num, B_num
    
    def _check_rank(self, A_num):
        """
        Checks if the sensitivity matrix A has full rank.
        """
        if np.linalg.matrix_rank(A_num) < A_num.shape[0]:
            raise KeyError("Sensitivity matrix A does not have full rank.")
    
    def _solve_linear_system(self,A_num,B_num, lin_solver="scipy"):
        """
        Solves the linear system of equations to calculate parametric sensitivities.
        Args:
            A_num: Numeric A-Matrix (dF/dz).
            B_num: Numeric B-Matrix (dF/dp).
            lin_solver: Linear solver to use. Options are "scipy", "casadi" and "lstq".
        Returns:
            parametric sensitivities (n_x,n_p)
        """
        if lin_solver == "scipy":
            param_sens = sp.linalg.solve(A_num, -B_num, assume_a="sym")
        elif lin_solver == "casadi":
            param_sens = solve(A_num, -B_num)
        elif lin_solver == "lstq":
            param_sens = np.linalg.lstsq(A_num, -B_num, rcond=None)[0]
        else:
            raise ValueError("Unknown linear solver.")
        return param_sens
    
    def _track_residures(self, A_num, B_num, param_sens):
        """
        Tracks the residues of the linear system of equations.
        """
        residues = np.linalg.norm(A_num.dot(param_sens)+B_num, ord=2)
        return residues

    # def solve_linear_system(self,A_num,B_num, verbose=False, track_residues=False):
    #     """Function to solve linear system of equations to calculate parametric sensitivities.

    #     Args:
    #         A_num: Numeric A-Matrix (dF/dz).
    #         B_num: Numeric B-Matrix (dF/dp).

    #     Returns:
    #         parametric sensitivities (n_x,n_p)

    #     Raises:
    #         KeyError: A-Matrix does not have full rank (i.e. is not invertible).
    #     """
    #     # 0. Solve sensitivity system for parametrics sensitivities
    #     try:
    #         # reg_id = 1e-4*np.eye(A_num.shape[0], dtype=np.float64)
    #         # param_sens = sp.linalg.solve(A_num+reg_id, -B_num, assume_a="sym")
    #         # TODO: use sparse solver
    #         # param_sens = sp.linalg.solve(A_num, -B_num, assume_a="sym")
    #         param_sens = solve(A_num, -B_num)
    #         # param_sens = np.linalg.solve(A_num, -B_num)
    #         LSE_method = np.array(["Linear Solver"])
    #         sens_matrix_rank = A_num.shape[0]
    #     except np.linalg.LinAlgError:
    #         param_sens, residues, sens_matrix_rank, s = np.linalg.lstsq(A_num, -B_num)
    #         LSE_method = np.array(["Least Squares"])
    #         print("LinAlgError suppressed, try least squares solution.")
        
    #     if track_residues:
    #         sens_residues = np.linalg.norm(A_num @ param_sens + B_num, ord = "fro")

    #     if verbose:
    #         print("LSE method: ", LSE_method)
    #         if track_residues:
    #             print("LSE residues: ", sens_residues)

    #     if sens_matrix_rank != A_num.shape[0]:
    #         raise KeyError("LSE rank not equal to A matrix rank")
    #     return param_sens
    
    @timeit
    def get_sensitivities(self, z_num, p_num, where_lam_not_zero, verbose=False, track_residues=False):
        """
        Returns the parametric sensitivities of the NLP.
        """
        A_num, B_num = self._get_sensitivity_matrices(z_num, p_num)
        A_num, B_num = self._reduce_sensitivity_matrices(A_num, B_num, where_lam_not_zero)
        param_sens = self.solve_linear_system(A_num,B_num, verbose=verbose, track_residues=track_residues)
        return param_sens
    
    def map_dxdp(self,param_sens):
        """
        Maps the parametric sensitivities to the original decision variables.
        """
        dx_dp = param_sens[:self.n_x,:]
        return dx_dp
    
    def map_dlamdp(self,param_sens, where_lam_not_zero):
        """
        Maps the parametric sensitivities to the original sensitivities of the lagrange multipliers.
        """
        dlam_dp = np.zeros((self.n_g+self.n_x,self.n_p))
        assert len(where_lam_not_zero) == param_sens.shape[0]-self.n_x, "Number of non-zero dual variables does not match number of parametric sensitivities for lagrange multipliers."
        dlam_dp[where_lam_not_zero,:] = param_sens[self.n_x:,:]
        return dlam_dp

    # TODO: move to separate class for handling do-mpc solution
    def build_sens_sym_struct(self,mpc):
        opt_x = mpc._opt_x
        opt_p = mpc._opt_p
        
        sens_struct = struct_symSX([
            entry("dxdp",shapestruct=(opt_x,opt_p)),
        ])

        return sens_struct

    def assign_num_to_sens_struct(self,sens_struct,dxdp_num,undet_sym_idx_dict):

        dxdp_init = dxdp_num#.copy()
        ins_idx_x = [val-idx for idx, val in enumerate(undet_sym_idx_dict["opt_x"])] # used for inserting zero rows in dxdp_init
        ins_idx_p = [val-idx for idx, val in enumerate(undet_sym_idx_dict["opt_p"])] # used for inserting zero columns in dxdp_init
        
        dxdp_init = np.insert(dxdp_init, ins_idx_x, 0.0, axis=0)
        dxdp_init = np.insert(dxdp_init, ins_idx_p, 0.0, axis=1)
        
        assert dxdp_init.shape == sens_struct["dxdp"].shape
        
        sens_num = sens_struct(0)
        
        sens_num["dxdp"] = dxdp_init

        return sens_num

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
    x_sym = SX.sym('x',2,1)

    ## Parameters
    p_sym = SX.sym('p',2,1)

    ## Objective Function
    f_sym = (p_sym[0] - x_sym[0])**2 + 0.2*(x_sym[1] - x_sym[0]**2)**2

    ## Constraint Functions
    # ubg = 0 (standard form)

    g_0 = (p_sym[1]**2)/4 - (x_sym[0]+0.5)**2 + x_sym[1]**2
    g_1 = (x_sym[0]+0.5)**2 + x_sym[1]**2- p_sym[1]**2
    g_2 = (x_sym[0]+0.5)

    # concat constraints
    g_sym = vertcat(g_0,g_1,g_2)

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


def reconstruct_nlp(nlp_standard_full_dict):
    # TODO: get bounds or remove
    
    # 1. create full nlp
    f_sym = nlp_standard_full_dict["f"]
    x_sym = nlp_standard_full_dict["x"]
    p_sym = nlp_standard_full_dict["p"]
    g_sym_list = []
    if "g" in nlp_standard_full_dict.keys():
        g_sym_list.append(nlp_standard_full_dict["g"])
    if "h" in nlp_standard_full_dict.keys():
        g_sym_list.append(nlp_standard_full_dict["h"])
    g_sym = vertcat(*g_sym_list)
    # g_sym = ca.vertcat(nlp_standard_full_dict["g"], nlp_standard_full_dict["h"])

    nlp_standard_full = {"f":f_sym, "x":x_sym, "p":p_sym, "g":g_sym}
    
    return nlp_standard_full

if __name__ == '__main__':
    nlp, nlp_bounds, nlp_id = setup_NLP_example_1()
    nlp_dict = {"nlp": nlp, "nlp_bounds": nlp_bounds}

    nlp_diff = NLPDifferentiator(nlp_dict)

    

    # nlp_diff.transform_nlp(variant='full_standard')

    if True:
        # Test 1
        p_num = np.array((1,1))
        # S0 = nlpsol('S', 'ipopt', nlp)

        nlp_sol_opts = {}
        ipopt_options = {"fixed_variable_treatment": "make_constraint"} # FIXME: very important to get correct lagrange multipliers # TODO: set this as default in do-mpc
        nlp_sol_opts["expand"] = False

        ipopt_options["print_level"] = 4

        nlp_sol_opts["ipopt"] = ipopt_options
        # nlp_sol_opts["expand"] = False

        S0 = nlpsol('S', 'ipopt', nlp,nlp_sol_opts) #,nlpsol_options=ipopt_options

        r0 = S0(x0=0, p=p_num, **nlp_bounds)

        z_num, where_lam_not_zero = nlp_diff.extract_primal_dual_solution(r0, eps=1e-6)
        param_sens = nlp_diff.get_sensitivities(z_num, p_num, where_lam_not_zero, verbose=True, track_residues=False)
        dx_dp = nlp_diff.map_dxdp(param_sens)
        dlam_dp = nlp_diff.map_dlamdp(param_sens, where_lam_not_zero)

    if False:
        eval_dict = validate_fd(dx_dp, S0, nlp_bounds, p_num, x0=r0["x"], n_eval = 1, step_size = 1e-3)
        print(eval_dict)

    if True:
        # fill nlp_diff.alpha with zero if lam is zero and 1 if not
        alpha = np.zeros(nlp_diff.alpha.shape)
        alpha[where_lam_not_zero] = 1
        rho = 0.0
        # param_sens_ldl = nlp_diff.ls_sol_reg_ldl_func(z_num,p_num,alpha,rho)

        B_adapted_num, D_num, LT_num, LDL_permutation_num = nlp_diff.ldl_func(z_num,p_num,alpha,rho)
        param_sens_ldl = ldl_solve(B_adapted_num, D_num, LT_num, np.array(LDL_permutation_num,dtype=int).reshape(-1))

        
        
        
    # # Test get Lagrangian
    # nlp_diff.get_Lagrangian_sym()

    # nlp_diff.prepare_hessian()

    # H = nlp_diff.get_hessian(**r0)


    







    



  