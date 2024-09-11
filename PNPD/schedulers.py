import numpy as np
from .math_extras import multiply_P_inverse, prox_h_star_TV
from typing import Callable

def nu_scheduler_classic(nu: float):
    return lambda i: .5*.85**i + nu
    
def nu_scheduler_increasing(nu: float):
    return lambda i: (1-i**(-.5))*(1-nu) + nu

def nu_scheduler_bootstrap(nu_0: float, bootstrap_iterations):
    c = nu_0**-(1/bootstrap_iterations)
    def scheduler(i):
        if i > bootstrap_iterations:
            return 1
        else:
            return c**(i-1-bootstrap_iterations)
    return scheduler

def mul_P_inv_scheduler(
        nu_scheduler: Callable[[int], float],
        psfAbsSq: np.ndarray
        ):
    def scheduler(i: int):
        nu = nu_scheduler(i)
        preconditioner_polynomial = np.polynomial.Polynomial([nu, 1-nu])
        return lambda x: multiply_P_inverse(
            p=preconditioner_polynomial,
            x=x,
            psfAbsSq=psfAbsSq
            )
    return scheduler

def mul_P_inv_scheduler_smart(
        nu_scheduler: Callable[[int], float],
        psfAbsSq: np.ndarray
        ):
    def scheduler(i: int):
        nu = nu_scheduler(i)
        if nu > 1 - 1e-13:
            return lambda x: x
        else:
            preconditioner_polynomial = np.polynomial.Polynomial([nu, 1-nu])
            return lambda x: multiply_P_inverse(p=preconditioner_polynomial, x=x, psfAbsSq=psfAbsSq)
    return scheduler

def mul_P_inv_scheduler_bootstrap(
        nu_scheduler: Callable[[int], float],
        psfAbsSq: np.ndarray,
        bootstrap_iterations: int
        ):
    def scheduler(i: int):
        if i > bootstrap_iterations:
            return lambda x: x
        else:
            nu = nu_scheduler(i)
            preconditioner_polynomial = np.polynomial.Polynomial([nu, 1-nu])
            return lambda x: multiply_P_inverse(p=preconditioner_polynomial, x=x, psfAbsSq=psfAbsSq)                
    return scheduler

def lam_scheduler_norm_precond(nu_scheduler, lam_bar):
    return lambda i: lam_bar / nu_scheduler(i)

def prox_h_star_scheduler(lam_scheduler: Callable[[int], float]):
    def scheduler(i: int):
        lam = lam_scheduler(i)
        return lambda alpha, x: prox_h_star_TV(lam, x)
    return scheduler

def constant_scheduler(c):
    return lambda i: c