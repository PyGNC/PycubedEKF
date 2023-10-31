import autograd.numpy as np
from scipy.linalg import sqrtm

from .core import EKFCore

R_EARTH = 6.3781363e6
OMEGA_EARTH = 7.292115146706979e-5


#altitude in km
def atm_density(alt):
    """
    Expoential fit model from SMAD data 
    """

    atm_d = np.exp(-1.63481928e-2*alt) * np.exp(-2.00838711e1)

    return atm_d


def process_dynamics(x):
    # position
    q = np.array(x[0:3])
    v = np.array(x[3:6])
    a_d = np.array(x[6:9])
    beta = np.array(x[9:12])

    # drag coefficient
    cd = 2.0

    # cross sectional area
    A = 0.1

    omega_earth = np.array([0, 0, OMEGA_EARTH])

    v_rel = v - np.cross(omega_earth, q)

    # get the altititude in km
    alt = (np.linalg.norm(q) - R_EARTH)*1e-3

    #print("this is altitude")
    #print(alt)

    # estimated rho
    rho_est = atm_density(alt)

    #for testing
    #rho_est = 1e-13

    # drag force
    f_drag = -0.5*cd*(A)*rho_est*np.linalg.norm(v_rel)*v_rel

    μ = 3.986004418e14  # m3/s2
    J2 = 1.08264e-3

    a_2bp = (-μ*q)/(np.linalg.norm(q))**3

    Iz = np.array([0, 0, 1])

    a_J2 = ((3*μ*J2*R_EARTH**2)/(2*np.linalg.norm(q)**5)) * \
        ((((5*np.dot(q, Iz)**2)/np.linalg.norm(q)**2)-1)*q - 2*np.dot(q, Iz)*Iz)

    # print(f"a_2bp: {a_2bp} a_J2: {a_J2} f_drag: {f_drag} a_d: {a_d}")
    a = a_2bp + a_J2 + f_drag + a_d

    a_d_dot = -np.diag(beta)@a_d

    x_dot = np.concatenate([v, a, a_d_dot, np.zeros(3)])

    return x_dot


def rk4(x, h, f):
    
    k1 = f(x)
    k2 = f(x + h/2 * k1)
    k3 = f(x + h/2 * k2)
    k4 = f(x + h * k3)
    return x + h/6 * (k1 + 2*k2 + 2*k3 + k4)


class EKF(EKFCore):

    def __init__(self, x0):

        def time_dynamics(x, dt):
            return rk4(x, dt, process_dynamics)

        def measurement_function(x):
            C = np.hstack((np.eye(3), np.zeros((3, 9))))
            # only return the position
            measurement = C@x
            return measurement, C

        std_gps_measurement = 10
        std_velocity = 0.01


        #original values that worked in julia
        #work well for dt=25
        q_betas = 8e-8 * np.ones(3)
        q_eps = 1.5e-10 * np.ones(3)

        #work well for dt=1
        #q_betas = 2e-9 * np.ones(3)
        #q_eps = 5.5e-11 * np.ones(3)

        #tuning here
        #q_betas = 8e-8 * np.ones(3)
        #q_eps = 5.5e-11 * np.ones(3)

        R_measure = np.identity(3) * ((std_gps_measurement)**2)/3

        P0 = np.identity(12) * np.hstack((np.ones(3) * ((std_gps_measurement)**2)/3,
                                          np.ones(3) * ((std_velocity)**2)/3, np.ones(3) * 5e-4**2, np.ones(3) * 4e-4**2))
        F0 = sqrtm(P0)

        x0 = np.concatenate([x0, np.zeros(3), 1e-3 * np.ones(3)])

        super().__init__(x0, F0, time_dynamics,
                         measurement_function, q_betas, q_eps, R_measure)
