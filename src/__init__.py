import autograd.numpy as np
from scipy.linalg import sqrtm

from .core import EKFCore, BatchLSQCore

R_EARTH = 6.3781363e6 #Radius of the Earth in meters
OMEGA_EARTH = 7.292115146706979e-5 #angular velocity of the Earth in rad/s

def atm_density(alt):
    """
    Expoential fit model for atmospheric density from from SMAD data. 
    For orbits with an altitude of 350-650 km 
    Input: altitude in km
    Output: density in kg/m^3
    """

    atm_d = np.exp(-1.63481928e-2*alt) * np.exp(-2.00838711e1)

    return atm_d

def process_dynamics(x):
    """"
    Process model for the EKF
    Input: state vector (x)  
    Output: Time derivative of the state vector (x_dot)
    """
    # position
    q = np.array(x[0:3])

    #velocity
    v = np.array(x[3:6])

    #unmodeled accelerations
    # a_d = np.array(x[6:9])

    #time correlation coefficients
    # beta = np.array(x[9:12])

    # drag coefficient
    cd = 2.0

    # cross sectional area (m^2)
    A = 0.1

    # angular velocity of the Earth
    omega_earth = np.array([0, 0, OMEGA_EARTH])

    # relative velocity
    v_rel = v - np.cross(omega_earth, q)

    # get the altititude in km
    alt = (np.linalg.norm(q) - R_EARTH)*1e-3

    # estimated rho from model
    rho_est = atm_density(alt)

    # drag force
    f_drag = -0.5*cd*(A)*rho_est*np.linalg.norm(v_rel)*v_rel

    # gravitational parameter of the Earth
    μ = 3.986004418e14  # m3/s2

    # J2 perturbation constant
    J2 = 1.08264e-3

    #Two body acceleration
    a_2bp = (-μ*q)/(np.linalg.norm(q))**3

    #z unit vector
    Iz = np.array([0, 0, 1])

    #accleration due to J2
    a_J2 = ((3*μ*J2*R_EARTH**2)/(2*np.linalg.norm(q)**5)) * \
        ((((5*np.dot(q, Iz)**2)/np.linalg.norm(q)**2)-1)*q - 2*np.dot(q, Iz)*Iz)

    #total acceleration (two body + J2 + drag + unmodeled accelerations)
    a = a_2bp + a_J2 + f_drag #+ a_d

    #unmodeled accelerations modeled as a first order gaussian process
    #time corellation coefficients modeled as a random walk (time derivative = 0)
    # a_d_dot = -np.diag(beta)@a_d

    #state derivative
    x_dot = np.concatenate([v, a])#, a_d_dot, np.zeros(3)])

    return x_dot

def rk4(x, h, f):
    """
    Runge-Kutta 4th order integrator
    """
    
    k1 = f(x)
    k2 = f(x + h/2 * k1)
    k3 = f(x + h/2 * k2)
    k4 = f(x + h * k3)
    return x + h/6 * (k1 + 2*k2 + 2*k3 + k4)

class EKF(EKFCore):
    """
    Defines the EKF for the orbit determination problem
    """

    def __init__(self, x0):

        def time_dynamics(x, dt):
            """
            Discrete-time dynamics function
            """
            return rk4(x, dt, process_dynamics)
        
        def measurement_function(x):
            """
            Measurement function for GPS measurements providing position and velocity
            """
            C = np.hstack((np.eye(6), np.zeros((6, 6))))
            measurement = C@x
            return measurement, C

        #standard deviation of the GPS position measurement in meters
        std_gps_measurement = 10

        #standard devation of the GPS velocity measurement in m/s
        std_velocity = 0.001
        #switch depending if we are using an accuracy of mm/s or cm/s on the velocity
        #std_velocity = 0.01

        #tuning parameters for the first order gauss markov model
        q_betas = 2e-9 * np.ones(3)
        q_eps = 5.5e-11 * np.ones(3)

        #measurement noise matrix
        R_measure = np.identity(6) * np.hstack([(((std_gps_measurement)**2)/3)*np.ones(3), ((std_velocity)**2)/3*np.ones(3)])

        #initial covariance matrix
        P0 = np.identity(12) * np.hstack((np.ones(3) * ((std_gps_measurement)**2)/3,
                                          np.ones(3) * ((std_velocity)**2)/3, np.ones(3) * 5e-4**2, np.ones(3) * 4e-4**2))
        #initial square root covariance matrix
        F0 = sqrtm(P0)

        #initial state
        x0 = np.concatenate([x0, np.zeros(3), 1e-3 * np.ones(3)])

        super().__init__(x0, F0, time_dynamics,
                         measurement_function, q_betas, q_eps, R_measure)
        
class BA(BatchLSQCore):
    """
    Defines the Batch Least-Squares solver for the orbit determination problem
    """

    def __init__(self, x0,y):

        def time_dynamics_single(x, dt):
            """
            Discrete-time dynamics function
            """
            x_cat = np.zeros(24)
            for i in range(3):
                x_i = x[i*6:(i+1)*6]
                xi_1 = rk4(x_i, dt, process_dynamics)
                x_cat[i*6:(i+1)*6] = xi_1
            return x_cat
        
        def time_dynamics(x, dt):
            """"
            Batch dynamics function
            """
            x_dyn = np.zeros_like(x)
            for i in range(x.shape[1]):
                x_dyn[:,i] = time_dynamics_single(x[:,i], dt)
            return x_dyn
        
        def measurement_function_single(x):
            """
            Measurement function for GPS measurements providing position and velocity, provides ranges between chief and deputy
            """
            measurement_gps = x[0:6]
            measurement_range = np.array([np.linalg.norm(x[0:3] - x[6:9]),np.linalg.norm(x[0:3] - x[12:15]),np.linalg.norm(x[0:3] - x[18:21])])
            #print(measurement_range.shape, measurement_gps.shape)
            measurement = np.concatenate((measurement_gps, measurement_range))
            return measurement
        
        def measurement_function(x):
            """
            batch measurement function
            """
            #print(x.shape)
            x_meas = np.zeros((9, x.shape[1]))
            for i in range(x.shape[1]):
                x_meas[:,i] = measurement_function_single(x[:,i])
            #print(x_meas.shape)
            return x_meas

        #standard deviation of the GPS position measurement in meters
        std_gps_measurement = 10

        #standard devation of the GPS velocity measurement in m/s
        std_velocity = 0.001
        #switch depending if we are using an accuracy of mm/s or cm/s on the velocity
        #std_velocity = 0.01

        # standard deviation of the range measurement in meters
        std_range = 1

        #tuning parameters for the first order gauss markov model
        # q_betas = 2e-9 * np.ones(3)
        # q_eps = 5.5e-11 * np.ones(3)

        # Process noise covariances
        pose_std_dynamics = 4e-6#*1e-3 #get to km
        velocity_std_dynamics = 8e-6 #*1e-3 #get to km/s

        #measurement noise matrix
        R_measure = np.identity(9) * np.hstack([(((std_gps_measurement)**2)/3)*np.ones(3), ((std_velocity)**2)/3*np.ones(3), ((std_range)**2)/3*np.ones(3)])

        #Process ovariance matrix for one satellite
        Q_ind = np.hstack((np.ones(3) * ((pose_std_dynamics)**2)/3, np.ones(3) * ((velocity_std_dynamics)**2)/3))
        #Repeat Q_ind for all satellites
        Q_proc = np.diag(np.hstack((Q_ind, Q_ind, Q_ind, Q_ind)))
        

        #initial state
        x0 = x0

        dt = 1
        #initial measurement
        # y = measurement_function(x0)

        super().__init__(x0, y, time_dynamics,
                         measurement_function, Q_proc, R_measure,dt)
        # self, x0, y, dynamics, measure, Q, R
