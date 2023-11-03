from autograd import jacobian, numpy as np
from scipy.linalg import sqrtm
from scipy.linalg import qr
from scipy.linalg import solve
import brahe

#The state vector is defined as follows:
# x[0], x[1], x[2] -> x,y,z position
# x[3], x[4], x[5] -> x,y,z velocity
# x[6], x[7], x[8] -> x,y,z unmodeled accelerations (epsilons)
# x[9], x[10], x[11] -> time correlation coefficients (betas)

class EKFCore:
    # constructor
    def __init__(self, x0, F0, dynamics, measure, betas, epsilons, R) -> None:
        self.x = x0  # initial state
        self.F = F0  # initial square root covariance
        self.f = dynamics  # discrete dynamics function used
        self.g = measure  # measurement function used
        # tuning parameters for the first order gauss markov model
        self.betas = betas  # this is the tuning parameters q_beta
        self.epsilons = epsilons  # this is the tuning parameters q epsilon
        self.R = R  # measurement noise

    def predict(self, h):
        """
        EKF prediction step
        h is the timestep
        """

        # discrete dynamics function of state
        x_predicted = self.f(self.x, h)

        # find the jacobian using autograd.
        A = jacobian(lambda x: self.f(x, h))(self.x)

        # get F by taking the sqrt of P
        F = self.F

        # closed form process noise (from Myers DMC paper)
        V_ = np.identity(3)*np.vstack((self.epsilons[0]*(1-np.exp(-2*self.x[9]*h))/(2*self.x[9]), self.epsilons[1]*(
            1-np.exp(-2*self.x[10]*h))/(2*self.x[10]), self.epsilons[2]*(1-np.exp(-2*self.x[11]*h))/(2*self.x[11])))

        Y_ = h*np.identity(3) * \
            np.array([self.betas[0], self.betas[1], self.betas[2]])

        Qt_1 = np.hstack((0.25*(h**4)*V_, 0.5*(h**3)*V_,
                         0.5*(h**2)*V_, np.zeros((3, 3))))
        Qt_2 = np.hstack((0.5*(h**3)*V_, h**2*V_, h*V_, np.zeros((3, 3))))
        Qt_3 = np.hstack((0.5*(h**2)*V_, h*V_, V_, np.zeros((3, 3))))
        Qt_4 = np.hstack((np.zeros((3, 9)), Y_))
        Qt_stack = np.vstack((Qt_1, Qt_2, Qt_3, Qt_4))
        
        Qt = np.identity(12)*np.diag(Qt_stack)

        # propogate the sqrt covariance
        n = np.vstack((F@A.T, sqrtm(np.real(Qt))))

        _, F_predicted = qr(np.real(n), mode='economic')

        return x_predicted, F_predicted

    # mesurement y is from GPS. Ensure it is from timestep k+1
    # g is the measurement function
    def innovation(self, y, x_predicted, F_predicted, epoch):
        """
        EKF Innovation Step. 
        y is the true GPS measurement
        x_predicted is the predicted state
        F_predicted is the predicted square root covariance
        epoch is the time associated with the measurement
        """

        #predicted measurement in ECI
        y_predicted, C = self.g(x_predicted)

        # transform the true gps measurement from ECEF to ECI at 
        y_eci = brahe.frames.sECEFtoECI(epoch, y)

        # innovation
        Z = y_eci - y_predicted

        return Z,  C

    def kalman_gain(self, F_predicted, C):
        """
        EKF Kalman Gain
        """

        m = np.vstack((F_predicted@C.T, sqrtm(self.R)))

        _, G = qr(m, mode='economic')

        M = solve(G.T, C)@F_predicted.T@F_predicted

        L_inside = solve(G, M)

        L = L_inside.T

        return L

    def update(self, y, dt, epoch):
        """
        EKF update step
        """
        #Predict the next state and covariance
        x_predicted, F_predicted = self.predict(dt)

        #innovation step
        Z, C = self.innovation(y, x_predicted, F_predicted, epoch)

        #calculate kalman gain
        L = self.kalman_gain(F_predicted, C)

        # update the state
        self.x = x_predicted + L @ Z

        e = np.vstack((F_predicted@(np.identity(12)-L@C).T, sqrtm(self.R)@L.T))

        #update the square root covariance
        _, self.F = qr(e, mode='economic')