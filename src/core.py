
from autograd import jacobian, numpy as np
from scipy.linalg import sqrtm

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

    # pass in just a timestep h
    def predict(self, h):

        # discrete dynamics function of state
        x_predicted = self.f(self.x, h)

        # find the jacobian using autograd
        A = jacobian(lambda x: self.f(x, h))(x_predicted)

        # get F by taking the sqrt of P
        F = self.F

        # closed form process noise (from Myers DMC paper)
        V_ = np.identity(3)*np.vstack((self.epsilons[0]*(1-np.exp(-2*self.x[9]*h))/(2*self.x[9]), self.epsilons[1]*(
            1-np.exp(-2*self.x[10]*h))/(2*self.x[10]), self.epsilons[2]*(1-np.exp(-2*self.x[11]*h))/(2*self.x[11])))

        Y_ = h*np.identity(3) * \
            np.array([self.betas[0], self.betas[1], self.betas[2]])

        # tested
        Qt_1 = np.hstack((0.25*(h**4)*V_, 0.5*(h**3)*V_,
                         0.5*(h**2)*V_, np.zeros((3, 3))))
        Qt_2 = np.hstack((0.5*(h**3)*V_, h**2*V_, h*V_, np.zeros((3, 3))))
        Qt_3 = np.hstack((0.5*(h**2)*V_, h*V_, V_, np.zeros((3, 3))))
        Qt_4 = np.hstack((np.zeros((3, 9)), Y_))
        Qt = np.vstack((Qt_1, Qt_2, Qt_3, Qt_4))

        # propogate the sqrt covariance
        n = np.vstack((F@A.T, sqrtm(Qt)))

        _, F_predicted = np.linalg.qr(np.real(n))

        return x_predicted, F_predicted

    # mesurement y is from GPS. Ensure it is from timestep k+1
    # g is the measurement function
    def innovation(self, y, x_predicted, F_predicted):

        y_predicted, C = self.g(x_predicted)

        # innovation
        Z = y - y_predicted

        return Z,  C

    def kalman_gain(self, F_predicted, C):

        m = np.vstack((F_predicted@C.T, sqrtm(self.R)))

        _, G = np.linalg.qr(m)

        M = np.linalg.solve(G.T, C)@F_predicted.T@F_predicted

        L_inside = np.linalg.solve(G, M)

        L = L_inside.T

        return L

    def update(self, y, dt):
        y = np.array(y)
        x_predicted, F_predicted = self.predict(dt)
        Z, C = self.innovation(y, x_predicted, F_predicted)
        L = self.kalman_gain(F_predicted, C)

        # update the class
        self.x = x_predicted + L @ Z

        # change to not hard code
        e = np.vstack((F_predicted@(np.identity(12)-L@C), sqrtm(self.R)@L.T))

        _, self.F = np.linalg.qr(e)

    # jupyter notebook is running python 3.8.8