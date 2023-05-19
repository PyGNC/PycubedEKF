import numpy as np
from autograd import jacobian


class EKFCore:
    def __init__(self, x0, P0, dynamics, measure, W, V) -> None:
        self.x = x0
        self.P = P0
        self.f = dynamics
        self.g = measure
        self.W = W
        self.V = V

    def predict(self, dt):
        x_predicted = self.f(self.x)
        A = jacobian(lambda x: self.f(x, dt))(x_predicted)
        P_predicted = A @ self.P @ A.T + self.W
        return x_predicted, P_predicted

    def innovation(self, y, x_predicted, P_predicted):
        y_predicted = self.g(x_predicted)
        C = jacobian(self.g)(x_predicted)

        Z = y - y_predicted
        S = C @ P_predicted @ C.T + self.V
        return Z, S, C

    def kalman_gain(self, P_predicted, C, S):
        return P_predicted @ C.T @ np.linalg.inv(S)

    def update(self, y, dt):
        (x_predicted, P_predicted) = self.predict(dt)
        (Z, S, C) = self.innovation(y, x_predicted, P_predicted)
        L = self.kalman_gain(P_predicted, C, S)
        self.x = x_predicted + L @ Z
        I = np.eye(self.P.shape[0])
        self.P = (I - L @ C) @ P_predicted @ (I - L @ C).T + L @ self.V @ L.T
