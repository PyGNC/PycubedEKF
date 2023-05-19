from src import EKFCore
import autograd.numpy as np

# State is defined at
# [x, y, vx, vy]


def dynamics(x, dt):
    return np.array([x[0]+x[2]*dt, x[1]+x[3]*dt, x[2], x[3]])


def measure(x):
    return np.array([x[0], x[1]])


x0 = np.array([0.0, 0.0, 0.0, 0.0])
P0 = np.eye(4)
EKF = EKFCore(x0, P0, dynamics, measure, np.eye(4), np.eye(2))

for i in range(100):
    EKF.update(np.array([0.4+i*0.1, 0.4+i*0.3]), 1.0)
    print(EKF.x)
