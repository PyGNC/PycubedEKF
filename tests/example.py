from src import EKFCore
import autograd.numpy as np
import unittest
from numpy.testing import assert_array_equal, assert_array_almost_equal

# State is defined at
# [x, y, vx, vy]


class TestBasicEKFConvergence(unittest.TestCase):

    def test_2d_point(self):

        def dynamics(x, dt):
            return np.array([x[0]+x[2]*dt, x[1]+x[3]*dt, x[2], x[3]])

        def measure(x):
            return np.array([x[0], x[1]])

        x0 = np.array([0.0, 0.0, 0.0, 0.0])
        P0 = np.eye(4)
        EKF = EKFCore(x0, P0, dynamics, measure, np.eye(4), np.eye(2))

        #from julia
        gps_measurement = np.array([-6.29367e6, -1.66319e6, -2.20896])

        next_state = np.array([
            -6.293670722492396e6,
            -1.6631898292000028e6,
            -2.208954003269442e6,
            -2387.246108140763,
            -538.1118478822381,
            7208.877748508874,
            9.489428442008259e-9,
            -6.772705033133127e-9,
            -3.054530330523015e-8,
            0.001,
            0.001,
            0.001
        ])

        for i in range(100):
            EKF.update(np.array([0.4+i*0.1, 0.4+i*0.3]))
        assert_array_almost_equal(EKF.x, next_state)


if __name__ == '__main__':
    unittest.main()
