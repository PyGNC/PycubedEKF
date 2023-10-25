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

        for i in range(100):
            EKF.update(np.array([0.4+i*0.1, 0.4+i*0.3]))
        assert_array_almost_equal(EKF.x, np.array([10.3, 30.3, 0.1, 0.3]))


if __name__ == '__main__':
    unittest.main()
