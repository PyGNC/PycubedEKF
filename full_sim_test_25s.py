#Full simulation test over 5 orbits

import autograd.numpy as np
import src
import unittest
from numpy.testing import assert_array_equal, assert_array_almost_equal

#plotting
import matplotlib.pyplot as plt

# State is defined at
# [x, y,z, vx, vy, vz, ax, ay, az, beta_x, beta_y, beta_z]


class TestEKFSimConvergence(unittest.TestCase):


    def test_sim(self): 

        #intitial state
        x0 = np.array([-6.291282146358055e6,
            -1.6626488883220146e6,
            -2.2161533651363864e6,
        -2394.9723284619768,
        -540.1536627486031,
        7206.153483511754,
        ])

        #Initialize the EKF class
        EKF_test = src.EKF(x0)

        #5 orbits worth of simulated GPS data
        GPS_measurements = np.loadtxt('julia_implementation/gps_measurements.txt', delimiter='\t')

        #this is the true atmospheric density from julia implementation. used to check if atmospheric density model is doing good
        #atm_measurements = np.loadtxt('julia_implementation/atm_density.txt', delimiter='\t')

        #number of GPS measurements
        GPS_num = GPS_measurements.shape[1]

        start = 25   # Start value
        end = 25000    # End value
        spacing = 25 # Fixed spacing

        horizon = np.linspace(start, end, int((end - start) / spacing) + 1)



        print("this is horizon 1: ", horizon[0])

        print("this is horizon end: ", horizon[-1])

        print("this is gps num: ", GPS_num)

        print("size of horizon: ", horizon.shape)

        #define all states
        all_states = np.zeros((12, horizon.shape[0]+1))
        all_sqrt_covariances = np.zeros((12,12, horizon.shape[0]+1))

        dt = 25.0

        #loop through the GPS measurements and update the EKF (assuming fixed timestep of 1.0 s)
        #for i in range(GPS_num-1):

        count = 0


        for i in horizon:

            count += 1
            print("Iteration: ", i)
            #gps measurement of the next state
            gps_measurement = GPS_measurements[:,int(i)]

            #atm_density = atm_measurements[i]

            #update the prediction of the next state with the measurement
            EKF_test.update(gps_measurement, dt)
            
            #the ekf state and covariance at this time step
            all_states[:,count] = EKF_test.x

            all_sqrt_covariances[:,:,count] = EKF_test.F

            print("x: ", EKF_test.x)
            print("sqrt cov: ", np.diag(EKF_test.F))

        

        print("done")

        #get the final sqrt covariance and state
        F_final = EKF_test.F
        x_final = EKF_test.x

        print("Final state: ", x_final)
        print("Final covariance: ", F_final*F_final.T)

        plt.plot(all_states[6,:])

        plt.show()

        #plot the EKF state trajetory
        # plt.figure()
        # #3d plot
        # ax = plt.axes(projection='3d')
        # ax.plot3D(all_states[0,:], all_states[1,:], all_states[2,:], 'gray')
        # ax.scatter3D(all_states[0,:], all_states[1,:], all_states[2,:], c=all_states[2,:], cmap='Greens');
        # ax.set_xlabel('x [m]')
        # ax.set_ylabel('y [m]')
        # ax.set_zlabel('z [m]')
        # plt.title("Position")
        # plt.show()


        #assert_array_almost_equal(EKF_test.x, x_update, decimal=5)
    

if __name__ == '__main__':
    unittest.main()