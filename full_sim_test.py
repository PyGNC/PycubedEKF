#Full simulation test over 5 orbits

import autograd.numpy as np
import src
import unittest
from numpy.testing import assert_array_less

#used for converting ECEF to ECI
import brahe


#plotting
import matplotlib.pyplot as plt

# State is defined at
# [x, y,z, vx, vy, vz, ax, ay, az, beta_x, beta_y, beta_z]

def RMSE(residuals):
            
    rmse = np.sqrt(np.sum(residuals*residuals)/residuals.shape[0])
            
    return rmse



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

        #GPS measurements in ECEF frame. (5 orbits worth of data) #accuracy of 10 m and 1 mm/s
        GPS_measurements = np.loadtxt('data/gps_data_10m_1mm_s/gps_measurements_ecef_new.txt', delimiter='\t')

        #number of GPS measurements
        GPS_num = GPS_measurements.shape[1]

        N = 1000 #number of timesteps to run the EKF for

        #N = GPS_num

        #define all states and sqrt covariances
        all_states = np.zeros((12, N))
        all_sqrt_covariances = np.zeros((12,12, N))

        #save the initial state
        all_states[:,0] = EKF_test.x
        all_sqrt_covariances[:,:,0] = EKF_test.F

        #timestep
        dt = 1.0

        #this is the epoch of the first GPS measurement
        epc0 = brahe.epoch.Epoch(2012, 11, 8, 11, 59, 25, 0.0)

        epoch = epc0

        #loop through the GPS measurements and update the EKF (assuming fixed timestep of 1.0 s)
        #for i in range(GPS_num-1):
        for i in range(N-1):

            epoch += dt

            print("Iteration: ", i)
            #gps measurement of the next state
            gps_measurement = GPS_measurements[:,i+1]

            #update the prediction of the next state with the measurement
            EKF_test.update(gps_measurement, dt, epoch)
            
            #the ekf state and covariance at this time step
            all_states[:,i+1] = EKF_test.x

            all_sqrt_covariances[:,:,i+1] = EKF_test.F

            print("x: ", EKF_test.x)
            print("sqrt cov: ", np.diag(EKF_test.F))

        

        print("EKF Done")

        #get the final sqrt covariance and state
        F_final = EKF_test.F
        x_final = EKF_test.x

        print("Final state: ", x_final)
        print("Final covariance diagonal: ", np.diag(F_final*F_final.T))

        #save all the standard deviations in a matrix to test filter consistency
        all_x_std = np.zeros(N)
        all_y_std = np.zeros(N)
        all_z_std = np.zeros(N)
        all_vx_std = np.zeros(N)
        all_vy_std = np.zeros(N)
        all_vz_std = np.zeros(N)
        all_ax_std = np.zeros(N)
        all_ay_std = np.zeros(N)
        all_az_std = np.zeros(N)

        for i in range(N):

            all_x_std[i] = all_sqrt_covariances[0,0, i]
            all_y_std[i] = all_sqrt_covariances[1,1, i]
            all_z_std[i] = all_sqrt_covariances[2,2, i]
            all_vx_std[i] = all_sqrt_covariances[3,3, i]
            all_vy_std[i] = all_sqrt_covariances[4,4, i]
            all_vz_std[i] = all_sqrt_covariances[5,5, i]
            all_ax_std[i] = all_sqrt_covariances[6,6, i]
            all_ay_std[i] = all_sqrt_covariances[7,7, i]
            all_az_std[i] = all_sqrt_covariances[8,8, i]


        ground_truth = np.loadtxt('data/Ground_Truth/groundtruth_data.txt', delimiter='\t')


        res1 = ground_truth[0,0:N] - all_states[0,:]
        res2 = ground_truth[1,0:N] - all_states[1,:]
        res3 = ground_truth[2,0:N] - all_states[2,:]
        res4 = ground_truth[3,0:N] - all_states[3,:]
        res5 = ground_truth[4,0:N] - all_states[4,:]
        res6 = ground_truth[5,0:N] - all_states[5,:]
        res7 = ground_truth[6,0:N] - all_states[6,:]
        res8 = ground_truth[7,0:N]- all_states[7,:]
        res9 = ground_truth[8,0:N] - all_states[8,:]

        #create a subplot of all the residuals
        fig, axs = plt.subplots(3, 3)

        axs[0, 0].plot(res1)
        axs[0,0].plot(3*all_x_std)
        axs[0,0].plot(-3*all_x_std)

        axs[0, 0].set_title('X position Residuals')

        axs[0, 1].plot(res2)
        axs[0,1].plot(3*all_y_std)
        axs[0,1].plot(-3*all_y_std)

        axs[0, 1].set_title('Y position Residuals')

        axs[0, 2].plot(res3)
        axs[0,2].plot(3*all_z_std)
        axs[0,2].plot(-3*all_z_std)

        axs[0, 2].set_title('Z position Residuals')

        axs[1, 0].plot(res4)
        axs[1,0].plot(3*all_vx_std)
        axs[1,0].plot(-3*all_vx_std)

        axs[1, 0].set_title('X velocity Residuals')

        axs[1, 1].plot(res5)
        axs[1,1].plot(3*all_vy_std)
        axs[1,1].plot(-3*all_vy_std)

        axs[1, 1].set_title('Y velocity Residuals')

        axs[1, 2].plot(res6)
        axs[1,2].plot(3*all_vz_std)
        axs[1,2].plot(-3*all_vz_std)

        axs[1, 2].set_title('Z velocity Residuals')

        axs[2, 0].plot(res7)
        axs[2,0].plot(3*all_ax_std)
        axs[2,0].plot(-3*all_ax_std)

        axs[2, 0].set_title('X acceleration Residuals')

        axs[2, 1].plot(res8)
        axs[2,1].plot(3*all_ay_std)
        axs[2,1].plot(-3*all_ay_std)

        axs[2, 1].set_title('Y acceleration Residuals')

        axs[2, 2].plot(res9)
        axs[2,2].plot(3*all_az_std)
        axs[2,2].plot(-3*all_az_std)

        axs[2, 2].set_title('Z acceleration Residuals')

        for ax in axs.flat:
            ax.set(xlabel='Timestep (s)', ylabel='Difference (m - m/s - m/s^2)')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        plt.show()

        #Finding the RMSE
        rmse_x = RMSE(res1)
        rmse_y = RMSE(res2)
        rmse_z = RMSE(res3)
        rmse_vx = RMSE(res4)
        rmse_vy = RMSE(res5)
        rmse_vz = RMSE(res6)
        rmse_ax = RMSE(res7)
        rmse_ay = RMSE(res8)
        rmse_az = RMSE(res9)

        print("RMSE x: ")
        print(rmse_x)

        print("RMSE y: ")
        print(rmse_y)

        print("RMSE z: ")
        print(rmse_z)

        print("RMSE vx: ")
        print(rmse_vx)

        print("RMSE vy: ")
        print(rmse_vy)

        print("RMSE vz: ")
        print(rmse_vz)

        print("RMSE ax: ")
        print(rmse_ax)

        print("RMSE ay: ")
        print(rmse_ay)

        print("RMSE az: ")
        print(rmse_az)

        position_rmse = np.array([rmse_x, rmse_y, rmse_z])

        #check that the RMS error for positions is less than a meter
        assert_array_less(position_rmse,np.array([1.0, 1.0, 1.0]))
    
if __name__ == '__main__':
    unittest.main()