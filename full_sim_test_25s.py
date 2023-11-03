#Full simulation test over 5 orbits

import autograd.numpy as np
import src
import unittest
from numpy.testing import assert_array_less
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

        #GPS measurements in the ECEF frame. (5 orbits worith of data)
        GPS_measurements = np.loadtxt('data/gps_data_10m_1mm_s/gps_measurements_ecef_new.txt', delimiter='\t')

        #number of GPS measurements
        GPS_num = GPS_measurements.shape[1]

        #this is to sample the GPS measurements to only read every 25 seconds
        start = 25   # Start value
        end = 5000    # End value for about 1 orbit worth of data
        #for about 5 orbits worth of data
        #end = 25000    # End value
        spacing = 25 # Fixed spacing

        horizon = np.linspace(start, end, int((end - start) / spacing) + 1)

        #define all states
        all_states = np.zeros((12, horizon.shape[0]+1))
        all_sqrt_covariances = np.zeros((12,12, horizon.shape[0]+1))

        #save the initial state
        all_states[:,0] = EKF_test.x
        all_sqrt_covariances[:,:,0] = EKF_test.F

        #timestep
        dt = 25.0

        #define a counter variable
        count = 0

        #define the initial epoch
        #year, month, day, hour, minute, second, microsecond
        epc0 = brahe.epoch.Epoch(2012, 11, 8, 11, 59, 25, 0.0)

        epoch = epc0

        for i in horizon:

            count += 1
            epoch += dt
            print("Iteration: ", i)
            #gps measurement of the next state
            gps_measurement = GPS_measurements[:,int(i)]

            #update the prediction of the next state with the measurement
            #along with the time associated with the measurement
            EKF_test.update(gps_measurement, dt, epoch)
            
            #the ekf state and covariance at this time step
            all_states[:,count] = EKF_test.x

            all_sqrt_covariances[:,:,count] = EKF_test.F

            print("x: ", EKF_test.x)
            print("sqrt cov: ", np.diag(EKF_test.F))

        

        print("EKF done")

        #get the final sqrt covariance and state
        F_final = EKF_test.F
        x_final = EKF_test.x

        print("Final state: ", x_final)
        print("Final covariance: ", F_final*F_final.T)

        N = horizon.shape[0]+1

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

        #get all the standard deviations
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
            
        end

        #find the residuals between the true state and the estimated state
        ground_truth = np.loadtxt('data/Ground_Truth/groundtruth_data.txt', delimiter='\t')

        #sample the ground truth data to only read every 25 seconds
        ground_truth_25 = np.zeros((9, all_states.shape[1]))
        ground_truth_25[:,0] = ground_truth[:,0]

        count2 = 0

        for i in horizon:

            count2 += 1 
            ground_truth_25[:,count2] = ground_truth[:,int(i)]


        #find the residuals between ground truth and estimated state
        res1 = ground_truth_25[0,:] - all_states[0,:]
        res2 = ground_truth_25[1,:] - all_states[1,:]
        res3 = ground_truth_25[2,:] - all_states[2,:]
        res4 = ground_truth_25[3,:] - all_states[3,:]
        res5 = ground_truth_25[4,:] - all_states[4,:]
        res6 = ground_truth_25[5,:] - all_states[5,:]
        res7 = ground_truth_25[6,:] - all_states[6,:]
        res8 = ground_truth_25[7,:]- all_states[7,:]
        res9 = ground_truth_25[8,:] - all_states[8,:]


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
            ax.set(xlabel='Time (s)', ylabel='Difference (m - m/s - m/s^2)')

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