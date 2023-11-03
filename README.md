# PycubedEKF
This is a core python implementation of an Extended Kalman Filter (EKF) for orbit determination of a cube satellite. 

## src
Contains the EKF class and all EKF related equations

## data
Contains GPS measurements in the ECEF frame as well as the ECI frame. There are two sets of measurements, one with a velocity accuracy of 1 mm/s and another with an accuracy of 1 cm/s. 

## full simulation tests
full sim test 25s tests the filter with a timestep of 25 seconds. 
full sim test tests the filter witha timestep of 1 second. 

Both check that the RMS error for the positions are less than 1 meter at the end of the simulation. 

## Results
Shows the consistency plots for the filter

## Generating GPS data
Inside the julia implementation folder, create_gps_data.ipynb generates GPS data for a specific orbit and GPS measurment errors. When using new measurement errors, make sure to update the filter intialization in the init.py file. 

## Running 
python full_sim_test_25s.py
python full_sim_test.py


