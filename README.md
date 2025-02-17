# Orientation Tracking using IMU Data:

The repo implements projected gradient descent using IMU data to perform orientation tracking.
The general overview of the code is;
1. Calibrate sensor data
2. Using motion model get the inital motion data. 
3. Using this initial data rollout, perform Gradient Descent.
4. Build a Panorama using the rotation estimates from Grad Descent and the images provided after performing timestamp synchronization 

runner.py is the is "main" python file needed to run the projected gradient descent for orientation tracking

To perform projected gradient descent,

Start a jupyter notebook
set data_base_path with the absolute path of the data folder in runner.py
import the the python file runner.py
call the function runner.run(dataset_idx, epochs) => dataset idx signifies which dataset to use for gradient descent and epochs denotes how many epochs to run the algorithm for
Note: save the dataset such that the path looks like ../ECE276A_PR1/data/cam/cam<dataset_idx>.p ../ECE276A_PR1/data/imu/imuRaw<dataset_idx>.p ../ECE276A_PR1/data/vicon/viconRot<dataset_idx>.p

To generate panorama use panaroma_gen.py

Open the file panaroma_gen.py, set the dataset index corresponding to which dataset you would like to create the panorama for.
Also, set data_base_path with the absolute path of the data folder ie; something like /path/../ECE276A_PR1/data/
open a terminal and then run "python panaroma_gen.py"
The output of the script is stored in the path "../ECE276A_PR1/data/outputs/<dataset_idx>/pan.jpg"
PS: CHECK "../ECE276A_PR1/data/outputs/" to see the results if you do not want to run the code yourself.

quaternion_ops.py => Has JAX implementation of quaternion functions motion_calibration.py => Has functions realted to IMU data calibration runner.py=> has the functions related to motion model, observation model, cost function and gradient descent.

Please contact me over email: bradhakrishnan@ucsd.edu
