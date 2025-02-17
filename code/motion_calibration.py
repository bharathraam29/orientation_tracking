import load_data
import os
import transforms3d as t3d
import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

data_base_path= "/home/bradhakrishnan/ECE276A_PR1/data/"

################ STATIC UTILS ####################
def if_rotation_identity(rotation, threshold=0.001, debug=False):
    diagonal = jnp.diag(rotation)
    if debug:
        print(jnp.abs(diagonal - 1.0))
    is_diagonal_close_to_one = jnp.all(jnp.abs(diagonal - 1.0) < threshold)
    if debug:
        print(is_diagonal_close_to_one)
    return is_diagonal_close_to_one

def find_stationary_end(vicd_data, threshold=0.001, debug=False):
    idx=0
    while(if_rotation_identity(vicd_data[:,:, idx], threshold, debug=debug)):
        idx+=1
    if debug:
        print(idx)
    return idx


def get_static_data(vicd_data, vicd_ts, imu_data, indentity_check_threshold=0.001, debug=False):
    stationary_end_idx= find_stationary_end(vicd_data, threshold=indentity_check_threshold, debug=debug)
    print("Number of Static entries:: ", stationary_end_idx)
    stationary_start_time= vicd_ts[:,0][0]  
    stationary_end_time= vicd_ts[:,stationary_end_idx][0]
    static_imu_data=imu_data[:,0:stationary_end_idx]
    # if stationary_start_time== static_imu_data[0, 0] and stationary_end_time==static_imu_data[0,stationary_end_idx-1]:
    #     print(f"We might have to match the timestamps :: \nVidcon start:: {stationary_start_time} \nIMU start:: {static_imu_data[0, 0]}" )
    #     print(f"\n\nVidcon end:: {stationary_end_time} \nIMU end:: {static_imu_data[0, stationary_end_idx-1]}" )
    #     print(stationary_start_time-static_imu_data[0, 0])
    #     print(stationary_end_time-static_imu_data[0, stationary_end_idx-1])
    return static_imu_data

######################





###################### BIAS AND CALIBRATION UTILS ################



def find_bias():
    all_static_data= None
    for ii in range(1,10):
        print(f"Dataset {ii} :: \n")
        imud, vicd, camd=load_data.load_dataset(ii)
        vicd_data = vicd['rots']
        vicd_ts = vicd['ts']
        imu_data= imud
        static_imu_data= get_static_data(vicd_data, vicd_ts, imu_data, indentity_check_threshold=0.01, debug=False)
        if all_static_data is None:
            all_static_data=static_imu_data
        else:
            all_static_data= np.concatenate((all_static_data, static_imu_data), axis=1)
    
    bias= np.mean(all_static_data[1:, :], axis=1)
    # we need to manually compute the bias for 
    print(f"Bias ::",bias)
    return bias
    
def find_bias_in_dataset(dataset_idx):
    print(f"Dataset {dataset_idx} :: \n")
    imud, vicd, camd=load_data.load_dataset(dataset_idx)
    vicd_data = vicd['rots']
    vicd_ts = vicd['ts']
    imu_data= imud
    
    static_imu_data= get_static_data(vicd_data, vicd_ts, imu_data, indentity_check_threshold=0.01, debug=False)
    bias= np.mean(static_imu_data[1:, :], axis=1)

    print(f"Bias ::",bias)
    return bias
def find_bias_from_hard_threshold(dataset_idx, idx_thres=500):
    print(f"Dataset {dataset_idx} :: \n")
    imud, vicd, camd=load_data.load_dataset(dataset_idx)

    imu_data= imud
    bias= jnp.mean(imu_data[1:,:idx_thres+1],axis=1)
    print(bias)
    return bias
def calibrate_v2(dataset_idx,idx_thres):
    ## SENSITIVITY 
    accl_sensitivity_all_axis=300 #mV/g
    yaw_sensitivity = 3.33*(180/jnp.pi) #mV/ °/s
    roll_sensitivity = 3.33*(180/jnp.pi)
    pitch_sensitivity = 3.33*(180/jnp.pi)
    
    ## REFERENCE VOLTAGES
    accl_vref= 3.3*1000
    yaw_vref= 3.3*1000 #z axis
    roll_vref=3.3*1000 # x axis
    pitch_vref=3.3*1000 # y axis
    
    print(f"Dataset {dataset_idx} :: \n")
        
    imud, vicd, camd=load_data.load_dataset(dataset_idx)
    imu_data= imud
    bias= find_bias_from_hard_threshold(dataset_idx, idx_thres=idx_thres)
        
    # in the bias list the value we get z axis linear acceleration is basically the measurement for 9.8 m/s( i.e; g) acceleration
    # so we can mathematically compute the bias using the avg(IMU measurement for 1g)-(1/scale_factor)
    acc_z_bias= bias[2]- 1/((accl_vref/1023)/accl_sensitivity_all_axis)
    bias= bias.at[2].set(acc_z_bias)
    # bias[2]=acc_z_bias
    # Compute scaling factors
    accl_factor = (accl_vref / 1023) / accl_sensitivity_all_axis
    gyro_factors = np.array([
        (roll_vref / 1023) / roll_sensitivity,
        (pitch_vref / 1023) / pitch_sensitivity,
        (yaw_vref / 1023) / yaw_sensitivity
    ])
    
    # Calibrate acceleration and angular velocity using vectorized operations
    calibrated_imu = np.zeros_like(imu_data)
    calibrated_imu[0, :] = imu_data[0, :]  # Preserve the 0th row
    
    # Apply bias correction and scaling in one step
    calibrated_imu[1:4, :] = (imu_data[1:4, :] - bias[:3, np.newaxis]) * accl_factor
    calibrated_imu[4:7, :] = (imu_data[4:7, :] - bias[3:, np.newaxis]) * gyro_factors[:, np.newaxis]
    return calibrated_imu,vicd, camd

##############################################

############## Motion Model ##################
def motion_model(calibrated_imu,q0=[1.0,0.0,0.0,0.0]):
    prev_q=jnp.asarray(q0)
    pred_qts= [prev_q]
    for sample_idx in range(0,calibrated_imu.shape[1]):
        del_t= (calibrated_imu[0,sample_idx]- calibrated_imu[0,sample_idx-1])*0.5 if sample_idx else 0.0100049*0.5
        w_vec=jnp.array(calibrated_imu[4:,sample_idx])
        exp_term = jnp.concatenate((jnp.array([0]), w_vec*del_t), axis=0)

        curr_q= t3d.quaternions.qmult(pred_qts[sample_idx],t3d.quaternions.qexp(exp_term))
        pred_qts.append(curr_q/jnp.linalg.norm(curr_q))
            
    pred_qts= jnp.column_stack(pred_qts)
    return pred_qts


