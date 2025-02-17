import sys
sys.path.insert(0,'/home/bradhakrishnan/ECE276A_PR1/code/')

import motion_calibration
import load_data
import jax
import jax.numpy as jnp
import quaternion_ops as qops 
import numpy as np

import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
data_base_path= "/home/bradhakrishnan/ECE276A_PR1/data/"


import transforms3d as t3d


def motion_model_rollover(calibrated_imu):
    pred_qts= [jnp.array([1,0,0,0])]
    exp_vals=[]
    for idx in range(0,calibrated_imu.shape[1]-1):
        exp_term= calibrated_imu[4:,idx]* (calibrated_imu[0,idx+1]- calibrated_imu[0,idx])/2
        exp_val= t3d.quaternions.qexp(jnp.concatenate((jnp.array([0]),exp_term)))
        exp_vals.append(exp_val)
        pred_qts.append(t3d.quaternions.qmult(pred_qts[idx],exp_val))
    pred_qts= jnp.column_stack(pred_qts)
    exp_vals=jnp.column_stack(exp_vals)
    return pred_qts, exp_vals

def obs_model(q):
    return qops.qmult(qops.qinverse(q),qops.qmult(jnp.array([0., 0., 0., 1]),q))

def motion_model(q,exp):
    return qops.qmult(q, exp)

def rotation_err(qt_1, ft):
    return jnp.linalg.norm(2*qops.qlog(qops.qmult(qops.qinverse(qt_1), ft))) **2

def position_err(at, ht):
    return jnp.linalg.norm(at-ht) **2

hts= jax.vmap(obs_model, in_axes=(1), out_axes=1)
fts= jax.vmap(motion_model, in_axes=(1,1), out_axes=1)

rot_error_vec=jax.vmap(rotation_err, in_axes=(1,1))
pos_error_vec= jax.vmap(position_err, in_axes=(1,1))

def vector_normalize(q):
    return q/ jnp.linalg.norm(q)
    
def cost_fn(QTS, linear_acc, exp_vals):
    return (pos_error_vec(linear_acc[:,1:], hts(QTS[:,1:])).sum()+rot_error_vec(QTS[:,1:],fts(QTS[:,:-1], exp_vals)).sum())/2


import os

def plot_and_save_euler_angles(pred_qts, imu_ts, vicd_data, save_folder):
    """
    Computes and plots Euler angles (roll, pitch, yaw) from predicted quaternions and ground truth rotation matrices.
    Saves the plots to the specified folder.
    
    Parameters:
    pred_qts (jnp.ndarray): Predicted quaternions of shape (4, N)
    vicd_data (dict): Dictionary containing ground truth rotation matrices under key 'rots' of shape (3,3,N)
    save_folder (str): Path to the folder where plots will be saved
    """
    os.makedirs(save_folder, exist_ok=True)
    
    N_pred = pred_qts.shape[-1]
    N_gt = vicd_data['rots'].shape[-1]
    
    pred_roll, pred_pitch, pred_yaw = np.zeros(N_pred), np.zeros(N_pred), np.zeros(N_pred)
    gt_roll, gt_pitch, gt_yaw = np.zeros(N_gt), np.zeros(N_gt), np.zeros(N_gt)
    
    for idx in range(N_pred):
        pred_roll[idx], pred_pitch[idx], pred_yaw[idx] = jnp.array(
            t3d.euler.quat2euler(pred_qts[:, idx], 'sxyz')
        ) * (180 / jnp.pi)
    
    for idx in range(N_gt):
        gt_roll[idx], gt_pitch[idx], gt_yaw[idx] = jnp.array(
            t3d.euler.mat2euler(vicd_data['rots'][:, :, idx], 'sxyz')
        ) * (180 / jnp.pi)
    
    angles = [('Roll', pred_roll, gt_roll), ('Pitch', pred_pitch, gt_pitch), ('Yaw', pred_yaw, gt_yaw)]
    
    for angle_name, pred_values, gt_values in angles:
        plt.figure()
        import ipdb; ipdb.set_trace()
        plt.plot(imu_ts.flatten(), pred_values, label=f'Predicted {angle_name}')
        plt.plot(vicd_data['ts'].flatten(), gt_values, color='red', label=f'Ground Truth {angle_name}')
        plt.xlabel('Time Step')
        plt.ylabel(f'{angle_name} (Degrees)')
        plt.legend()
        plt.title(f'{angle_name} Comparison')
        plt.savefig(os.path.join(save_folder, f'{angle_name.lower()}_comparison.png'))
        plt.close()

    print(f'Plots saved to {save_folder}')

def plot_and_save_euler_angles_V2(pred_qts, imu_ts, vicd_data, save_folder):
    """
    Computes and plots Euler angles (roll, pitch, yaw) from predicted quaternions and ground truth rotation matrices.
    Saves the plot to the specified folder.
    
    Parameters:
    pred_qts (jnp.ndarray): Predicted quaternions of shape (4, N)
    imu_ts (array-like): Timestamps for predicted quaternions
    vicd_data (dict): Dictionary containing ground truth rotation matrices under key 'rots' of shape (3,3,N)
    save_folder (str): Path to the folder where the plot will be saved
    """
    os.makedirs(save_folder, exist_ok=True)
    
    N_pred = pred_qts.shape[-1]
    N_gt = vicd_data['rots'].shape[-1]
    
    pred_roll, pred_pitch, pred_yaw = np.zeros(N_pred), np.zeros(N_pred), np.zeros(N_pred)
    gt_roll, gt_pitch, gt_yaw = np.zeros(N_gt), np.zeros(N_gt), np.zeros(N_gt)
    
    for idx in range(N_pred):
        pred_roll[idx], pred_pitch[idx], pred_yaw[idx] = jnp.array(
            t3d.euler.quat2euler(pred_qts[:, idx], 'sxyz')
        ) * (180 / jnp.pi)
    
    for idx in range(N_gt):
        gt_roll[idx], gt_pitch[idx], gt_yaw[idx] = jnp.array(
            t3d.euler.mat2euler(vicd_data['rots'][:, :, idx], 'sxyz')
        ) * (180 / jnp.pi)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)  # 3 subplots in a column
    
    angles = [('Roll', pred_roll, gt_roll), ('Pitch', pred_pitch, gt_pitch), ('Yaw', pred_yaw, gt_yaw)]
    gt_colors = ['red', 'green', 'blue']  # Colors for ground truth
    
    for ax, (angle_name, pred_values, gt_values), gt_color in zip(axes, angles, gt_colors):
        ax.plot(imu_ts.flatten(), pred_values, linestyle='dashed', color='orange', label=f'Predicted {angle_name}')
        ax.plot(vicd_data['ts'].flatten(), gt_values, color=gt_color, label=f'Ground Truth {angle_name}')
        ax.set_ylabel(f'{angle_name} (Degrees)')
        ax.legend()
        ax.grid(True)
    
    axes[-1].set_xlabel('Time Step')  # Only set xlabel for the last subplot
    fig.suptitle('Comparison of Predicted and Ground Truth Euler Angles', fontsize=14)
    
    save_path = os.path.join(save_folder, 'euler_angles_subplots.png')
    plt.savefig(save_path)
    plt.close()
    
    vicd_qts= []
    for idx in range(N_gt):
        vicd_qts.append( jnp.array(
            t3d.quaternions.mat2quat(vicd_data['rots'][:, :, idx])))
    vicd_qts= jnp.column_stack(vicd_qts)
    acc_vicd= hts(vicd_qts)
    acc_pred= hts(pred_qts)
    
    # Create 3 separate subplots (excluding the first row)
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    subplot_labels = ['Ax', 'Ay', 'Az']
    for i in range(1, 4):  # Start from row index 1
        axes[i-1].plot(acc_vicd[i], label='Ground Truth', color='blue')   # Ground Truth (data)
        axes[i-1].plot(acc_pred[i], label='Predicted', color='orange', linestyle='dashed')  # Predicted (data2)
        axes[i-1].set_ylabel(subplot_labels[i-1])
        axes[i-1].grid(True)
        axes[i-1].legend()  # Add legend to each subplot
    
    axes[-1].set_xlabel('Time')
    plt.suptitle('Comparison of Predicted and Ground Truth Accelerations')
    plt.tight_layout()
    
    # Save the image
    save_path = os.path.join(save_folder, "acc_subplots.png")
    plt.savefig(save_path)
    plt.close()  # Close the plot to free memory
  
    print(f'Plot saved to {save_path}')


def plot_and_save_euler_angles_V3(pred_qts, imu_ts, vicd_data, save_folder, dataset_idx):
    """
    Computes and plots Euler angles (roll, pitch, yaw) from predicted quaternions and ground truth rotation matrices
    (if provided). Saves the plot to the specified folder.
    
    Parameters:
    pred_qts (jnp.ndarray): Predicted quaternions of shape (4, N)
    imu_ts (array-like): Timestamps for predicted quaternions
    vicd_data (dict or None): Dictionary containing ground truth rotation matrices under key 'rots' of shape (3,3,N).
                              If None, ground truth data is ignored.
    save_folder (str): Path to the folder where the plot will be saved
    """
    os.makedirs(save_folder, exist_ok=True)
    
    N_pred = pred_qts.shape[-1]
    pred_roll, pred_pitch, pred_yaw = np.zeros(N_pred), np.zeros(N_pred), np.zeros(N_pred)

    for idx in range(N_pred):
        pred_roll[idx], pred_pitch[idx], pred_yaw[idx] = jnp.array(
            t3d.euler.quat2euler(pred_qts[:, idx], 'sxyz')
        ) * (180 / jnp.pi)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)  # 3 subplots in a column
    angles = [('Roll', pred_roll), ('Pitch', pred_pitch), ('Yaw', pred_yaw)]
    
    for ax, (angle_name, pred_values) in zip(axes, angles):
        pred_linestyle = 'dashed' if vicd_data is not None else 'solid'
        ax.plot(imu_ts.flatten(), pred_values, linestyle=pred_linestyle, color='orange', label=f'Predicted {angle_name}')
        ax.set_ylabel(f'{angle_name} (Degrees)')
        ax.legend()
        ax.grid(True)

    if vicd_data is not None:
        N_gt = vicd_data['rots'].shape[-1]
        gt_roll, gt_pitch, gt_yaw = np.zeros(N_gt), np.zeros(N_gt), np.zeros(N_gt)
        
        for idx in range(N_gt):
            gt_roll[idx], gt_pitch[idx], gt_yaw[idx] = jnp.array(
                t3d.euler.mat2euler(vicd_data['rots'][:, :, idx], 'sxyz')
            ) * (180 / jnp.pi)
        
        gt_colors = ['red', 'green', 'blue']
        for ax, gt_values, gt_color in zip(axes, [gt_roll, gt_pitch, gt_yaw], gt_colors):
            ax.plot(vicd_data['ts'].flatten(), gt_values, color=gt_color, label='Ground Truth')
            ax.legend()

    axes[-1].set_xlabel('Time Step')
    fig.suptitle(f'Comparison of Predicted and Ground Truth Euler Angles - Dataset {dataset_idx}', fontsize=14)

    save_path = os.path.join(save_folder, 'euler_angles_subplots.png')
    plt.savefig(save_path)
    plt.close()
    print(f'Plot saved to {save_path}')

    # Suppress VICD computations if vicd_data is None
    if vicd_data is not None:
        # vicd_qts = [jnp.array(t3d.quaternions.mat2quat(vicd_data['rots'][:, :, idx])) for idx in range(N_gt)]
        vicd_qts = [jnp.array(t3d.quaternions.mat2quat(vicd_data['rots'][:, :, idx])) 
    for idx in range(N_gt) 
    if not jnp.isnan(vicd_data['rots'][:, :, idx]).any()]
        vicd_qts = jnp.column_stack(vicd_qts)
        acc_vicd = hts(vicd_qts)
        acc_pred = hts(pred_qts)

        # Create 3 separate subplots (excluding the first row)
        fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
        subplot_labels = ['Ax', 'Ay', 'Az']
        
        for i in range(1, 4):
            axes[i-1].plot(acc_vicd[i], label='Ground Truth', color='blue')
            axes[i-1].plot(acc_pred[i], label='Predicted', color='orange', linestyle='dashed')
            axes[i-1].set_ylabel(subplot_labels[i-1])
            axes[i-1].grid(True)
            axes[i-1].legend()

        axes[-1].set_xlabel('Time')
        plt.suptitle(f'Comparison of Predicted and Ground Truth Accelerations - Dataset {dataset_idx}')
        plt.tight_layout()

        save_path = os.path.join(save_folder, "acc_subplots.png")
        plt.savefig(save_path)
        plt.close()
        print(f'Plot saved to {save_path}')
    else: 

        acc_pred = hts(pred_qts)

        # Create 3 separate subplots (excluding the first row)
        fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
        subplot_labels = ['Ax', 'Ay', 'Az']
        
        for i in range(1, 4):
            axes[i-1].plot(acc_pred[i], label='Predicted', color='blue')
            axes[i-1].set_ylabel(subplot_labels[i-1])
            axes[i-1].grid(True)
            axes[i-1].legend()

        axes[-1].set_xlabel('Time')
        plt.suptitle(f'Predicted Accelerations - Dataset {dataset_idx}')
        plt.tight_layout()

        save_path = os.path.join(save_folder, "acc_subplots.png")
        plt.savefig(save_path)
        plt.close()
        print(f'Plot saved to {save_path}')



def run(dataset_idx, epochs=500):
    dataset_idx= dataset_idx
    calibrated_imu, vicd_data, camera_data= motion_calibration.calibrate_v2(dataset_idx,400)
    qts,exp_vals= motion_model_rollover(calibrated_imu)
    linear_acc= jnp.vstack((jnp.zeros(calibrated_imu.shape[1]), calibrated_imu[1:4,:]))
    
    
    from tqdm import tqdm
    norm_vec= jax.vmap(vector_normalize, in_axes=(1),out_axes=1)
    grads=[]
    
    pred_qts= qts
    step_size= 0.01
    epochs=epochs
    epsilon= 1e-6
    curr_step_size=step_size
    for epoch in tqdm(range(epochs)):
        cost=cost_fn(pred_qts,linear_acc,exp_vals)
        grad= jax.jacrev(cost_fn)(pred_qts,linear_acc,exp_vals)
        grad=jnp.where(jnp.isnan(grad), epsilon, grad)
        pred_qts=norm_vec(pred_qts- (grad*curr_step_size))
        pred_qts=pred_qts.at[:,0].set(jnp.array([1.,0.,0.,0.]))
        grads.append(cost)
        curr_step_size=step_size * jnp.exp(-0.01 * epoch)
    save_folder=os.path.join(data_base_path, "outputs", str(dataset_idx))
    os.makedirs(save_folder, exist_ok=True)
    plt.plot(grads)
    plt.xlabel('Epochs')  # Label for x-axis
    plt.ylabel('Cost')    # Label for y-axis
    plt.title('Cost vs Epochs')  # Optional: Add a title
    plt.savefig(os.path.join(save_folder, 'cost.png'))
    plt.close()
    
    plot_and_save_euler_angles_V3(pred_qts,calibrated_imu[0, :][None, :], vicd_data, save_folder, dataset_idx)
    return pred_qts,calibrated_imu[0,:] 
    

