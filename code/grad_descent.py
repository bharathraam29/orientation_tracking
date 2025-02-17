import quaternion_ops as qops
import jax
import jax.numpy as jnp
import quaternion_ops as qops
jax.config.update("jax_enable_x64", True)

def compute_rot_error(qt_1, qt):
    return  4 * ((jnp.linalg.norm(qops.qlog(qops.qmult(qops.qinverse(qt_1), qt))))**2)

def compute_pos_error(qt, acc_vec):
    h_qt=qops.qmult(qops.qmult(qops.qinverse(qt) , jnp.array([0.0,0.0,0.0,1.0])), qt)
    return jnp.linalg.norm(jnp.concatenate((jnp.array([0]), acc_vec), axis=0)-h_qt)**2


rot_error_vec = jax.vmap(compute_rot_error, in_axes=(1, 1))
pos_error_vec= jax.vmap(compute_pos_error, in_axes=(1,1))

def cost(pred_qts,calibrated_imu, rot_error_vec = jax.vmap(compute_rot_error, in_axes=(1, 1)),
pos_error_vec= jax.vmap(compute_pos_error, in_axes=(1))):
    
    cost_op=(rot_error_vec(pred_qts[:, :-1], pred_qts[:, 1:]).sum()+pos_error_vec(pred_qts[:, 1:],jnp.array(calibrated_imu[1:4,:])).sum())*0.5
    return cost_op

from tqdm import tqdm
def compute_new_preds(q, grad):
    new_q= q-(grad)
    return new_q/ jnp.linalg.norm(new_q)
    
    
def projected_grad_descent(pred_qts,calibrated_imu, step_size=1, epochs=2, stopping_criteria=0.001):
    new_pred_vec=jax.vmap(compute_new_preds)
    initial_step=step_size
    decay_rate=0.01
    cost_ls=[]
    for epoch in tqdm(range(epochs)):
        if epoch%10==0:
            # print( "COST::",cost(pred_qts,calibrated_imu))
            cost_ls.append(cost(pred_qts,calibrated_imu))
        if epoch%100==0:
            plt_graph_imu(calibrated_imu, pred_qts, epoch)
            
        grad=jax.jacrev(cost)(pred_qts,calibrated_imu) *step_size
        new_qts=new_pred_vec(pred_qts, grad, )
        # if check_threshold(pred_qts, new_qts):
        #     print("Stopped at epoch:: ", epoch)
        #     return new_qts
        pred_qts= new_qts
        step_size = initial_step / (1 + decay_rate * epoch)

    return pred_qts, cost_ls

import transforms3d as t3d

def plt_graph_imu(calibrated_imu, a, epoch):
    pred_roll=[0]* calibrated_imu.shape[-1]
    pred_pitch=[0]*  calibrated_imu.shape[-1] 
    pred_yaw =[0]* calibrated_imu.shape[-1]
    
    
    for idx in range(1,calibrated_imu.shape[-1]):    
        pred_roll[idx], pred_pitch[idx], pred_yaw[idx] = jnp.array(
    t3d.euler.mat2euler(t3d.quaternions.quat2mat(a[:, idx]), 'sxyz')
) * (180 / jnp.pi)
    import matplotlib.pyplot as plt
    # Create a figure and axis
    plt.figure(figsize=(8, 6))
    
    # Plot a line graph
    plt.plot(calibrated_imu[0,:],pred_roll, label='pred', color='red')
    
    # Adding title and labels
    plt.title(f'Epoch {epoch}')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Adding a legend
    plt.legend()
    
    # Show the plot
    plt.grid(True)
    plt.show()