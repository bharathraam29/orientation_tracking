import numpy as np
import cv2
import runner
import motion_calibration
import load_data
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import os
from load_data import load_dataset
import transforms3d as t3d

data_base_path= "/home/bradhakrishnan/ECE276A_PR1/data/"


def project_to_panorama(image, R, W_pan=1960, H_pan=960, fov_x=60, fov_y=45):
    H, W, _ = image.shape

    fov_x_rad = np.radians(fov_x)
    fov_y_rad = np.radians(fov_y)


    x = np.linspace(-0.5, 0.5, W) * fov_x_rad  
    y = np.linspace(-0.5, 0.5, H) * fov_y_rad  

    lambda_grid, phi_grid = np.meshgrid(x, y)

    X = np.cos(phi_grid) * np.cos(lambda_grid)
    Y = np.cos(phi_grid) * np.sin(lambda_grid)
    Z = np.sin(phi_grid)

    coords = np.stack([X, Y, Z], axis=-1) 
    coords_world = coords @ R.T  

    # Convert world coordinates to spherical coordinates
    r = np.sqrt(coords_world[..., 0]**2 + coords_world[..., 1]**2)
    lambda_world = np.arctan2(coords_world[..., 1], coords_world[..., 0])
    phi_world = np.arctan2(coords_world[..., 2], r)

    # Convert to cylindrical panorama coordinates
    x_pan = ((lambda_world + np.pi) / (2 * np.pi) * W_pan).astype(int)
    y_pan = ((phi_world + (np.pi / 2)) / np.pi * H_pan).astype(int)

    panorama = np.zeros((H_pan, W_pan, 3), dtype=np.uint8)

    # Map image pixels to the panorama
    valid_idx = (y_pan >= 0) & (y_pan < H_pan) & (x_pan >= 0) & (x_pan < W_pan)
    # Flatten arrays to avoid shape mismatches
    x_pan_flat = x_pan.ravel()[valid_idx.ravel()]
    y_pan_flat = y_pan.ravel()[valid_idx.ravel()]
    image_flat = image.reshape(-1, 3)[valid_idx.ravel()]

    # Assign pixels to the panorama
    panorama[y_pan_flat, x_pan_flat] = image_flat

    return panorama

def merge_images(image_list, R_list, W_pan=2048, H_pan=1024):
    """
    Merges multiple images into a panoramic view based on their corresponding rotations.
    """
    # Initialize the panorama canvas
    final_panorama = np.zeros((H_pan, W_pan, 3), dtype=np.uint8)

    for image, R in zip(image_list, R_list):
        projected = project_to_panorama(image, R, W_pan, H_pan)
        mask = projected > 0 
        final_panorama[mask] = projected[mask]

    return final_panorama  

dataset_idx= 10
imud,vicd, cam=load_dataset(dataset_idx)
if not cam:
    print("No cam data")
else:
    images=[cam['cam'][:,:,:,i] for i in range(cam['cam'].shape[-1])]
    if vicd:
        cam_ts_2_vicd_ts_idx = {}
        for cam_ts in cam['ts'][0]:
            temp = np.abs(vicd['ts'][0] - cam_ts)
            idx = np.argmin(temp)
            cam_ts_2_vicd_ts_idx[cam_ts] = idx
        R_matrices= [vicd['rots'][:, :, cam_ts_2_vicd_ts_idx[key]] for key in cam_ts_2_vicd_ts_idx.keys()]
    else:
        print("NO VICON DATA, Running Estimation..")
        pred_qts, ts= runner.run(dataset_idx=dataset_idx,epochs=500)
        matrices= [t3d.quaternions.quat2mat(pred_qts[:,ii]) for ii in range(pred_qts.shape[-1])]
        cam_ts_2_vicd_ts_idx = {}
        for cam_ts in cam['ts'][0]:
            temp = np.abs(ts - cam_ts)
            idx = np.argmin(temp)
            cam_ts_2_vicd_ts_idx[cam_ts] = idx
        R_matrices= [matrices[cam_ts_2_vicd_ts_idx[key]] for key in cam_ts_2_vicd_ts_idx.keys()]
    
    save_folder = os.path.join(data_base_path, "outputs", str(dataset_idx))
    os.makedirs(save_folder, exist_ok=True)  # Ensure the folder exists
    save_path = os.path.join(save_folder, "pan.jpg")
    
    # Generate and save panorama
    panorama = merge_images(images, R_matrices)
    plt.figure(figsize=(15, 7))
    plt.imshow(panorama)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)  # Save the figure
    plt.close()  # Close the figure to free memory
    
    print(f"Panorama saved at: {save_path}")