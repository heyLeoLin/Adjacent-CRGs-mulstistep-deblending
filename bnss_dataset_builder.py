'''
Author: Lin Shicong
Task: PDB Data Deblending - BNSS Operation
Description: 
- Performs a Blending Noise Simulation-Subtraction (BNSS) operation.
- Uses results from the previous-step deblending to estimate blending noise.
- Constructs and updates the dataset (supports both 3-CRG and 1-CRG formats).
'''

import os
import hdf5storage
import scipy.io as sio
import numpy as np

# ==============================================================================
# 1. Load Original & Synthetic Data
# ==============================================================================
print("Loading original synthetic data...")
mat_contents = hdf5storage.loadmat('../data/dataset_1channel.mat')
pdb_data = mat_contents['hun_test']
data = mat_contents['data_test']
data_shooting_he = mat_contents['Data_Shooting_he'].astype('int32').flatten()

# ==============================================================================
# 2. BNSS (Blending Noise Simulation-Subtraction) Operation
# ==============================================================================
print("Performing BNSS operation...")

# Load last-step predictions
save_dir = "../result/3c1_step1"
pred_test = hdf5storage.loadmat(os.path.join(save_dir, 'data_pred.mat'))['pred_test']

nr, nt, nx = pred_test.shape
print(f"pred_test shape: {pred_test.shape}", flush=True)

dt = 0.004
dx = 12
x = np.arange(0, nx) * dx / 1000 
t = np.arange(0, nt) * dt 

# Initialize arrays for blending simulation
data_matrix = np.zeros((nt, nx))
data_l_tmp = np.zeros((2 * nx * nt, ))
pred_test_blending = np.zeros_like(pred_test)

# Extract valid shooting indices
indx = np.where(data_shooting_he > 0)
data_shooting = data_shooting_he[indx]

# Simulate blending noise
for j in range(nr):
    b1 = pred_test[j, :, :]
    data_l_tmp.fill(0) 
    data_matrix.fill(0)

    for k, a1 in enumerate(data_shooting):
        data_l_tmp[a1-1 : a1+nt-1] += b1[:, k]

    for k, a1 in enumerate(data_shooting):
        data_matrix[:, k] = data_l_tmp[a1-1 : a1+nt-1]

    pred_test_blending[j, :, :] = data_matrix

# Compute BNSS: Original PDB data - (Simulated Blended Data - Predicted Data)
pred_test_bnss = pdb_data - (pred_test_blending - pred_test)

# ==============================================================================
# 3. Dataset Construction
# ==============================================================================
# INSTRUCTION: The following block constructs a 3-CRG (3-channel) dataset. 
# If you want to construct a 1-CRG dataset, COMMENT OUT the "3-CRG Block" below,
# and UNCOMMENT the "1-CRG Block" further down.
# ==============================================================================
print("Constructing Dataset...")

idx = np.arange(2, 256, 3)
valid_id = np.round(np.linspace(3, len(idx), 15)).astype(int) - 1
train_id = np.setdiff1d(np.arange(len(idx)), valid_id)

# Use these indices to get validation and training sets
valid_idx = idx[valid_id]
train_idx = np.concatenate(([1], idx[train_id], [256]))
test_idx = np.arange(1, nr + 1)

# -------------------------- [START OF 3-CRG BLOCK] --------------------------
# Initialize the arrays (N, C, H, W) where C=3
hun_train = np.zeros((len(train_idx), 3, nt, nx))
hun_valid = np.zeros((len(valid_idx), 3, nt, nx))
hun_test = np.zeros((len(test_idx), 3, nt, nx))

hun_all1 = np.expand_dims(pred_test_bnss, axis=1)
data1 = data

# Extract Training Set
data_train = np.zeros((len(train_idx), nt, nx))
for i, j in enumerate(train_idx):
    if j == 1:
        # Boundary padding for the first gather
        hun_train[i, 0, :, :] = hun_all1[j, 0, :, :]
        hun_train[i, 1, :, :] = hun_all1[j-1, 0, :, :]
        hun_train[i, 2, :, :] = hun_all1[j, 0, :, :]
    elif j == nr:
        # Boundary padding for the last gather
        hun_train[i, 0, :, :] = hun_all1[j-2, 0, :, :]
        hun_train[i, 1, :, :] = hun_all1[j-1, 0, :, :]
        hun_train[i, 2, :, :] = hun_all1[j-2, 0, :, :]
    else:
        # Normal 3 adjacent gathers
        hun_train[i, 0, :, :] = hun_all1[j-2, 0, :, :]
        hun_train[i, 1, :, :] = hun_all1[j-1, 0, :, :]
        hun_train[i, 2, :, :] = hun_all1[j, 0, :, :]
    data_train[i, :, :] = data1[j-1, :, :]

# Extract Validation Set
data_valid = np.zeros((len(valid_idx), nt, nx))
for i, j in enumerate(valid_idx):
    hun_valid[i, 0, :, :] = hun_all1[j-2, 0, :, :]
    hun_valid[i, 1, :, :] = hun_all1[j-1, 0, :, :]
    hun_valid[i, 2, :, :] = hun_all1[j, 0, :, :]
    data_valid[i, :, :] = data1[j-1, :, :]

# Extract Test Set
data_test = np.zeros((len(test_idx), nt, nx))
for i, j in enumerate(test_idx):
    if j == 1:
        hun_test[i, 0, :, :] = hun_all1[j, 0, :, :]
        hun_test[i, 1, :, :] = hun_all1[j-1, 0, :, :]
        hun_test[i, 2, :, :] = hun_all1[j, 0, :, :]
    elif j == nr:
        hun_test[i, 0, :, :] = hun_all1[j-2, 0, :, :]
        hun_test[i, 1, :, :] = hun_all1[j-1, 0, :, :]
        hun_test[i, 2, :, :] = hun_all1[j-2, 0, :, :]
    else:
        hun_test[i, 0, :, :] = hun_all1[j-2, 0, :, :]
        hun_test[i, 1, :, :] = hun_all1[j-1, 0, :, :]
        hun_test[i, 2, :, :] = hun_all1[j, 0, :, :]
    data_test[i, :, :] = data1[j-1, :, :]
# --------------------------- [END OF 3-CRG BLOCK] ---------------------------


# -------------------------- [START OF 1-CRG BLOCK] --------------------------
# INSTRUCTION: Uncomment this block (and comment the 3-CRG block above) 
# if you want to use the 1-CRG format.
# print("Constructing 1-CRG (1-Channel) Dataset...")

# data_train = data[train_idx - 1, :, :] 
# data_valid = data[valid_idx - 1, :, :]
# data_test = data[test_idx - 1, :, :]

# hun_train = pred_test_bnss[train_idx - 1, :, :]
# hun_valid = pred_test_bnss[valid_idx - 1, :, :]
# hun_test = pred_test_bnss[test_idx - 1, :, :]
# --------------------------- [END OF 1-CRG BLOCK] ---------------------------

# ==============================================================================
# 4. Save Results
# ==============================================================================
print(f"hun_train shape: {hun_train.shape}")
print(f"hun_valid shape: {hun_valid.shape}")
print(f"hun_test shape:  {hun_test.shape}")

sio.savemat(os.path.join(save_dir, 'data_pred_bnss.mat'), {
    'hun_train': hun_train,
    'hun_valid': hun_valid,
    'hun_test': hun_test
})

print(f"Successfully saved updated train, validation, and test datasets to {save_dir}/data_pred_bnss.mat")