
'''
Date: 2024 - WBF Group - Tongji University
Authors: Lin Shicong
Description:  
    (Using results from the previous-step deblending)
    Perform a blending noise simulation-subtraction operation 
    Update adjacent CRGs-assisted dataset
'''
import hdf5storage
import scipy.io as sio
import numpy as np

# Load original data and shooting time
mat_contents = hdf5storage.loadmat('./data/origin_data.mat')
pdb_data = mat_contents['pdb_data']
data = mat_contents['data']
Data_Shooting_he = mat_contents['Data_Shooting_he'].astype('int32').flatten()

# ================================ BNSS =================================
# Load last-step predictions
pred_test = hdf5storage.loadmat('./3c1_step1/data_pred.mat')['pred_test']
nr, nt, nx = pred_test.shape[0:]
print(f"pred_test.shape:{str(pred_test.shape)}", flush=True)

dt = 0.004
dx = 12
NN = 2 * nx
x = np.arange(0, nx) * dx / 1000 
t = np.arange(0, nt) * dt 

DataL = np.zeros((NN * nt, nx))

# Fill DataL
for j in range(nr):
    b1 = pred_test[j, :, :]
    DataL_tmp = np.zeros((NN * nt, ))
    for k in range(nx):
        a1 = Data_Shooting_he[k]
        if a1 != 0:
            DataL_tmp[a1-1:a1+nt-1] += b1[:, k]
    DataL[:, j] = DataL_tmp

pred_test_blending = np.zeros_like(pred_test)

# Fill pred_test_blending
for j in range(nr):
    b1 = DataL[:, j]
    Data_matrix = np.zeros((nt, nx))
    for k in range(nx):
        a1 = Data_Shooting_he[k]
        if a1 != 0:
            Data_matrix[:, k] = b1[a1-1:a1+nt-1]
    pred_test_blending[j, :, :] = Data_matrix

# Use original pdb_data to compute pred_test_bnss
pred_test_bnss = pdb_data - (pred_test_blending - pred_test)

# ================================ Dataset construction =================================
# Indices for training and validation
idx = np.arange(2, 256, 3)
valid_id = np.round(np.linspace(3, len(idx), 15)).astype(int) - 1
train_id = np.setdiff1d(np.arange(len(idx)), valid_id)

# Use these indices to get validation and training sets
valid_idx = idx[valid_id]
train_idx = np.concatenate(([1], idx[train_id], [256]))
test_idx = np.arange(1, 257)

# Initialize the arrays
hun_train = np.zeros((len(train_idx), 3, 256, 256))
hun_valid = np.zeros((len(valid_idx), 3, 256, 256))
hun_test = np.zeros((len(test_idx), 3, 256, 256))
hun_all1 = np.expand_dims(pred_test_bnss, axis=1)
data1 = data

# Extract training, validation, and test sets
data_train = np.zeros((len(train_idx), 256, 256))
for i, j in enumerate(train_idx):
    if j == 1:
        hun_train[i, 0, :, :] = hun_all1[j-1, 0, :, :]
        hun_train[i, 1, :, :] = hun_all1[j-1, 0, :, :]
        hun_train[i, 2, :, :] = hun_all1[j, 0, :, :]
    elif j == 256:
        hun_train[i, 0, :, :] = hun_all1[j-2, 0, :, :]
        hun_train[i, 1, :, :] = hun_all1[j-1, 0, :, :]
        hun_train[i, 2, :, :] = hun_all1[j-1, 0, :, :]
    else:
        hun_train[i, 0, :, :] = hun_all1[j-2, 0, :, :]
        hun_train[i, 1, :, :] = hun_all1[j-1, 0, :, :]
        hun_train[i, 2, :, :] = hun_all1[j, 0, :, :]
    data_train[i, :, :] = data1[j-1, :, :]

data_valid = np.zeros((len(valid_idx), 256, 256))
for i, j in enumerate(valid_idx):
    hun_valid[i, 0, :, :] = hun_all1[j-2, 0, :, :]
    hun_valid[i, 1, :, :] = hun_all1[j-1, 0, :, :]
    hun_valid[i, 2, :, :] = hun_all1[j, 0, :, :]
    data_valid[i, :, :] = data1[j-1, :, :]

data_test = np.zeros((len(test_idx), 256, 256))
for i, j in enumerate(test_idx):
    if j == 1:
        hun_test[i, 0, :, :] = hun_all1[j-1, 0, :, :]
        hun_test[i, 1, :, :] = hun_all1[j-1, 0, :, :]
        hun_test[i, 2, :, :] = hun_all1[j, 0, :, :]
    elif j == 256:
        hun_test[i, 0, :, :] = hun_all1[j-2, 0, :, :]
        hun_test[i, 1, :, :] = hun_all1[j-1, 0, :, :]
        hun_test[i, 2, :, :] = hun_all1[j-1, 0, :, :]
    else:
        hun_test[i, 0, :, :] = hun_all1[j-2, 0, :, :]
        hun_test[i, 1, :, :] = hun_all1[j-1, 0, :, :]
        hun_test[i, 2, :, :] = hun_all1[j, 0, :, :]
    data_test[i, :, :] = data1[j-1, :, :]

# Print sizes
print(f"hun_train shape: {hun_train.shape}")
print(f"hun_valid shape: {hun_valid.shape}")
print(f"hun_test shape: {hun_test.shape}")
print(f"data_train shape: {data_train.shape}")
print(f"data_valid shape: {data_valid.shape}")
print(f"data_test shape: {data_test.shape}")

# Save results if needed
sio.savemat('./3c1_step1/data_pred_bnss.mat', {
    'hun_train': hun_train,
    'hun_valid': hun_valid,
    'hun_test': hun_test,
    'data_train': data_train,
    'data_valid': data_valid,
    'data_test': data_test
})

print(f"Updated train, validation, and test datasets have been saved.")
