
'''
Date: 2024 - WBF Group - Tongji University
Authors: Lin Shicong, Mo Tongtong
Description: Multistep deblending with adjacent CRGs-assisted dataset
'''
import os
import scipy.io as sio
import hdf5storage
import numpy as np
import time
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from unet_tool import *
from warmup_tool import *


# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(1)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ================================ set parameters ===============================
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

num_epochs = 150
batch_size = 2

# Create directory to save data & model
save_dir = "./3c1_step1"
# save_dir = "./3c1_step2"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ================================= load dataset ================================
mat_contents = hdf5storage.loadmat('./data/dataset_3channel.mat')

# Load updated dataset: using results of last-step deblending after BNSS (bnss.py)
# mat_contents = hdf5storage.loadmat('./3c1_step1/data_pred_bnss.mat')

# Extract data from the loaded .mat file
x_train = mat_contents["hun_train"]
y_train = mat_contents["data_train"]
x_valid = mat_contents["hun_valid"]
y_valid = mat_contents["data_valid"]
x_test = mat_contents["hun_test"]
y_test = mat_contents["data_test"]

# Expand channel dimension for 3-channel (1-channel) training pairs
# 1-channel
# x_train = np.expand_dims(x_train, 1)
# x_valid = np.expand_dims(x_valid, 1)
# x_test = np.expand_dims(x_test, 1)

# 3-channel
y_train = np.expand_dims(y_train, 1)
y_valid = np.expand_dims(y_valid, 1)
y_test = np.expand_dims(y_test, 1)

nr, nt, nx = x_train.shape[1:]
print(f"nr={nr}, nt={nt}, nx={nx}", flush=True)
print(f"x_train.shape:\t{str(x_train.shape)}", flush=True)
print(f"y_train.shape:\t{str(y_train.shape)}", flush=True)
print(f"x_valid.shape:\t{str(x_valid.shape)}", flush=True)
print(f"y_valid.shape:\t{str(y_valid.shape)}", flush=True)
print(f"x_test.shape:\t{str(x_test.shape)}", flush=True)
print(f"y_test.shape:\t{str(y_test.shape)}", flush=True)

# Transform arrays into tensors (share memory)
x_train_tensor = torch.from_numpy(x_train.astype("float32"))
y_train_tensor = torch.from_numpy(y_train.astype("float32"))
x_valid_tensor = torch.from_numpy(x_valid.astype("float32"))
y_valid_tensor = torch.from_numpy(y_valid.astype("float32"))
x_test_tensor = torch.from_numpy(x_test.astype("float32"))
y_test_tensor = torch.from_numpy(y_test.astype("float32"))

# Construct dataset & dataloader
dataset_train = TensorDataset(x_train_tensor, y_train_tensor)
dataset_valid = TensorDataset(x_valid_tensor, y_valid_tensor)
dataset_test = TensorDataset(x_test_tensor, y_test_tensor)
train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# ================================== unet parameter ==================================
# modified U-Net used
model = Unet2(in_channel=3, out_channel=1)  
model = model.to(device)

## Transfer learning: Initialize the model with parameters from the previous step
# state_dict = torch.load('./3c1_step1/model.pth')
# model.load_state_dict(state_dict)

lr = 4e-3
optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=num_epochs / 10, total_epochs=num_epochs, flag=0)
# (step2) lr:1e-3, flag=1; 
# (step3) lr:5e-4, flag=1.

loss_fn = nn.MSELoss()

# # ========================== Define network validate function =======================
def model_validate(model, dataloader, device):
    """
    Perform model validation on a given dataset.

    Args:
    - model: The trained model to evaluate.
    - dataloader: DataLoader providing the validation data.
    - device: CPU or GPU.

    Returns:
    - avg_loss (float): Average loss.
    - avg_snr (float): Average Signal-to-Noise Ratio.
    """
    total_loss = 0.0
    total_snr = 0.0
    n = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        y_pred = model(x)

        loss = loss_fn(y_pred, y)  # supervised

        total_loss += loss.cpu().item() * y.shape[0]
        total_snr += snr_fn2d(y_pred, y) * y.shape[0]
        n += y.shape[0]

    avg_loss = total_loss / n
    avg_snr = total_snr / n

    return avg_loss, avg_snr

# =================================== Network training ===================================
total_train_loss, total_valid_loss = [], []
total_train_snr, total_valid_snr = [], []
total_lr = []
start_time0 = time.time()

for epoch in range(num_epochs):

    for p in optimizer.param_groups:
        lr = p['lr']

    start_time = time.time()

    # Training step
    model.train()
    for x, y in train_dataloader:
        x, y = x.to(device), y.to(device)

        y_pred = model(x)

        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Update the learning rate
    scheduler.step()

    # Validation step
    model.eval()
    with torch.no_grad():
        train_loss, train_snr = model_validate(model, train_dataloader, device)
        valid_loss, valid_snr = model_validate(model, valid_dataloader, device)

    run_time = time.time() - start_time
    already_time = (time.time() - start_time0) / 60.0
    remain_time = run_time * (num_epochs - 1 - epoch) / 60.0

    show_str = f"epoch:{epoch + 1}"
    show_str += f"   loss_train: {train_loss:.5f}"
    show_str += f"   snr_train: {train_snr:.3f}"
    show_str += f"   loss_valid: {valid_loss:.5f}"
    show_str += f"   snr_valid: {valid_snr:.3f}"
    show_str += f"   lr: {lr:.7f}"
    show_str += f"   time: {run_time:.1f}(sec)"
    show_str += f"   already_time: {already_time:.1f}(min)"
    show_str += f"   remain_time: {remain_time:.1f}(min)"
    print(show_str, flush=True)

    total_train_loss.append(train_loss)
    total_train_snr.append(train_snr)
    total_valid_loss.append(valid_loss)
    total_valid_snr.append(valid_snr)
    total_lr.append(lr)

# Save parameters curves
sio.savemat(save_dir + '/data_para.mat',
            {'loss_train': total_train_loss,
             'loss_test': total_valid_loss,
             'snr_train': total_train_snr,
             'snr_test': total_valid_snr,
             'lr': total_lr})

# Save network model
torch.save(model.state_dict(), save_dir + "/model.pth")
print("Curves and model have been saved")

# ============================ process data using the pre-trained net =============================
def testing(model, dataloader, device):
    model.eval()
    predictions = np.empty((0, nt, nx))
    snr, n = 0.0, 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            y_pred = model(x)

            snr += snr_fn2d(y_pred, y) * y.shape[0]
            n += y.shape[0]

            y_np = y_pred.cpu().detach().numpy()
            y_np = np.squeeze(y_np, axis=1)
            predictions = np.concatenate((predictions, y_np), axis=0)

        avg_snr = snr / n

    return avg_snr, predictions

# Evaluate on training dataset
train_dataloader2 = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
snr_train, pred_train = testing(model, train_dataloader2, device)
print(f"Train SNR: {snr_train:.3f}", flush=True)

# Evaluate on valid dataset
snr_valid, pred_valid = testing(model, valid_dataloader, device)
print(f"Valid SNR: {snr_valid:.3f}", flush=True)

# Evaluate on testing dataset
snr_test, pred_test = testing(model, test_dataloader, device)
print(f"Test SNR: {snr_test:.3f}", flush=True)

## Save predictions
sio.savemat(save_dir + '/data_pred.mat', {'pred_test': pred_test})
print("predictions have been saved")
