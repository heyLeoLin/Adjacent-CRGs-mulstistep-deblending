'''
Author: Lin Shicong
Task: PDB Data Deblending (3CRG)
Description: 
- Architecture: lightweight U-Net (Dropout 0.5)
- Optimizer: AdamW with Warmup Cosine Annealing Learning Rate
- Stage: Step 1 (Supervised Learning)
'''

import os
import time
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import hdf5storage

# Local module imports (ensure these are included in your open-source repo)
from unet_tool import Unet, snr_fn2d
from warmup_tool import WarmupCosineAnnealingLR

# ==============================================================================
# 1. Initialization and Reproducibility Setup
# ==============================================================================
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==============================================================================
# 2. Hyperparameters & Configuration
# ==============================================================================
epochs = 150
batch_size = 2
initial_lr = 4e-3

# Setup directory for saving model weights and evaluation data
save_dir = "../result/3c1_step1"
os.makedirs(save_dir, exist_ok=True)

# ==============================================================================
# 3. Data Loading & Preprocessing
# ==============================================================================
print("Loading synthetic dataset...")

# Load training and validation sets
mat_contents = hdf5storage.loadmat('../data/dataset_3channel.mat')
x_train = mat_contents["hun_train"]
y_train = mat_contents["data_train"]
x_valid = mat_contents["hun_valid"]
y_valid = mat_contents["data_valid"]

# Add channel dimension for PyTorch compatibility (N, C, H, W)
y_train = np.expand_dims(y_train, 1)
y_valid = np.expand_dims(y_valid, 1)

# Extract dimensions
nr, nt, nx = x_train.shape[0], x_train.shape[2], x_train.shape[3]
print(f"Dimensions - nr: {nr}, nt: {nt}, nx: {nx}", flush=True)
print(f"x_train shape:\t{x_train.shape}", flush=True)
print(f"y_train shape:\t{y_train.shape}", flush=True)
print(f"x_valid shape:\t{x_valid.shape}", flush=True)
print(f"y_valid shape:\t{y_valid.shape}", flush=True)

# Convert numpy arrays to PyTorch tensors
x_train_tensor = torch.from_numpy(x_train.astype("float32"))
y_train_tensor = torch.from_numpy(y_train.astype("float32"))
x_valid_tensor = torch.from_numpy(x_valid.astype("float32"))
y_valid_tensor = torch.from_numpy(y_valid.astype("float32"))

# Construct Datasets and DataLoaders
dataset_train = TensorDataset(x_train_tensor, y_train_tensor)
dataset_valid = TensorDataset(x_valid_tensor, y_valid_tensor)

train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)

# Load test set
mat_contents_test = hdf5storage.loadmat('../data/dataset_3channel.mat')
x_test = mat_contents_test["hun_test"]
y_test = mat_contents_test["data_test"]
y_test = np.expand_dims(y_test, 1)

print(f"x_test shape:\t{x_test.shape}", flush=True)
print(f"y_test shape:\t{y_test.shape}", flush=True)

x_test_tensor = torch.from_numpy(x_test.astype("float32"))
y_test_tensor = torch.from_numpy(y_test.astype("float32"))
dataset_test = TensorDataset(x_test_tensor, y_test_tensor)
test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# ==============================================================================
# 4. Model Setup (U-Net, Optimizer, Scheduler, Loss)
# ==============================================================================
model = Unet(in_channel=3, out_channel=1).to(device)

optimizer = optim.AdamW(model.parameters(), lr=initial_lr, betas=(0.9, 0.95))
scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=epochs / 10, total_epochs=epochs, flag=0)

loss_fn = nn.MSELoss()

# ==============================================================================
# 5. Helper Functions
# ==============================================================================
def model_validate(model, dataloader, device):
    """Evaluates the model on a given dataloader to calculate MSE loss and SNR."""
    model.eval()
    total_snr, total_loss, total_samples = 0.0, 0.0, 0
    
    with torch.no_grad(): # Disable gradient calculation for evaluation
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            
            loss = loss_fn(y_hat, y)
            batch_size = y.shape[0]
            
            total_loss += loss.cpu().item() * batch_size
            total_snr += snr_fn2d(y_hat, y) * batch_size
            total_samples += batch_size
            
    model.train()
    return total_loss / total_samples, total_snr / total_samples

# ==============================================================================
# 6. Training Loop
# ==============================================================================
total_train_loss, total_test_loss = [], []
total_train_snr, total_test_snr = [], []
all_lr = []

start_time_global = time.time()
print("Starting training...", flush=True)

for epoch in range(epochs):
    # Fetch current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    
    start_time_epoch = time.time()

    # --- Training Phase ---
    model.train()
    for x, y in train_dataloader:
        x, y = x.to(device), y.to(device)

        # Forward pass
        y_hat = model(x)
        loss = loss_fn(y_hat, y) 

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Step the learning rate scheduler
    scheduler.step()

    # --- Evaluation Phase ---
    model.eval()
    with torch.no_grad():
        loss_train, snr_train = model_validate(model, train_dataloader, device)
        loss_test, snr_test = model_validate(model, valid_dataloader, device)


    # Record metrics
    total_train_loss.append(loss_train)
    total_train_snr.append(snr_train)
    total_test_loss.append(loss_test)
    total_test_snr.append(snr_test)
    all_lr.append(current_lr)

    # Calculate time metrics
    run_time = time.time() - start_time_epoch
    elapsed_time_min = (time.time() - start_time_global) / 60.0
    remain_time_min = run_time * (epochs - 1 - epoch) / 60.0

    # Print epoch summary
    show_str = (f"Epoch: {epoch + 1:03d} | "
                f"Train Loss: {loss_train:.5f} | Train SNR: {snr_train:.3f} | "
                f"Test Loss: {loss_test:.5f} | Test SNR: {snr_test:.3f} | "
                f"LR: {current_lr:.7f} | "
                f"Time/Epoch: {run_time:.2f}s | "
                f"Elapsed: {elapsed_time_min:.1f}m | ETA: {remain_time_min:.1f}m")
    print(show_str, flush=True)

# ==============================================================================
# 7. Save Models and Metrics
# ==============================================================================
sio.savemat(os.path.join(save_dir, 'data_para.mat'), {
    'loss_train': total_train_loss,
    'loss_test': total_test_loss,
    'snr_train': total_train_snr,
    'snr_test': total_test_snr,
    'lr': all_lr
})

torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
print("Model and parameters have been saved successfully.")

# ==============================================================================
# 8. Final Inference and Evaluation
# ==============================================================================
# Final evaluation on the entire training set without shuffling
pred_train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

_, final_snr_train = evaluate_test_loss(model, pred_train_dataloader, device)
print(f"Final Train SNR: {final_snr_train:.3f}", flush=True)

# Process test data using the trained network
print("Running inference on test dataset...", flush=True)
pred_test = np.empty((0, nt, nx))
model.eval()

total_snr_test2 = 0.0
total_samples_test2 = 0

with torch.no_grad():
    for x, y in test_dataloader:
        x, y = x.to(device), y.to(device)
        y_hat = model(x)

        batch_size = y.shape[0]
        total_snr_test2 += snr_fn2d(y_hat, y) * batch_size
        total_samples_test2 += batch_size

        # Format output predictions
        y_np = y_hat.cpu().detach().numpy()
        y_np = np.squeeze(y_np, axis=1) # Remove channel dimension
        pred_test = np.concatenate((pred_test, y_np), axis=0)

final_snr_test = total_snr_test2 / total_samples_test2
print(f"Final Test SNR: {final_snr_test:.3f}", flush=True)

# Save test predictions
sio.savemat(os.path.join(save_dir, 'data_pred.mat'), {'pred_test': pred_test})
print("Inference completed and predictions saved.")