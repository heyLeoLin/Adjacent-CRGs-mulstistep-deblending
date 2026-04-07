<h1 align="center">
  Adjacent-CRGs-Multistep-Deblending
</h1>

  **Shicong Lin, Benfeng Wang***. (Tongji University)

<div align="center">
<img width="3608" height="1460" alt="architecture" src="https://github.com/user-attachments/assets/965151ad-77be-478f-bcac-8e3def842a7c" />
</div>

# Environment & requirements
Our code is developed and tested on the following hardware:
* CPU: Intel(R) Xeon(R) Gold 5115 @ 2.40GHz
* GPU: NVIDIA GeForce GTX 1080 Ti
  
To ensure exact reproducibility, please use Python 3.8 and install the required dependencies. You can easily set up the environment using the provided `requirements.txt`:
>pip install -r requirements.txt

(Note: Depending on your specific hardware and CUDA version, you may need to install the GPU-compatible version of PyTorch manually via the official PyTorch website.)

# File Description
* :file_folder:**data**: Training/validation/test datasets of synthetic seismic data.
* :page_facing_up:**main_step1.py/main_step2.py/main_step3.py**: Main execution scripts for the multistep deblending process.
* :page_facing_up:**bnss_dataset_builder.py**: Script for the Blending Noise Simulation-Subtraction (BNSS) operation to update datasets.
* :page_facing_up:**unet_tool.py**: Defines the modified U-Net architecture and the SNR evaluation metric.
* :page_facing_up:**warmup_tool.py**: Learning rate scheduler featuring warm-up and cosine annealing decay.
* :file_folder:**result**: The optimized model weights from the step1 and step2.

# Workflow
1. **Prepare Dataset**: Download the required datasets using the Google Drive link provided in the :file_folder:**data** to obtain the datasets.
2. **Stage 1 deblending**: Run :page_facing_up:**main_step1.py** to train the initial model and obtain the first-step deblending results.
3. **Update Dataset (BNSS)**: Once Step 1 is complete, run :page_facing_up:**bnss_dataset_builder.py**. This script utilizes the predictions from the previous stage to simulate and subtract blending noise, generating an updated dataset for the next deblending stage.
4. **Stage 2 & 3 deblending**: Run :page_facing_up:**main_step2** (and subsequently :page_facing_up:**main_step3.py**) using the newly generated dataset to perform further deblending and refinement.

💡 **Note for Step 2/Step 3**:
When executing the subsequent stages, please ensure you manually adjust the following parameters and paths in the scripts to load the previous outputs properly:
* Hyperparameters: Adjust the number of epochs `num_epochs` and learning rate `lr`;
* Input Data Path: Point to the newly built dataset (e.g.,`hdf5storage.loadmat('./result/3c1_step1/data_pred_bnss.mat')`);
* Pre-trained Weights: Load the optimized model weights from the prior step (e.g.,`state_dict = torch.load('./result/3c1_step1/model.pth')`).
