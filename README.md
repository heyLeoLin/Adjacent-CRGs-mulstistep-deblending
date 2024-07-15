<h1 align="center">Adjacent-CRGs-Multistep-Deblending</h1>

**Lin Shicong, Wang Benfeng***. (Tongji University)

<div align="center">
<img src=https://github.com/user-attachments/assets/40f9ff36-7bb0-4a83-bf9f-f2b1c764e649>
</div>

# Environment
Our code runs on:
* Python 3.8
* Pytorch 1.12
* CPU: Intel(R) Xeon(R) Gold 5115 @ 2.40GHz
* GPU: NVIDIA GeForce GTX 1080 Ti

# File Description
* :file_folder:**data**: Training/validation/test datasets of synthetic seismic data
* :page_facing_up:**main.py**: Main program for multistep deblending
* :page_facing_up:**bnss.py**: Blending noise simulation-subtraction operation
* :page_facing_up:**unet_tool.py**: Modified U-Net structure and SNR calculation function
* :page_facing_up:**warmup_tool.py**: Learning rate warm-up and cosine decay function

# Workflow
1. Use Google Drive link provided in :file_folder:**data** to obtain the datasets;
2. Run :page_facing_up:**main.py** first to get the first-step deblending results;
3. After the previous stage of deblending, run :page_facing_up:**bnss.py** to update the input dataset for the next stage of deblending;
4. (if needed) Modify parts of the commented code in :page_facing_up:**main.py** and use the updated dataset from :page_facing_up:**bnss.py** to perform the second step of deblending.
