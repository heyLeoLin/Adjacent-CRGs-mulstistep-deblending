<h1 align="center">Adjacent CRGs-assisted mulstistep deblending</h1>

 Material for **Deep learning-based multistep deblending with adjacent common receiver gathers constraint - Lin Shicong, Wang Benfeng.**

# File Description and Workflow
1. **data**: Use the provided Google Drive link to obtain the datasets
2. **main.py**: Main program for multistep deblending. Run for the first time to get the first-step deblending result
4. **bnss.py**: Blending noise simulation-subtraction operation. After the previous stage of deblending, update the input dataset for the next stage of deblending
5. (if needed)**main.py**: Modify parts of the commented code and use the updated dataset from `bnss.py` to perform the second step of deblending
* **unet_tool.py**: Modified U-Net structure and SNR calculation functions
* **warmup_tool.py**: Learning rate warm-up and cosine decay function
