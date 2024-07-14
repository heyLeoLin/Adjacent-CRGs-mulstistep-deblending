<h1 align="center">Deep learning-based multistep deblending with adjacent common receiver gathers constraint</h1>

**Lin Shicong, Wang Benfeng**

# Abstract:
*<div align="justify">Considering the consistency of seismic events in adjacent common receiver gathers (CRGs) can contribute to better deblending performance, we propose an adjacent CRGs-assisted multistep deblending algorithm. Three adjacent CRGs after pseudo-deblending are used to form a three-channel input to enhance the signal coherence of the middle CRG, and the desired single-channel output is the corresponding unblended middle CRG. Additionally, a modified U-Net training framework is designed with the AdamW optimizer, incorporating learning rate warm-up and cosine decay strategies. A multistep strategy with blending noise simulation-subtraction is also implemented to progressively attenuate blending noise and update the training input with signal preservation.</div>*

# Python requirements

# File Descriptions and Workflow
1. :file_folder:**data**: Use the provided Google Drive link to obtain the datasets
2. :page_facing_up:**main.py**: Main program for multistep deblending. Run for the first time to get the first-step deblending result
4. :page_facing_up:**bnss.py**: Blending noise simulation-subtraction operation. After the previous stage of deblending, update the input dataset for the next stage of deblending
5. (if needed): Modify parts of the commented code in `main.py` and use the updated dataset from `bnss.py` to perform the second step of deblending
* :page_facing_up:**unet_tool.py**: Modified U-Net structure and SNR calculation functions
* :page_facing_up:**warmup_tool.py**: Learning rate warm-up and cosine decay function
