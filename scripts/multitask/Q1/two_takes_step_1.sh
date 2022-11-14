export CUDA_DEVICE_ORDER=PCI_BUS_ID
EXP_NAME="two_takes_step_1"
# =========================================== CODET5 ================================================
PTM="Salesforce/codet5-base"
PTM_NAME="Salesforce-codet5-base"
CHECKPOINT_PATH="/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/${EXP_NAME}/${PTM_NAME}/best_model.ckpt"

# Train
CUDA_VISIBLE_DEVICES=0,2 python main_multitask.py -ptm $PTM --batch_size 12 --gpus 2 --prefix \
--tasks clone generation defect refine translate summarization -i 10000 \
--sum_lang javascript ruby go java python php \
--output_dir $EXP_NAME

# #=========================================== CODEBERT ================================================
PTM="microsoft/codebert-base"
PTM_NAME="microsoft-codebert-base"
CHECKPOINT_PATH="/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/${EXP_NAME}/${PTM_NAME}/best_model.ckpt"

# Train
CUDA_VISIBLE_DEVICES=0,2 python main_multitask.py -ptm $PTM --batch_size 10 --gpus 2 --prefix \
--tasks clone generation defect refine translate summarization -i 10000 \
--sum_lang javascript ruby go java python php \
--output_dir $EXP_NAME

#=========================================== GRAPHCODEBERT ================================================
PTM="microsoft/graphcodebert-base"
PTM_NAME="microsoft-graphcodebert-base"
CHECKPOINT_PATH="/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/${EXP_NAME}/${PTM_NAME}/best_model.ckpt"

# Train
CUDA_VISIBLE_DEVICES=0,2 python main_multitask.py -ptm $PTM --batch_size 10 --gpus 2 --prefix \
--tasks clone generation defect refine translate summarization -i 10000 \
--sum_lang javascript ruby go java python php \
--output_dir $EXP_NAME