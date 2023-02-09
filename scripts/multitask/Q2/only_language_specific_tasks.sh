export CUDA_DEVICE_ORDER=PCI_BUS_ID
EXP_NAME="only_java_tasks"
# =========================================== CODET5 ================================================
PTM="Salesforce/codet5-base"
PTM_NAME="Salesforce-codet5-base"
CHECKPOINT_PATH="/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/${EXP_NAME}/${PTM_NAME}/best_model.ckpt"

############################ JAVA
# Train
CUDA_VISIBLE_DEVICES=0,2 python main_multitask.py -ptm $PTM --batch_size 12 --gpus 2 --prefix \
--tasks refine translate generation summarization codesearch  -i 100000 --sum_lang java --cs_lang java \
--output_dir $EXP_NAME

# # Eval on codeserach
CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang java -ptm $PTM -mt --prefix

############################ PYTHON
EXP_NAME="only_python_tasks"
CHECKPOINT_PATH="/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/${EXP_NAME}/${PTM_NAME}/best_model.ckpt"

# Train
CUDA_VISIBLE_DEVICES=0,2 python main_multitask.py -ptm $PTM --batch_size 12 --gpus 2 --prefix \
--tasks code2test summarization codesearch  -i 100000 --sum_lang python --cs_lang python \
--output_dir $EXP_NAME

# Eval on code2test
CUDA_VISIBLE_DEVICES=0 python eval_code2test.py -ptm $PTM -ckpt $CHECKPOINT_PATH -mt --prefix

# Eval on codeserach
CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang python -ptm $PTM -mt --prefix