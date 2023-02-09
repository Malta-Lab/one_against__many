export CUDA_DEVICE_ORDER=PCI_BUS_ID
# EXP_NAME="only_codesearchnet_specific_javascript"
# # =========================================== JAVASCRIPT ================================================
PTM="Salesforce/codet5-base"
PTM_NAME="Salesforce-codet5-base"
CHECKPOINT_PATH="/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/${EXP_NAME}/${PTM_NAME}/best_model.ckpt"

# Train
CUDA_VISIBLE_DEVICES=0,2 python main_multitask.py -ptm $PTM --batch_size 12 --gpus 2 --prefix \
--tasks codesearch summarization -i 50000 \
--cs_lang javascript --sum_lang javascript \
--output_dir $EXP_NAME

# Eval on codeserach
CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang javascript -ptm $PTM -mt --prefix

# # =========================================== JAVA ================================================
# EXP_NAME="only_codesearchnet_specific_java"
# CHECKPOINT_PATH="/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/${EXP_NAME}/${PTM_NAME}/best_model.ckpt"

# # Train
# CUDA_VISIBLE_DEVICES=0,2 python main_multitask.py -ptm $PTM --batch_size 12 --gpus 2 --prefix \
# --tasks codesearch summarization -i 100000 \
# --cs_lang java --sum_lang java \
# --output_dir $EXP_NAME

# # Eval on codeserach
# CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang java -ptm $PTM -mt --prefix

# # =========================================== GO ================================================
# EXP_NAME="only_codesearchnet_specific_go"
# CHECKPOINT_PATH="/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/${EXP_NAME}/${PTM_NAME}/best_model.ckpt"

# # Train
# CUDA_VISIBLE_DEVICES=0,2 python main_multitask.py -ptm $PTM --batch_size 12 --gpus 2 --prefix \
# --tasks codesearch summarization -i 100000 \
# --cs_lang go --sum_lang go \
# --output_dir $EXP_NAME

# # Eval on codeserach
# CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang go -ptm $PTM -mt --prefix

# # =========================================== PYTHON ================================================
# EXP_NAME="only_codesearchnet_specific_python"
# CHECKPOINT_PATH="/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/${EXP_NAME}/${PTM_NAME}/best_model.ckpt"

# # Train
# CUDA_VISIBLE_DEVICES=0,2 python main_multitask.py -ptm $PTM --batch_size 12 --gpus 2 --prefix \
# --tasks codesearch summarization -i 100000 \
# --cs_lang python --sum_lang python \
# --output_dir $EXP_NAME

# # Eval on codeserach
# CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang python -ptm $PTM -mt --prefix

# # =========================================== RUBY ================================================
# EXP_NAME="only_codesearchnet_specific_ruby"
# CHECKPOINT_PATH="/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/${EXP_NAME}/${PTM_NAME}/best_model.ckpt"

# # Train
# CUDA_VISIBLE_DEVICES=0,2 python main_multitask.py -ptm $PTM --batch_size 12 --gpus 2 --prefix \
# --tasks codesearch summarization -i 20000 \
# --cs_lang ruby --sum_lang ruby \
# --output_dir $EXP_NAME

# # Eval on codeserach
# CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang ruby -ptm $PTM -mt --prefix

# # =========================================== PHP ================================================
# EXP_NAME="only_codesearchnet_specific_php"
# CHECKPOINT_PATH="/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/${EXP_NAME}/${PTM_NAME}/best_model.ckpt"

# # Train
# CUDA_VISIBLE_DEVICES=0,2 python main_multitask.py -ptm $PTM --batch_size 12 --gpus 2 --prefix \
# --tasks codesearch summarization -i 100000 \
# --cs_lang php --sum_lang php \
# --output_dir $EXP_NAME

# # Eval on codeserach
# CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang php -ptm $PTM -mt --prefix