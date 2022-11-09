export CUDA_DEVICE_ORDER=PCI_BUS_ID
EXP_NAME="one_take"
# =========================================== CODET5 ================================================
PTM="Salesforce/codet5-base"
PTM_NAME="Salesforce-codet5-base"
CHECKPOINT_PATH="/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/${EXP_NAME}/${PTM_NAME}/best_model.ckpt"

# Train
CUDA_VISIBLE_DEVICES=0,2 python main_multitask.py -ptm $PTM --batch_size 12 --gpus 2 --prefix \
--tasks codesearch code2test clone generation defect refine translate summarization -i 10000 \
--cs_lang javascript ruby go java python php --sum_lang javascript ruby go java python php \
--output_dir $EXP_NAME

# Eval on codeserach
CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang go -ptm $PTM -mt --prefix
CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang java -ptm $PTM -mt --prefix
CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang javascript -ptm $PTM -mt --prefix
CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang php -ptm $PTM -mt --prefix
CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang python -ptm $PTM -mt --prefix
CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang ruby -ptm $PTM -mt --prefix
# Eval on code2test
CUDA_VISIBLE_DEVICES=0 python eval_code2test.py -ptm $PTM -ckpt $CHECKPOINT_PATH -mt --prefix

# #=========================================== CODEBERT ================================================
PTM="microsoft/codebert-base"
PTM_NAME="microsoft-codebert-base"
CHECKPOINT_PATH="/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/${EXP_NAME}/${PTM_NAME}/best_model.ckpt"

# Train
CUDA_VISIBLE_DEVICES=0,2 python main_multitask.py -ptm $PTM --batch_size 10 --gpus 2 --prefix \
--tasks codesearch code2test clone generation defect refine translate summarization -i 10000 \
--cs_lang javascript ruby go java python php --sum_lang javascript ruby go java python php \
--output_dir $EXP_NAME

# Eval on codeserach
CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang go -ptm $PTM -mt --prefix
CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang java -ptm $PTM -mt --prefix
CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang javascript -ptm $PTM -mt --prefix
CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang php -ptm $PTM -mt --prefix
CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang python -ptm $PTM -mt --prefix
CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang ruby -ptm $PTM -mt --prefix
Eval on code2test
CUDA_VISIBLE_DEVICES=0 python eval_code2test.py -ptm $PTM -ckpt $CHECKPOINT_PATH -mt --prefix

#=========================================== GRAPHCODEBERT ================================================
PTM="microsoft/graphcodebert-base"
PTM_NAME="microsoft-graphcodebert-base"
CHECKPOINT_PATH="/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/${EXP_NAME}/${PTM_NAME}/best_model.ckpt"

# Train
CUDA_VISIBLE_DEVICES=0,2 python main_multitask.py -ptm $PTM --batch_size 10 --gpus 2 --prefix \
--tasks codesearch code2test clone generation defect refine translate summarization -i 10000 \
--cs_lang javascript ruby go java python php --sum_lang javascript ruby go java python php \
--output_dir $EXP_NAME

# Eval on codeserach
CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang go -ptm $PTM -mt --prefix
CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang java -ptm $PTM -mt --prefix
CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang javascript -ptm $PTM -mt --prefix
CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang php -ptm $PTM -mt --prefix
CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang python -ptm $PTM -mt --prefix
CUDA_VISIBLE_DEVICES=0 python eval_codesearch.py -ckpt $CHECKPOINT_PATH -lang ruby -ptm $PTM -mt --prefix
# Eval on code2test
CUDA_VISIBLE_DEVICES=0 python eval_code2test.py -ptm $PTM -ckpt $CHECKPOINT_PATH -mt --prefix