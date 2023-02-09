export CUDA_DEVICE_ORDER=PCI_BUS_ID
PTM="microsoft/graphcodebert-base"
PTM_NAME="microsoft-graphcodebert-base"

# =========================================== REMOVING CLONE ================================================
EXP_NAME="removing_clone"
CHECKPOINT_PATH="/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/${EXP_NAME}/${PTM_NAME}/best_model.ckpt"

# Train
CUDA_VISIBLE_DEVICES=0,2 python main_multitask.py -ptm $PTM --batch_size 10 --gpus 2 --prefix \
--tasks codesearch code2test generation defect refine translate summarization -i 100000 \
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

# =========================================== REMOVING GENERATION ================================================
EXP_NAME="removing_generation"
CHECKPOINT_PATH="/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/${EXP_NAME}/${PTM_NAME}/best_model.ckpt"

# Train
CUDA_VISIBLE_DEVICES=0,2 python main_multitask.py -ptm $PTM --batch_size 10 --gpus 2 --prefix \
--tasks codesearch code2test clone defect refine translate summarization -i 100000 \
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

# =========================================== REMOVING DEFECT ================================================
EXP_NAME="removing_defect"
CHECKPOINT_PATH="/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/${EXP_NAME}/${PTM_NAME}/best_model.ckpt"

# Train
CUDA_VISIBLE_DEVICES=0,2 python main_multitask.py -ptm $PTM --batch_size 10 --gpus 2 --prefix \
--tasks codesearch code2test clone generation refine translate summarization -i 100000 \
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

# =========================================== REMOVING REFINE ================================================
EXP_NAME="removing_refine"
CHECKPOINT_PATH="/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/${EXP_NAME}/${PTM_NAME}/best_model.ckpt"

# Train
CUDA_VISIBLE_DEVICES=0,2 python main_multitask.py -ptm $PTM --batch_size 10 --gpus 2 --prefix \
--tasks codesearch code2test clone generation defect translate summarization -i 100000 \
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

# =========================================== REMOVING TRANSLATE ================================================
EXP_NAME="removing_translate"
CHECKPOINT_PATH="/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/${EXP_NAME}/${PTM_NAME}/best_model.ckpt"

# Train
CUDA_VISIBLE_DEVICES=0,2 python main_multitask.py -ptm $PTM --batch_size 10 --gpus 2 --prefix \
--tasks codesearch code2test clone generation defect refine summarization -i 100000 \
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

# =========================================== REMOVING SUMMARIZATION ================================================
EXP_NAME="removing_summarization"
CHECKPOINT_PATH="/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/${EXP_NAME}/${PTM_NAME}/best_model.ckpt"

# Train
CUDA_VISIBLE_DEVICES=0,2 python main_multitask.py -ptm $PTM --batch_size 10 --gpus 2 --prefix \
--tasks codesearch code2test clone generation defect refine translate -i 100000 \
--cs_lang javascript ruby go java python php \
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