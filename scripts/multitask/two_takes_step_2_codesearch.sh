EXP_NAME='two_takes_step_2'
### WARNING: THE BATCH SIZE FOR BERT MODELS IS ORIGINALLY 32, HERE I USE THE SAME OF CODET5 MODEL (16)
# ===================================== CODEBERT =================================
CKPT_PATH='/home/parraga/projects/_masters/multitask_code/checkpoints/codesearch'
BASE_CKPT='/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/two_takes_step_1'
MODEL='microsoft-codebert-base'
PTM='microsoft/codebert-base'

LANG='javascript'
CUDA_VISIBLE_DEVICES=1,3 python main_codesearch.py -lang $LANG --gpus 2 -ptm $PTM --scheduler linear \
-ckpt $BASE_CKPT/$MODEL/best_model.ckpt --output_dir $EXP_NAME -bs 16 --prefix
CUDA_VISIBLE_DEVICES=3 python eval_codesearch.py --is_multitask -ckpt $CKPT_PATH/$EXP_NAME/$LANG/$MODEL/best_model.ckpt -lang $LANG -ptm $PTM --prefix

LANG='go'
CUDA_VISIBLE_DEVICES=1,3 python main_codesearch.py -lang $LANG --gpus 2 -ptm $PTM  \
-ckpt $BASE_CKPT/$MODEL/best_model.ckpt --output_dir $EXP_NAME -bs 16 --prefix
CUDA_VISIBLE_DEVICES=3 python eval_codesearch.py --is_multitask -ckpt $CKPT_PATH/$EXP_NAME/$LANG/$MODEL/best_model.ckpt -lang $LANG -ptm $PTM --prefix

LANG='java'
CUDA_VISIBLE_DEVICES=1,3 python main_codesearch.py -lang $LANG --gpus 2 -ptm $PTM  \
-ckpt $BASE_CKPT/$MODEL/best_model.ckpt --output_dir $EXP_NAME -bs 16 --prefix
CUDA_VISIBLE_DEVICES=3 python eval_codesearch.py --is_multitask -ckpt $CKPT_PATH/$EXP_NAME/$LANG/$MODEL/best_model.ckpt -lang $LANG -ptm $PTM --prefix

LANG='php'
CUDA_VISIBLE_DEVICES=1,3 python main_codesearch.py -lang $LANG --gpus 2 -ptm $PTM  \
-ckpt $BASE_CKPT/$MODEL/best_model.ckpt --output_dir $EXP_NAME -bs 16 --prefix
CUDA_VISIBLE_DEVICES=3 python eval_codesearch.py --is_multitask -ckpt $CKPT_PATH/$EXP_NAME/$LANG/$MODEL/best_model.ckpt -lang $LANG -ptm $PTM --prefix

LANG='ruby'
CUDA_VISIBLE_DEVICES=1,3 python main_codesearch.py -lang $LANG --gpus 2 -ptm $PTM  \
-ckpt $BASE_CKPT/$MODEL/best_model.ckpt --output_dir $EXP_NAME -bs 16 --prefix
CUDA_VISIBLE_DEVICES=3 python eval_codesearch.py --is_multitask -ckpt $CKPT_PATH/$EXP_NAME/$LANG/$MODEL/best_model.ckpt -lang $LANG -ptm $PTM --prefix

LANG='python'
CUDA_VISIBLE_DEVICES=1,3 python main_codesearch.py -lang $LANG --gpus 2 -ptm $PTM  \
-ckpt $BASE_CKPT/$MODEL/best_model.ckpt --output_dir $EXP_NAME -bs 16 --prefix
CUDA_VISIBLE_DEVICES=3 python eval_codesearch.py --is_multitask -ckpt $CKPT_PATH/$EXP_NAME/$LANG/$MODEL/best_model.ckpt -lang $LANG -ptm $PTM --prefix

# ===================================== CODET5 =================================
CKPT_PATH='/home/parraga/projects/_masters/multitask_code/checkpoints/codesearch'
BASE_CKPT='/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/two_takes_step_1'
MODEL='Salesforce-codet5-base'
PTM='Salesforce/codet5-base'

LANG='javascript'
CUDA_VISIBLE_DEVICES=1,3 python main_codesearch.py -lang $LANG --gpus 2 -ptm $PTM --scheduler linear \
-ckpt $BASE_CKPT/$MODEL/best_model.ckpt --output_dir $EXP_NAME -bs 16 --prefix
CUDA_VISIBLE_DEVICES=3 python eval_codesearch.py --is_multitask -ckpt $CKPT_PATH/$EXP_NAME/$LANG/$MODEL/best_model.ckpt -lang $LANG -ptm $PTM --prefix

LANG='go'
CUDA_VISIBLE_DEVICES=1,3 python main_codesearch.py -lang $LANG --gpus 2 -ptm $PTM  \
-ckpt $BASE_CKPT/$MODEL/best_model.ckpt --output_dir $EXP_NAME -bs 16 --prefix
CUDA_VISIBLE_DEVICES=3 python eval_codesearch.py --is_multitask -ckpt $CKPT_PATH/$EXP_NAME/$LANG/$MODEL/best_model.ckpt -lang $LANG -ptm $PTM --prefix

LANG='java'
CUDA_VISIBLE_DEVICES=1,3 python main_codesearch.py -lang $LANG --gpus 2 -ptm $PTM  \
-ckpt $BASE_CKPT/$MODEL/best_model.ckpt --output_dir $EXP_NAME -bs 16 --prefix
CUDA_VISIBLE_DEVICES=3 python eval_codesearch.py --is_multitask -ckpt $CKPT_PATH/$EXP_NAME/$LANG/$MODEL/best_model.ckpt -lang $LANG -ptm $PTM --prefix

LANG='php'
CUDA_VISIBLE_DEVICES=1,3 python main_codesearch.py -lang $LANG --gpus 2 -ptm $PTM  \
-ckpt $BASE_CKPT/$MODEL/best_model.ckpt --output_dir $EXP_NAME -bs 16 --prefix
CUDA_VISIBLE_DEVICES=3 python eval_codesearch.py --is_multitask -ckpt $CKPT_PATH/$EXP_NAME/$LANG/$MODEL/best_model.ckpt -lang $LANG -ptm $PTM --prefix

LANG='ruby'
CUDA_VISIBLE_DEVICES=1,3 python main_codesearch.py -lang $LANG --gpus 2 -ptm $PTM  \
-ckpt $BASE_CKPT/$MODEL/best_model.ckpt --output_dir $EXP_NAME -bs 16 --prefix
CUDA_VISIBLE_DEVICES=3 python eval_codesearch.py --is_multitask -ckpt $CKPT_PATH/$EXP_NAME/$LANG/$MODEL/best_model.ckpt -lang $LANG -ptm $PTM --prefix

LANG='python'
CUDA_VISIBLE_DEVICES=1,3 python main_codesearch.py -lang $LANG --gpus 2 -ptm $PTM  \
-ckpt $BASE_CKPT/$MODEL/best_model.ckpt --output_dir $EXP_NAME -bs 16 --prefix
CUDA_VISIBLE_DEVICES=3 python eval_codesearch.py --is_multitask -ckpt $CKPT_PATH/$EXP_NAME/$LANG/$MODEL/best_model.ckpt -lang $LANG -ptm $PTM --prefix

# ===================================== GRAPHCODEBERT =================================
CKPT_PATH='/home/parraga/projects/_masters/multitask_code/checkpoints/codesearch'
BASE_CKPT='/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/two_takes_step_1'
MODEL='microsoft-graphcodebert-base'
PTM='microsoft/graphcodebert-base'

LANG='javascript'
CUDA_VISIBLE_DEVICES=1,3 python main_codesearch.py -lang $LANG --gpus 2 -ptm $PTM --scheduler linear \
-ckpt $BASE_CKPT/$MODEL/best_model.ckpt --output_dir $EXP_NAME -bs 16 --prefix
CUDA_VISIBLE_DEVICES=3 python eval_codesearch.py --is_multitask -ckpt $CKPT_PATH/$EXP_NAME/$LANG/$MODEL/best_model.ckpt -lang $LANG -ptm $PTM --prefix

LANG='go'
CUDA_VISIBLE_DEVICES=1,3 python main_codesearch.py -lang $LANG --gpus 2 -ptm $PTM  \
-ckpt $BASE_CKPT/$MODEL/best_model.ckpt --output_dir $EXP_NAME -bs 16 --prefix
CUDA_VISIBLE_DEVICES=3 python eval_codesearch.py --is_multitask -ckpt $CKPT_PATH/$EXP_NAME/$LANG/$MODEL/best_model.ckpt -lang $LANG -ptm $PTM --prefix

LANG='java'
CUDA_VISIBLE_DEVICES=1,3 python main_codesearch.py -lang $LANG --gpus 2 -ptm $PTM  \
-ckpt $BASE_CKPT/$MODEL/best_model.ckpt --output_dir $EXP_NAME -bs 16 --prefix
CUDA_VISIBLE_DEVICES=3 python eval_codesearch.py --is_multitask -ckpt $CKPT_PATH/$EXP_NAME/$LANG/$MODEL/best_model.ckpt -lang $LANG -ptm $PTM --prefix

LANG='php'
CUDA_VISIBLE_DEVICES=1,3 python main_codesearch.py -lang $LANG --gpus 2 -ptm $PTM  \
-ckpt $BASE_CKPT/$MODEL/best_model.ckpt --output_dir $EXP_NAME -bs 16 --prefix
CUDA_VISIBLE_DEVICES=3 python eval_codesearch.py --is_multitask -ckpt $CKPT_PATH/$EXP_NAME/$LANG/$MODEL/best_model.ckpt -lang $LANG -ptm $PTM --prefix

LANG='ruby'
CUDA_VISIBLE_DEVICES=1,3 python main_codesearch.py -lang $LANG --gpus 2 -ptm $PTM  \
-ckpt $BASE_CKPT/$MODEL/best_model.ckpt --output_dir $EXP_NAME -bs 16 --prefix
CUDA_VISIBLE_DEVICES=3 python eval_codesearch.py --is_multitask -ckpt $CKPT_PATH/$EXP_NAME/$LANG/$MODEL/best_model.ckpt -lang $LANG -ptm $PTM --prefix

LANG='python'
CUDA_VISIBLE_DEVICES=1,3 python main_codesearch.py -lang $LANG --gpus 2 -ptm $PTM  \
-ckpt $BASE_CKPT/$MODEL/best_model.ckpt --output_dir $EXP_NAME -bs 16 --prefix
CUDA_VISIBLE_DEVICES=3 python eval_codesearch.py --is_multitask -ckpt $CKPT_PATH/$EXP_NAME/$LANG/$MODEL/best_model.ckpt -lang $LANG -ptm $PTM --prefix