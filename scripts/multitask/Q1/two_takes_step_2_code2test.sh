# here the idea is to fine tune the models from the checkpoint in step 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
EXP_NAME='two_takes_step_2'
#=============================CODE T5=================================
PTM="Salesforce/codet5-base"
PTM_NAME="Salesforce-codet5-base"
BASE_CKPT="/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/two_takes_step_1/${PTM_NAME}/best_model.ckpt"
FINETUNED_CKPT="/home/parraga/projects/_masters/multitask_code/checkpoints/code2test/two_takes_step_2/${PTM_NAME}/best_model.ckpt"

CUDA_VISIBLE_DEVICES=0,2 python main_code2test.py -ptm $PTM --batch_size 8 --gpus 2 --prefix \
-ckpt $BASE_CKPT --output_dir $EXP_NAME

CUDA_VISIBLE_DEVICES=0 python eval_code2test.py -ptm $PTM \
-ckpt $FINETUNED_CKPT --is_multitask --prefix

#===========================CODEBERT===============================
PTM="microsoft/codebert-base"
PTM_NAME="microsoft-codebert-base"
BASE_CKPT="/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/two_takes_step_1/${PTM_NAME}/best_model.ckpt"
FINETUNED_CKPT="/home/parraga/projects/_masters/multitask_code/checkpoints/code2test/two_takes_step_2/${PTM_NAME}/best_model.ckpt"

CUDA_VISIBLE_DEVICES=0,2 python main_code2test.py -ptm $PTM --batch_size 8 --gpus 2 --prefix \
-ckpt $BASE_CKPT --output_dir $EXP_NAME

CUDA_VISIBLE_DEVICES=0 python eval_code2test.py -ptm $PTM \
-ckpt $FINETUNED_CKPT --is_multitask --prefix

#===========================GRAPHCODEBERT===============================
PTM="microsoft/graphcodebert-base"
PTM_NAME="microsoft-graphcodebert-base"
BASE_CKPT="/home/parraga/projects/_masters/multitask_code/checkpoints/multitask/two_takes_step_1/${PTM_NAME}/best_model.ckpt"
FINETUNED_CKPT="/home/parraga/projects/_masters/multitask_code/checkpoints/code2test/two_takes_step_2/${PTM_NAME}/best_model.ckpt"

CUDA_VISIBLE_DEVICES=0,2 python main_code2test.py -ptm $PTM --batch_size 8 --gpus 2 --prefix \
-ckpt $BASE_CKPT --output_dir $EXP_NAME

CUDA_VISIBLE_DEVICES=0 python eval_code2test.py -ptm $PTM \
-ckpt $FINETUNED_CKPT --is_multitask --prefix