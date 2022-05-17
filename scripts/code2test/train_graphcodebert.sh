CUDA_VISIBLE_DEVICES=2,3 python main_code2test.py -ptm microsoft/graphcodebert-base --batch_size 8 --gpus 2
CUDA_VISIBLE_DEVICES=3 python eval_code2test.py -ptm 'microsoft/graphcodebert-base' \
-ckpt '/home/parraga/projects/_masters/code2test/checkpoints/multitask_code/code2test/microsoft-graphcodebert-base/best_model.ckpt'