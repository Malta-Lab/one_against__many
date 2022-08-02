# test a codesearch model
ckpt_path='/home/parraga/projects/_masters/multitask_code/checkpoints/sanity/codesearch'
model='microsoft-codebert-base'
ptm='microsoft/codebert-base'

# CUDA_VISIBLE_DEVICES=0,1 python main_codesearch.py -lang javascript --gpus 2 --scheduler linear --output_dir ./checkpoints/sanity
# lang='javascript'
# CUDA_VISIBLE_DEVICES=1 python eval_codesearch.py -ckpt $ckpt_path/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

# CUDA_VISIBLE_DEVICES=0,1 python main_codesearch.py -lang ruby --gpus 2 --output_dir ./checkpoints/sanity
# lang='ruby'
# CUDA_VISIBLE_DEVICES=1 python eval_codesearch.py -ckpt $ckpt_path/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

# test a code2test model
# CUDA_VISIBLE_DEVICES=0,1,2 python main_code2test.py -ptm $ptm --batch_size 8 --gpus 3 --output_dir ./checkpoints/sanity
# CUDA_VISIBLE_DEVICES=1 python eval_code2test.py -ptm $ptm\
# -ckpt '/home/parraga/projects/_masters/multitask_code/checkpoints/sanity/code2test/microsoft-codebert-base/best_model.ckpt'