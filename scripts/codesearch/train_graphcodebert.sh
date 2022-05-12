ckpt_path='/home/parraga/projects/_masters/code_search/checkpoints'
model='microsoft-graphcodebert-base'
ptm='microsoft/graphcodebert-base'

CUDA_VISIBLE_DEVICES=0,1 python main_codesearch.py -lang javascript -ptm $ptm --gpus 1 --scheduler linear
lang='javascript'
CUDA_VISIBLE_DEVICES=0 python evaluation_codesearch.py -ckpt $ckpt_path/codesearch/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

CUDA_VISIBLE_DEVICES=0,1 python main_codesearch.py -lang go -ptm $ptm --gpus 1
lang='go'
CUDA_VISIBLE_DEVICES=0 python evaluation_codesearch.py -ckpt $ckpt_path/codesearch/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

CUDA_VISIBLE_DEVICES=0,1 python main_codesearch.py -lang java -ptm $ptm --gpus 1
lang='java'
CUDA_VISIBLE_DEVICES=0 python evaluation_codesearch.py -ckpt $ckpt_path/codesearch/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

CUDA_VISIBLE_DEVICES=0,1 python main_codesearch.py -lang php -ptm $ptm --gpus 1
lang='php'
CUDA_VISIBLE_DEVICES=0 python evaluation_codesearch.py -ckpt $ckpt_path/codesearch/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

CUDA_VISIBLE_DEVICES=0,1 python main_codesearch.py -lang ruby -ptm $ptm --gpus 1
lang='ruby'
CUDA_VISIBLE_DEVICES=0 python evaluation_codesearch.py -ckpt $ckpt_path/codesearch/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

CUDA_VISIBLE_DEVICES=0,1 python main_codesearch.py -lang python -ptm $ptm --gpus 1
lang='python'
CUDA_VISIBLE_DEVICES=0 python evaluation_codesearch.py -ckpt $ckpt_path/codesearch/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm
