ckpt_path='/home/parraga/projects/_masters/code_search/checkpoints'
model='microsoft-codebert-base'
ptm='microsoft/codebert-base'

CUDA_VISIBLE_DEVICES=0,1 python main_codesearch.py -lang javascript --gpus 2 --scheduler linear
lang='javascript'
CUDA_VISIBLE_DEVICES=1 python evaluation_codesearch.py -ckpt $ckpt_path/codesearch/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

CUDA_VISIBLE_DEVICES=0,1 python main_codesearch.py -lang go --gpus 2
lang='go'
CUDA_VISIBLE_DEVICES=1 python evaluation_codesearch.py -ckpt $ckpt_path/codesearch/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

CUDA_VISIBLE_DEVICES=0,1 python main_codesearch.py -lang java --gpus 2
lang='java'
CUDA_VISIBLE_DEVICES=1 python evaluation_codesearch.py -ckpt $ckpt_path/codesearch/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

CUDA_VISIBLE_DEVICES=0,1 python main_codesearch.py -lang php --gpus 2
lang='php'
CUDA_VISIBLE_DEVICES=1 python evaluation_codesearch.py -ckpt $ckpt_path/codesearch/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

CUDA_VISIBLE_DEVICES=0,1 python main_codesearch.py -lang ruby --gpus 2
lang='ruby'
CUDA_VISIBLE_DEVICES=1 python evaluation_codesearch.py -ckpt $ckpt_path/codesearch/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm

CUDA_VISIBLE_DEVICES=0,1 python main_codesearch.py -lang python --gpus 2
lang='python'
CUDA_VISIBLE_DEVICES=1 python evaluation_codesearch.py -ckpt $ckpt_path/codesearch/$lang/$model/best_model.ckpt -lang $lang -ptm $ptm
