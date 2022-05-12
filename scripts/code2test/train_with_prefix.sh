CUDA_VISIBLE_DEVICES=1,2 python main_code2test.py -ptm Salesforce/codet5-base --batch_size 8 --gpus 2 --prefix --output_dir ./checkpoints/code2test/prefix 
CUDA_VISIBLE_DEVICES=2 python eval_code2test.py -ptm 'Salesforce/codet5-base' \
-ckpt '/home/parraga/projects/_masters/code2test/checkpoints/code2test/prefix/Salesforce-codet5-base/best_model.ckpt'

CUDA_VISIBLE_DEVICES=1,2 python main_code2test.py -ptm microsoft/codebert-base --batch_size 8 --gpus 2 --prefix --output_dir ./checkpoints/code2test/prefix 
CUDA_VISIBLE_DEVICES=2 python eval_code2test.py -ptm 'microsoft/codebert-base' \
-ckpt '/home/parraga/projects/_masters/code2test/checkpoints/code2test/prefix/microsoft-codebert-base/best_model.ckpt'

CUDA_VISIBLE_DEVICES=2,3 python main_code2test.py -ptm microsoft/graphcodebert-base --batch_size 8 --gpus 2 --prefix --output_dir ./checkpoints/code2test/prefix 
CUDA_VISIBLE_DEVICES=3 python eval_code2test.py -ptm 'microsoft/graphcodebert-base' \
-ckpt '/home/parraga/projects/_masters/code2test/checkpoints/code2test/prefix/microsoft-graphcodebert-base/best_model.ckpt'