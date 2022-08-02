from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from utils import set_seed
from datasets import MultiTaskDataset, Code2TestDataset, CloneDataset, CodeSearchNetDataset, ConcodeDataset, DefectDataset, RefineDataset, TranslateDataset
from args import parse_multi_task_args
from models import load_seq2seq_model_and_tokenizer, MultiTaskModel
from pytorch_lightning.loggers import TensorBoardLogger
import json

def build_dataset_dict(args, tokenizer, split='train'):
    datasets = {}
    for task in args.tasks:
        if task == 'code2test':
            print('Adding Code2test')
            datasets['code2test'] = Code2TestDataset(split=split, tokenizer=tokenizer, prefix=args.prefix)
        elif task == 'codesearch':
            for lang in args.cs_lang:
                print('Adding CodeSearch for {}'.format(lang))
                datasets['codesearch ' + lang] = CodeSearchNetDataset(split=split, tokenizer=tokenizer, prefix=args.prefix, 
                                                                    language=lang, lang_prefix=args.lang_prefix)
        elif task == 'summarization':
            for lang in args.sum_lang:
                print('Adding Summarization for {}'.format(lang))
                datasets['summarization ' + lang] = CodeSearchNetDataset(split=split, tokenizer=tokenizer, prefix=args.prefix, 
                                                                    language=lang, lang_prefix=args.lang_prefix, task='summarization')
        elif task == 'clone':
            print('Adding Clone')
            datasets['clone'] = CloneDataset(split=split, tokenizer=tokenizer, prefix=args.prefix)
        elif task == 'generation':
            print('Adding Concode')
            datasets['generation'] = ConcodeDataset(split=split, tokenizer=tokenizer, prefix=args.prefix)
        elif task == 'defect':
            print('Adding Defect')
            datasets['defect'] = DefectDataset(split=split, tokenizer=tokenizer, prefix=args.prefix)
        elif task == 'refine':
            print('Adding Refine')
            datasets['refine'] = RefineDataset(split=split, tokenizer=tokenizer, prefix=args.prefix, mode=args.refine_mode)
        elif task == 'translate':
            print('Adding Translate')
            datasets['translate'] = TranslateDataset(split=split, tokenizer=tokenizer, prefix=args.prefix, mode=args.translate_order)

    return datasets

if __name__ == '__main__':
    args = parse_multi_task_args('train')
    set_seed()

    ptm = args.pretrained_model
    output_dir = Path('checkpoints/multitask') / args.output_dir /ptm.replace('/', '-')  
    output_dir.mkdir(parents=True, exist_ok=True)  
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f)

    print('Loading Model and Tokenizer...')
    pretrained_model, tokenizer = load_seq2seq_model_and_tokenizer(
        ptm)

    print('Loading Dataset...')
    train_datasets = build_dataset_dict(args, tokenizer)
    train_mt_dataset = MultiTaskDataset(train_datasets, bsz=args.batch_size, tokenizer=tokenizer, iterations=args.iterations, same_probs=args.same_probs)

    val_datasets = build_dataset_dict(args, tokenizer, split='valid')
    val_mt_dataset = MultiTaskDataset(val_datasets, bsz=args.batch_size, tokenizer=tokenizer, iterations=args.val_iterations, same_probs=args.same_probs)

    print('Loading DataLoader...')
    train_loader = DataLoader(train_mt_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    val_loader = DataLoader(val_mt_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    

    model = MultiTaskModel(pretrained_model, tokenizer, train_size=len(train_loader), epochs=args.epochs, scheduler=args.scheduler)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='best_model',
        save_top_k=1,
        save_last=True,
        verbose=True,
        monitor='val_loss')

    early_stop_callback = EarlyStopping('val_loss', patience=2)

    logger = TensorBoardLogger(save_dir='lightning_logs/', name=str(output_dir))

    print('Training...')
    trainer = pl.Trainer(gpus=args.gpus,
                         max_epochs=args.epochs,
                         accelerator='gpu',
                         strategy='ddp',
                         callbacks=[checkpoint_callback,
                                    early_stop_callback],
                         logger=logger)
    trainer.fit(model, train_loader, val_loader)
