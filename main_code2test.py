from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from utils import set_seed
from datasets import Code2TestDataset
from args import parse_code2test_args
from models import load_seq2seq_model_and_tokenizer, Code2TestModel, MultiTaskModel
from pytorch_lightning.loggers import TensorBoardLogger
import json


if __name__ == '__main__':
    args = parse_code2test_args('train')
    set_seed()

    data_dir = Path(args.data_dir)
    ptm = args.pretrained_model
    output_dir = Path('checkpoints/code2test') / args.output_dir /ptm.replace('/', '-')  
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f)

    print('Loading Model and Tokenizer...')
    pretrained_model, tokenizer = load_seq2seq_model_and_tokenizer(
        ptm)

    print('Loading Dataset...')
    train_data = Code2TestDataset(data_dir, 'train', tokenizer, args.prefix)
    eval_data = Code2TestDataset(data_dir, 'eval', tokenizer, args.prefix)

    print('Loading DataLoader...')
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(
        eval_data, batch_size=args.batch_size, shuffle=False)

    if args.checkpoint_path is not None:
        model = MultiTaskModel.load_from_checkpoint(checkpoint_path=args.checkpoint_path,
                                                            pretrained_model=pretrained_model,
                                                            tokenizer=tokenizer, train_size=len(train_loader), 
                                                            epochs=args.epochs, scheduler=args.scheduler, exclusive_task='code2test')
    else:
        model = Code2TestModel(pretrained_model, tokenizer, train_size=len(
            train_loader), epochs=args.epochs, scheduler=args.scheduler)

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
                         callbacks=[checkpoint_callback,
                                    early_stop_callback],
                         strategy='ddp',
                         logger=logger)
    trainer.fit(model, train_loader, eval_loader)
