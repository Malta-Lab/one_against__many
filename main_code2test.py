from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from utils import set_seed
from datasets import Code2TestDataset
from args import parse_code2test_args
from models import load_seq2seq_model_and_tokenizer, Code2TestModel


if __name__ == '__main__':
    args = parse_code2test_args('train')
    set_seed()

    data_dir = Path(args.data_dir)
    ptm = args.pretrained_model
    output_dir = Path(args.output_dir) / 'code2test' /ptm.replace('/', '-')    

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

    model = Code2TestModel(pretrained_model, tokenizer, train_size=len(train_loader), epochs=args.epochs, scheduler=args.scheduler)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='best_model',
        save_top_k=1,
        save_last=True,
        verbose=True,
        monitor='val_loss')

    early_stop_callback = EarlyStopping('val_loss', patience=2)

    print('Training...')
    trainer = pl.Trainer(gpus=args.gpus,
                         max_epochs=args.epochs,
                         callbacks=[checkpoint_callback,
                                    early_stop_callback],
                         strategy='ddp')
    trainer.fit(model, train_loader, eval_loader)
