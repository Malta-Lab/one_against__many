from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from models import CodeSearchModel, load_tokenizer
from datasets import CodeSearchNetDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from args import parse_codesearch_args
from utils import set_seed
from pathlib import Path
import json

if __name__ == '__main__':
    args = parse_codesearch_args('train')

    set_seed()

    # main configs
    data_dir = Path(args.data_dir)
    language = args.language
    ptm = args.pretrained_model
    output_dir = Path(args.output_dir) / 'codesearch' / language / ptm.replace('/', '-')
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f)


    # creating tokenizer
    tokenizer = load_tokenizer(ptm, './pretrained_stuff')

    # creating dataset
    train_dataset = CodeSearchNetDataset(data_dir,'train', tokenizer, args.prefix, language)
    val_dataset = CodeSearchNetDataset(data_dir, 'valid', tokenizer, args.prefix, language)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # creating model
    model = CodeSearchModel(ptm, './pretrained_stuff', train_size=len(trainloader), epochs=args.epochs, scheduler=args.scheduler)

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='best_model',
        save_top_k=1,
        save_last=True,
        verbose=True,
        monitor='val_loss')

    early_stop_callback = EarlyStopping('val_loss', patience=2)

    logger = TensorBoardLogger(save_dir='lightning_logs/codesearch', name=args.output_dir)

    # creating trainer
    trainer = pl.Trainer(gpus=args.gpus,
                         max_epochs=args.epochs,
                         gradient_clip_val=1,
                         strategy='ddp',
                         callbacks=[checkpoint_callback,
                                    early_stop_callback,
                                    ],
                        logger=logger)
    trainer.fit(model, trainloader, valloader)
