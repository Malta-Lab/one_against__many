import pytorch_lightning as pl
from pathlib import Path
import torch.optim as optim
import torch.nn as nn
import torch
from transformers import (EncoderDecoderModel, T5ForConditionalGeneration, T5Model, RobertaTokenizer, 
                        AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup)


def load_tokenizer(model_name, cache_path='./pretrained_stuff'):
    cache_path = Path(cache_path)
    cache_path.mkdir(exist_ok=True, parents=True)
    if 'codet5' in model_name:
        tokenizer = RobertaTokenizer.from_pretrained(
            model_name, cache_dir=cache_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_path)
    return tokenizer


def load_model(model_name, cache_path='./pretrained_stuff'):
    cache_path = Path(cache_path)
    cache_path.mkdir(exist_ok=True, parents=True)
    if 'codet5' in model_name:
        model = T5Model.from_pretrained(
            model_name, cache_dir=cache_path)
    else:
        model = AutoModel.from_pretrained(
            model_name, cache_dir=cache_path)
    return model


def load_seq2seq_model_and_tokenizer(model_name, cache_path='./pretrained_stuff'):
    cache_path = Path(cache_path)
    cache_path.mkdir(exist_ok=True, parents=True)
    if 't5' in model_name.lower():
        return T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_path), \
            RobertaTokenizer.from_pretrained(
                model_name, cache_dir=cache_path)
    else:
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            model_name, model_name, cache_dir=cache_path)
        tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base', cache_dir=cache_path)
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer


class BertEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.bert_encoder = model

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        embedding = self.bert_encoder(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
        )

        embedding = embedding[1]
        return embedding


class T5Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.t5_encoder = model

    def forward(self, input_ids, attention_mask):
        embedding = self.t5_encoder.encoder(
            input_ids=input_ids, attention_mask=attention_mask,
        )

        embedding = embedding.last_hidden_state[:, 0, :]
        return embedding


class CodeSearchModel(pl.LightningModule):
    def __init__(self, model_name, cache_path='./pretrained_stuff', train_size=None, epochs=None, scheduler='step'):
        super(CodeSearchModel, self).__init__()
        model = load_model(model_name, cache_path)
        self.encoder = T5Encoder(
            model) if 't5' in model_name else BertEncoder(model)
        self.criterion = nn.CrossEntropyLoss()
        self.train_size = train_size
        self.epochs = epochs
        self.scheduler = scheduler

    def forward(self, code, comment):
        code = self.encoder(**code)
        comment = self.encoder(**comment)
        return code, comment

    def training_step(self, batch, batch_idx):
        code, comment, _ = batch
        encoded_code, encoded_comment = self(code, comment)
        scores = torch.einsum("ab,cb->ac", encoded_code, encoded_comment)
        loss = self.criterion(scores, torch.arange(
            encoded_code.size(0), device=scores.device))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        code, comment, _ = batch
        encoded_code, encoded_comment = self(code, comment)
        scores = torch.einsum("ab,cb->ac", encoded_code, encoded_comment)
        loss = self.criterion(scores, torch.arange(
            encoded_code.size(0), device=scores.device))
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=2e-5)
        if self.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=0.1)
        elif self.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=self.train_size*self.epochs)
        elif self.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=self.train_size*self.epochs)
        elif self.scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=3)
        return [optimizer], [scheduler]


class Code2TestModel(pl.LightningModule):
    def __init__(self, pretrained_model, tokenizer, train_size=None, epochs=None, scheduler='step'):
        super(Code2TestModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.tokenizer = tokenizer
        self.train_size = train_size
        self.epochs = epochs
        self.scheduler = scheduler

    def training_step(self, batch, batch_idx):
        source, target = batch
        labels = target['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        outputs = self.pretrained_model(input_ids=source['input_ids'],
                                        attention_mask=source['attention_mask'],
                                        labels=labels)
        loss = outputs[0]
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        source, target = batch
        labels = target['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        outputs = self.pretrained_model(input_ids=source['input_ids'],
                                        attention_mask=source['attention_mask'],
                                        labels=labels)
        loss = outputs[0]
        self.log('val_loss', loss)
        return loss

    def generate(self, **inputs):
        return self.pretrained_model.generate(**inputs)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=2e-5)
        if self.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=0.1)
        elif self.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=self.train_size*self.epochs)
        elif self.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=self.train_size*self.epochs)
        elif self.scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=3)
        return [optimizer], [scheduler]
