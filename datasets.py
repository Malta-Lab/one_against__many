from pathlib import Path
from torch.utils.data import Dataset
from utils import squeeze_dict
import json


def load_dataset(path):
    """
    Loads the dataset from the given path.
    """
    with open(path / 'input.methods.txt', 'r', encoding='latin') as f:
        data = f.readlines()
    with open(path / 'output.tests.txt', 'r', encoding='latin') as f:
        labels = f.readlines()
    return data, labels

def read_jsonl(filepath):
    with open(filepath, 'r') as json_file:
        json_list = list(json_file)

    lines = []
    for json_str in json_list:
        result = json.loads(json_str)
        lines.append(result)
    return lines


class Code2TestDataset(Dataset):
    def __init__(self, path, split='train', tokenizer=None, add_prefix=False):
        self.path = Path(path)
        self.full_path = self.path / split
        self.data, self.labels = load_dataset(self.full_path)
        self.tokenizer = tokenizer
        if add_prefix: 
            self.data = [f'Code to test: {d}' for d in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source = self.data[idx]
        target = self.labels[idx]

        if self.tokenizer:
            source = self.tokenizer(source, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
            target = self.tokenizer(target, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
            source = squeeze_dict(source)
            target = squeeze_dict(target)

        return source, target


class CodeSearchNetDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.data = read_jsonl(data_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]
        code = instance["code"]
        comment = instance["docstring"]
        url = instance["url"]

        code = self.tokenizer(code, max_length=128, truncation=True, return_tensors='pt', padding='max_length')
        comment = self.tokenizer(comment, max_length=128, truncation=True, return_tensors='pt', padding='max_length')
        code = squeeze_dict(code)
        comment = squeeze_dict(comment)

        return code, comment, url