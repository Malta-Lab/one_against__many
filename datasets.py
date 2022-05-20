from pathlib import Path
from torch.utils.data import Dataset
from utils import squeeze_dict
import json
import numpy as np
import random


def load_methods_2_test_dataset(path):
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


def tokenize(tokenizer, source):
    return tokenizer(source, max_length=128, padding='max_length', truncation=True, return_tensors='pt')


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__[key]


def read_translate_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            src = line1.strip()
            trg = line2.strip()
            examples.append(
                Example(
                    idx=idx,
                    source=src,
                    target=trg,
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_refine_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0

    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip(),
                    target=line2.strip(),
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_concode_examples(filename, data_num):
    """Read examples from filename."""
    examples = []

    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["nl"].strip(),
                    target=x["code"].strip()
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_summarize_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_defect_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)

            code = ' '.join(js['func'].split())
            examples.append(
                Example(
                    idx=js['idx'],
                    source=code,
                    target=js['target']
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_clone_examples(filename, data_num):
    """Read examples from filename."""
    index_filename = filename
    url_to_code = {}
    with open('/'.join(index_filename.split('/')[:-1]) + '/data.jsonl') as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())
            url_to_code[js['idx']] = code

    data = []
    with open(index_filename) as f:
        idx = 0
        for line in f:
            line = line.strip()
            url1, url2, label = line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label == '0':
                label = 0
            else:
                label = 1
            # data.append(CloneExample(
            #     url_to_code[url1], url_to_code[url2], label, url1, url2))
            data.append(Example(
                idx=idx,
                source=url_to_code[url1] + url_to_code[url2],
                target=label
            ))
            idx += 1
            if idx == data_num:
                break
    return data


class BaseDataset(Dataset):
    def __init__(self, data_path, split='train', tokenizer=None, prefix=False):
        self.data_path = Path(data_path)
        self. split = split
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.examples = None

    def __len__(self):
        return len(self.examples)


class Code2TestDataset(BaseDataset):
    def __init__(self, data_path=Path('datasets/methods2test/corpus/raw/fm/'), split='train', tokenizer=None, prefix=False):
        super().__init__(data_path=data_path, split=split, tokenizer=tokenizer, prefix=prefix)
        self.full_path = self.data_path / split
        self.data, self.labels = load_methods_2_test_dataset(self.full_path)
        self.tokenizer = tokenizer
        if prefix:
            self.data = [f'code to test: {d}' for d in self.data]
        self.examples = [{'source': self.data[i], 'target':self.labels[i]}
                         for i in range(len(self.data))]

    def __getitem__(self, idx):
        example = self.examples[idx]
        source = example['source']
        target = example['target']

        if self.tokenizer:
            source = tokenize(self.tokenizer, source)
            target = tokenize(self.tokenizer, target)
            source = squeeze_dict(source)
            target = squeeze_dict(target)
        return source, target


class CodeSearchNetDataset(BaseDataset):
    def __init__(self, data_path=Path('./datasets/CodeSearchNet'), split='train', tokenizer=None, prefix=False, language='javascript'):
        super().__init__(data_path=data_path, split=split, tokenizer=tokenizer, prefix=prefix)
        self.language = language
        self.examples = read_jsonl(data_path / self.language / f'{split}.jsonl')
        if self.prefix:
            for i in self.examples:
                i['code'] = f'code search: {i["code"]}'

    def __getitem__(self, idx):
        instance = self.examples[idx]
        code = instance["code"]
        comment = instance["docstring"]
        url = instance["url"]
        
        if self.tokenizer:
            code = tokenize(self.tokenizer, code)
            comment = tokenize(self.tokenizer, comment)
            code = squeeze_dict(code)
            comment = squeeze_dict(comment)

        return code, comment, url


class TranslateDataset(Dataset):
    def __init__(self, data_path=Path('./datasets/multi_task/translate'), split='train', tokenizer=None, prefix=False, mode='java to cs'):
        self.data_path = Path(data_path)
        self.prefix = prefix
        self.split = split
        self.tokenizer = tokenizer
        self.mode = mode
        self.java_path = self.data_path / f'{self.split}.java-cs.txt.java'
        self.csharp_path = self.data_path / f'{self.split}.java-cs.txt.cs'
        if self.mode == 'java to cs':
            self.examples = read_translate_examples(
                f'{str(self.java_path)},{str(self.csharp_path)}', data_num=None)
        else:
            self.examples = read_translate_examples(
                f'{str(self.csharp_path)},{str(self.java_path)}', data_num=None)
        if prefix:
            for i in self.examples:
                i.source = f'translate {self.mode}: {i.source}'

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        source = example['source']
        target = example['target']

        if self.tokenizer:
            source = tokenize(self.tokenizer, source)
            target = tokenize(self.tokenizer, target)
            source = squeeze_dict(source)
            target = squeeze_dict(target)
        return source, target


class RefineDataset(BaseDataset):
    def __init__(self, data_path=Path('./datasets/multi_task/refine'), split='train', tokenizer=None, prefix=False, mode='small'):
        super().__init__(data_path=data_path, split=split, tokenizer=tokenizer, prefix=prefix)
        self.mode = mode
        self.buggy_path = self.data_path/self.mode / \
            f'{self.split}.buggy-fixed.buggy'
        self.fixed_path = self.data_path/self.mode / \
            f'{self.split}.buggy-fixed.fixed'
        self.examples = read_refine_examples(
            f'{self.buggy_path},{self.fixed_path}', data_num=None)
        if self.prefix:
            for i in self.examples:
                i.source = f'refine {self.mode}: {i.source}'

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        source = example['source']
        target = example['target']

        if self.tokenizer:
            source = tokenize(self.tokenizer, source)
            target = tokenize(self.tokenizer, target)
            source = squeeze_dict(source)
            target = squeeze_dict(target)
        return source, target


class ConcodeDataset(BaseDataset):
    def __init__(self, data_path=Path('./datasets/multi_task/concode'), split='train', tokenizer=None, prefix=False):
        super().__init__(data_path=data_path, split=split, tokenizer=tokenizer, prefix=prefix)
        self.examples = read_concode_examples(
            f'{self.data_path}/{self.split}.json', data_num=None)
        if self.prefix:
            for i in self.examples:
                i.source = f'concode: {i.source}'

    def __getitem__(self, idx):
        example = self.examples[idx]
        source = example['source']
        target = example['target']

        if self.tokenizer:
            source = tokenize(self.tokenizer, source)
            target = tokenize(self.tokenizer, target)
            source = squeeze_dict(source)
            target = squeeze_dict(target)
        return source, target


class DefectDataset(BaseDataset):
    def __init__(self, data_path=Path('./datasets/multi_task/defect'), split='train', tokenizer=None, prefix=False):
        super().__init__(data_path=data_path, split=split, tokenizer=tokenizer, prefix=prefix)
        self.examples = read_defect_examples(
            f'{self.data_path}/{self.split}.jsonl', data_num=None)
        if self.prefix:
            for i in self.examples:
                i.source = f'defect: {i.source}'

    def __getitem__(self, idx):
        example = self.examples[idx]
        source = example['source']
        target = example['target']

        if self.tokenizer:
            source = tokenize(self.tokenizer, source)
            target = tokenize(self.tokenizer, target)
            source = squeeze_dict(source)
            target = squeeze_dict(target)
        return source, target


class CloneDataset(BaseDataset):
    # Source ids is the concatenation of code1 and code2
    def __init__(self, data_path=Path('./datasets/multi_task/clone'), split='train', tokenizer=None, prefix=False):
        super().__init__(data_path=data_path, split=split, tokenizer=tokenizer, prefix=prefix)
        self.examples = read_clone_examples(
            f'{self.data_path}/{self.split}.txt', data_num=None)
        if self.prefix:
            for i in self.examples:
                i.source = f'clone: {i.source}'

    def __getitem__(self, idx):
        example = self.examples[idx]
        source = example['source']
        target = example['target']

        if self.tokenizer:
            source = tokenize(self.tokenizer, source)
            target = tokenize(self.tokenizer, target)
            source = squeeze_dict(source)
            target = squeeze_dict(target)
        return source, target


class MultiTaskDataset(Dataset):
    # TODO: add a field in the return indicating which task is being performed
    # TODO: receive the list of datasets as a parameter list
    def __init__(self, datasets, tokenizer=None, iterations=10000, same_probs=False):
        super().__init__()
        self.bsz = 8
        self.iterations = iterations
        self.same_probs = same_probs
        self.tokenizer = tokenizer
        self.datasets = [d.examples for d in datasets]
        self.probabilities = self.__probabilities_codet5()
        self.examples = self.__compose_dataset()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        source = example['source']
        target = example['target']

        if self.tokenizer:
            source = tokenize(self.tokenizer, source)
            target = tokenize(self.tokenizer, target)
            source = squeeze_dict(source)
            target = squeeze_dict(target)
        return source, target

    def __compose_dataset(self):
        final_dataset = []
        for _ in range(self.iterations):
            if self.same_probs:
                choice = np.random.choice(self.datasets)
            else:
                choice = np.random.choice(self.datasets, p=self.probabilities)

            for _ in range(self.bsz):
                final_dataset.append(choice.pop(
                    random.randint(0, len(choice)-1)))

        return final_dataset

    def __probabilities_codet5(self):
        """Same probs calculation done in CodeT5 source code"""
        probs = [len(x) for x in self.datasets]
        probs = [x / sum(probs) for x in probs]
        probs = [x ** 0.7 for x in probs]
        probs = [x / sum(probs) for x in probs]
        return probs
