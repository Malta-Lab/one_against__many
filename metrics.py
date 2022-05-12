from nltk.translate.bleu_score import corpus_bleu
import re
import numpy as np


def bleu(hypothesis, reference):
    hypothesis = [sentence.split() for sentence in hypothesis]
    reference = [[sentence.split()] for sentence in reference]
    return corpus_bleu(reference, hypothesis)

def tokenized_bleu(hypothesis, reference, tokenizer):
    hypothesis = [tokenizer.convert_ids_to_tokens(tokenizer(sentence)['input_ids']) for sentence in hypothesis]
    reference = [[tokenizer.convert_ids_to_tokens(tokenizer(sentence)['input_ids'])] for sentence in reference]
    return corpus_bleu(reference, hypothesis)

def alpha_bleu(hypothesis, reference):
    hypothesis = [re.split('([^a-zA-Z0-9])', sentence.strip()) for sentence in hypothesis]
    for sentence in hypothesis:
        sentence.remove("") if "" in sentence else None
        sentence.remove(" ") if " " in sentence else None
    reference = [[re.split('([^a-zA-Z0-9])', sentence.strip())] for sentence in reference]
    for sentence in reference:
        sentence[0].remove("") if "" in sentence[0] else None
        sentence[0].remove(" ") if " " in sentence[0] else None
    return corpus_bleu(reference, hypothesis)

def mrr(query_embed, base_embed, query_index, base_index):
    """This functions comes from graphcode bert"""

    scores = np.matmul(query_embed,base_embed.T)

    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]

    ranks=[]
    for url, sort_id in zip(query_index,sort_ids):
        rank=0
        find=False
        for idx in sort_id[:1000]:
            if find is False:
                rank+=1
            if base_index[idx]==url:
                find=True
        if find:
            ranks.append(1/rank)
        else:
            ranks.append(0)

    result = {
        "mrr": float(np.mean(ranks))
    }
    return result