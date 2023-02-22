"""
This file implements converters between FastTextKeyedVectors and torchft.
"""

import torch
import typing as tp

from compress_fasttext.torchft.torch_fasttext import FastTextTokenizer, FastTextModel
from gensim.models.fasttext import FastTextKeyedVectors


def torch_to_gensim(model: FastTextModel, tokenizer: FastTextTokenizer) -> FastTextKeyedVectors:
    ft = FastTextKeyedVectors(
        vector_size=model.dim,
        min_n=tokenizer.min_n,
        max_n=tokenizer.max_n,
        bucket=tokenizer.bucket,
    )
    ft.index_to_key = tokenizer.vocab
    ft.key_to_index = {w: i for i, w in enumerate(ft.index_to_key)}
    embeds = model.embedding.weight.data.cpu().numpy()
    ft.vectors_ngrams = embeds[:model.ngram_size]
    ft.vectors_vocab = embeds[model.ngram_size:model.ngram_size + model.vocab_size]
    ft.recalc_char_ngram_buckets()
    ft.adjust_vectors()
    return ft


def gensim_to_torch(keyed_vectors: FastTextKeyedVectors) -> tp.Tuple[FastTextModel, FastTextTokenizer]:
    tokenizer = FastTextTokenizer(
        vocab=keyed_vectors.index_to_key,
        bucket=keyed_vectors.bucket,
        min_n=keyed_vectors.min_n,
        max_n=keyed_vectors.max_n,
    )
    model = FastTextModel(
        ngram_size=tokenizer.ngram_size,
        vocab_size=tokenizer.vocab_size,
        dim=keyed_vectors.vector_size,
    )
    model.embedding.weight.data[:model.ngram_size] = torch.tensor(keyed_vectors.vectors_ngrams)
    model.embedding.weight.data[model.ngram_size:model.ngram_size + model.vocab_size] = torch.tensor(keyed_vectors.vectors_vocab)
    return model, tokenizer
