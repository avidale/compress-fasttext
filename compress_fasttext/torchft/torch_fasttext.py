"""
This file defines FastTextTokenizer, FastTextModel, and a few of their extensions.
"""

import typing as tp
import torch

from .hashing import ft_ngram_hashes_
from .tokenization import re_tokenize


class FastTextTokenizer:
    """ A tokenizer (transformers-like) that prepares input for FastText. """
    def __init__(self, min_n=3, max_n=6, bucket=100_000, vocab=None, word_counts=None):
        self.min_n = min_n
        self.max_n = max_n
        self.bucket = bucket
        self.vocab = vocab or []
        self.word2id = {w: i for i, w in enumerate(self.vocab)}
        self.word_counts = word_counts or []

    @property
    def ngram_size(self) -> int:
        return self.bucket

    @property
    def vocab_size(self) -> int:
        return len(self.word2id)

    @property
    def word_padding_id(self) -> int:
        return len(self.word2id)

    @property
    def ngram_padding_id(self) -> int:
        return self.bucket

    def tokenize(self, text: str) -> tp.List[str]:
        """ The basic tokenization method. """
        return re_tokenize(text)

    def __call__(self, text, return_tensors=None, padding=False):
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        words = [self.tokenize(text) for text in texts]
        word_ids = [[self.word2id.get(w, self.word_padding_id) for w in row] for row in words]
        ngram_ids = [[ft_ngram_hashes_(w, self.min_n, self.max_n, self.bucket) for w in row] for row in words]

        # todo: support truncation for words with too many ngrams

        if return_tensors == 'pt':
            padding = True

        if padding:
            max_seq_len = max(len(row) for row in words)
            for row in word_ids:
                row.extend([self.word_padding_id] * (max_seq_len - len(row)))
            for row in ngram_ids:
                row.extend([[]] * (max_seq_len - len(row)))
            max_ngrams = max(len(word) for row in ngram_ids for word in row)
            for row in ngram_ids:
                for w in row:
                    pass
                    w.extend([self.ngram_padding_id] * (max_ngrams - len(w)))
        if return_tensors:
            word_ids, ngram_ids = torch.tensor(word_ids), torch.tensor(ngram_ids)

        return {'word_ids': word_ids, 'ngram_ids': ngram_ids}


class FastTextModel(torch.nn.Module):
    """ A base transformers-like model, representing a FastText encoder (i.e. bag-of-ngram embeddings) """
    def __init__(self, ngram_size, vocab_size, dim, sparse=False):
        super(FastTextModel, self).__init__()
        self.ngram_size = ngram_size
        self.vocab_size = vocab_size
        self.dim = dim
        self.sparse = sparse
        self.embedding = torch.nn.EmbeddingBag(
            self.vocab_size + self.ngram_size + 1, self.dim, mode='mean', padding_idx=-1, sparse=sparse
        )

    def forward(self, word_ids, ngram_ids, **kwargs):
        """Given word_ids ang ngram_ids, compute word embeddings """
        input_ids = []
        offsets = []
        seq_len = max(len(row) for row in word_ids)
        batch_size = len(word_ids)
        for word_row, ngram_row in zip(word_ids, ngram_ids):
            for word_id, grams_ids in zip(word_row, ngram_row):
                offsets.append(len(input_ids))
                input_ids.extend(grams_ids)
                input_ids.append(word_id + self.ngram_size)
            for _ in range(len(word_row), seq_len):
                offsets.append(len(input_ids))

        device = self.embedding.weight.device
        e = self.embedding(torch.tensor(input_ids, device=device), torch.tensor(offsets, device=device))
        return e.reshape(batch_size, seq_len, self.dim)


def get_sentence_loss(out1, out2, margin=0.3, mult=1.0):
    """ Calculate translation ranking loss using CLS tokens """
    device = out1.device
    emb1 = torch.nn.functional.normalize(out1)
    emb2 = torch.nn.functional.normalize(out2)
    batch_size = emb1.shape[0]
    sims = torch.matmul(emb1, emb2.T)
    if margin:
        sims = sims - torch.eye(batch_size).to(device) * margin
    loss_fn = torch.nn.CrossEntropyLoss()
    labels = torch.arange(batch_size).to(device)
    loss = (
        loss_fn(torch.log_softmax(sims, -1) * mult, labels)
        + loss_fn(torch.log_softmax(sims.T, -1) * mult, labels)
    )
    return loss


class ConvMLMHead(torch.nn.Module):
    """ Computes CBOW-like loss with masked 1D convolution and a dense layer atop it. """

    def __init__(self, embedding_dim, hidden_dim=1024, kernel_size=5):
        super(ConvMLMHead, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.conv = torch.nn.Conv1d(self.embedding_dim, self.hidden_dim, self.kernel_size, padding='same')
        self.proj = torch.nn.Linear(self.hidden_dim, self.embedding_dim)

    def masked_convolution(self, x):
        # never use the middle element: it is always masked
        self.conv.weight.data[:, :, self.kernel_size // 2] = 0
        x_conv = torch.swapaxes(self.conv(torch.swapaxes(x, 1, 2)), 1, 2)
        x_proj = self.proj(torch.nn.functional.softplus(x_conv))
        return x_proj

    def forward(self, x, filter_zeros=True):
        x_proj = self.masked_convolution(x)

        if filter_zeros:
            mask = torch.norm(x, dim=-1) > 0
            nonempty_id = torch.nonzero(mask, as_tuple=True)
            x = x[nonempty_id]
            x_proj = x_proj[nonempty_id]

        loss = get_sentence_loss(x, x_proj, margin=0.3)
        return loss
