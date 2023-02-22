import os
import gensim
import numpy as np
import torch

from compress_fasttext.torchft.torch_fasttext import FastTextModel, FastTextTokenizer, get_sentence_loss
from compress_fasttext.torchft.converters import gensim_to_torch, torch_to_gensim

BIG_MODEL_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/test_data/ft_leipzig_ru_mini.bin')


def test_conversion():
    word1 = 'синий'
    orig_ft = gensim.models.fasttext.FastTextKeyedVectors.load(BIG_MODEL_FILE)
    print(orig_ft.vectors_vocab.shape, orig_ft.vectors_ngrams.shape)
    vec0 = orig_ft[word1]

    model, tokenizer = gensim_to_torch(orig_ft)
    with torch.inference_mode():
        vec1 = model(**tokenizer(word1, return_tensors='pt'))[0][0].numpy()
    assert np.allclose(vec0, vec1)

    new_ft = torch_to_gensim(model, tokenizer)
    vec2 = new_ft[word1]
    assert np.allclose(vec0, vec2)


def test_torch_training():
    tokenizer = FastTextTokenizer(bucket=32, vocab=[])
    model = FastTextModel(ngram_size=tokenizer.bucket, vocab_size=len(tokenizer.vocab), dim=5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    data1, data2 = ['cat', 'dog'], ['chat', 'chien']
    losses = []
    for _ in range(30):
        vecs1 = model(**tokenizer(data1, return_tensors='pt'))
        assert list(vecs1.shape) == [2, 1, 5]
        vecs2 = model(**tokenizer(data2, return_tensors='pt'))
        loss = get_sentence_loss(vecs1[:, 0], vecs2[:, 0])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    # test that the loss is really decreasing
    print(losses)
    assert losses[-1] < losses[0]

    # test that after training, cat is similar to chat and dog is similar to chien
    with torch.no_grad():
        sims = torch.matmul(
            torch.nn.functional.normalize(vecs1[:, 0]),
            torch.nn.functional.normalize(vecs2[:, 0]).T
        ).numpy()
    print(sims)
    assert sims[0, 0] > 0.5
    assert sims[1, 1] > 0.5
    assert sims[0, 1] < 0.5
    assert sims[1, 0] < 0.5
