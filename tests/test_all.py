import os
import gensim
import pytest

import compress_fasttext

BIG_MODEL_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/test_data/ft_leipzig_ru_mini.bin')


def cosine_sim(x, y):
    return sum(x * y) / (sum(x**2) * sum(y**2)) ** 0.5


@pytest.mark.parametrize('method, params', [
    (compress_fasttext.quantize_ft, dict(qdim=32)),
    (compress_fasttext.prune_ft_freq, dict(pq=False, new_ngrams_size=10_000, new_vocab_size=10_000)),
    (compress_fasttext.prune_ft_freq, dict(pq=True, new_ngrams_size=10_000, new_vocab_size=10_000, qdim=16)),
    (compress_fasttext.prune_ft, dict(new_ngrams_size=10_000, new_vocab_size=10_000)),
    (compress_fasttext.svd_ft, dict(n_components=32)),
])
def test_prune_save_load(method, params):
    word1 = 'синий'
    word2 = 'белый'
    big_ft = gensim.models.fasttext.FastTextKeyedVectors.load(BIG_MODEL_FILE)
    vec0 = big_ft[word1]

    small_model = method(big_ft, **params)
    assert cosine_sim(vec0, small_model[word1]) > 0.75
    out1 = small_model.most_similar(word1)
    assert word2 in {w for w, sim in out1}

    small_model.save('tmp_small.bin')
    small_model2 = compress_fasttext.models.CompressedFastTextKeyedVectors.load('tmp_small.bin')
    assert cosine_sim(vec0, small_model2[word1]) > 0.75
    out2 = small_model2.most_similar(word1)
    assert word2 in {w for w, sim in out2}
    assert out1[0][1] == pytest.approx(out2[0][1])
