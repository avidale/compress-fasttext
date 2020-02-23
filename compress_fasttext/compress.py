import gensim
import numpy as np

from .quantization import quantize
from .decomposition import DecomposedMatrix
from .prune import prune_ngrams, prune_vocab, count_buckets, RowSparseMatrix


def make_new_fasttext_model(ft, new_vectors, new_vectors_ngrams, new_vocab=None):
    new_ft = gensim.models.keyedvectors.FastTextKeyedVectors(
        vector_size=ft.vector_size,
        min_n=ft.min_n,
        max_n=ft.max_n,
        bucket=new_vectors_ngrams.shape[0],
        compatible_hash=ft.compatible_hash
    )
    new_ft.vectors_vocab = None  # if we don't fine tune the model we don't need these vectors
    new_ft.vectors = new_vectors  # quantized vectors top_vectors
    if new_vocab is None:
        new_ft.vocab = ft.vocab
    else:
        new_ft.vocab = new_vocab
    new_ft.vectors_ngrams = new_vectors_ngrams
    return new_ft


def quantize_ft(ft, qdim=100, centroids=255, sample=None):
    print('quantizing vectors...')
    new_vectors = quantize(ft.vectors, qdim=qdim, centroids=centroids, verbose=True, sample=sample)
    print('quantizing ngrams...')
    new_vectors_ngrams = quantize(ft.vectors_ngrams, qdim=qdim, centroids=centroids, verbose=True, sample=sample)

    return make_new_fasttext_model(ft, new_vectors=new_vectors, new_vectors_ngrams=new_vectors_ngrams)


def svd_ft(ft, n_components=30, fp16=True):
    print('compressing vectors...')
    new_vectors = DecomposedMatrix.compress(ft.vectors, n_components=n_components, fp16=fp16)
    print('compressing ngrams...')
    new_vectors_ngrams = DecomposedMatrix.compress(ft.vectors_ngrams, n_components=n_components, fp16=fp16)

    return make_new_fasttext_model(ft, new_vectors=new_vectors, new_vectors_ngrams=new_vectors_ngrams)


def prune_ft(ft, new_vocab_size=1_000, new_ngrams_size=20_000, fp16=True):
    print('compressing vectors...')
    top_vocab, top_vectors = prune_vocab(ft, new_vocab_size=new_vocab_size)
    print('compressing ngrams...')
    new_ngrams = prune_ngrams(ft, new_ngrams_size)
    if fp16:
        top_vectors = top_vectors.astype(np.float16)
        new_ngrams = new_ngrams.astype(np.float16)
    return make_new_fasttext_model(
        ft,
        new_vectors=top_vectors,
        new_vectors_ngrams=new_ngrams,
        new_vocab=top_vocab,
    )


def prune_ft_freq(ft, new_vocab_size=20_000, new_ngrams_size=100_000, fp16=True, pq=True, qdim=100, centroids=255):
    new_to_old_buckets, old_hash_count = count_buckets(ft, list(ft.vocab.keys()), new_ngrams_size=new_ngrams_size)
    id_and_count = sorted(old_hash_count.items(), key=lambda x: x[1], reverse=True)
    ids = [x[0] for x in id_and_count[:new_ngrams_size]]
    top_ngram_vecs = ft.vectors_ngrams[ids]
    if pq and len(top_ngram_vecs) > 0:
        top_ngram_vecs = quantize(top_ngram_vecs, qdim=qdim, centroids=centroids)
    elif fp16:
        top_ngram_vecs = top_ngram_vecs.astype(np.float16)
    rsm = RowSparseMatrix.from_small(ids, top_ngram_vecs, nrows=ft.vectors_ngrams.shape[0])

    top_voc, top_vec = prune_vocab(ft, new_vocab_size=new_vocab_size)
    if pq and len(top_vec) > 0:
        top_vec = quantize(top_vec, qdim=qdim, centroids=centroids)
    elif fp16:
        top_vec = top_vec.astype(np.float16)

    return make_new_fasttext_model(ft, top_vec, rsm, new_vocab=top_voc)
