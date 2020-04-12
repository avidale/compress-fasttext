import numpy as np

from gensim.models.keyedvectors import FastTextKeyedVectors
from gensim.models.utils_any2vec import ft_ngram_hashes


from compress_fasttext.compress import make_new_fasttext_model


EPSILON = 1e-24


class CompressedFastTextKeyedVectors(FastTextKeyedVectors):
    """ This class extends FastTextKeyedVectors by fixing several issues:
        - index2word of a freshly created model is initialized from its vocab
        - the model does not keep heavy and useless vectors_ngrams_norm
        - word_vec() method with use_norm applies normalization in the right place
    """
    def __init__(self, *args, **kwargs):
        super(CompressedFastTextKeyedVectors, self).__init__(*args, **kwargs)
        self.update_index2word()

    @classmethod
    def load(cls, *args, **kwargs):
        loaded = super(CompressedFastTextKeyedVectors, cls).load(*args, **kwargs)
        return make_new_fasttext_model(
            loaded,
            new_vectors=loaded.vectors,
            new_vectors_ngrams=loaded.vectors_ngrams,
            new_vocab=loaded.vocab,
            cls=cls,
        )

    def update_index2word(self):
        if not self.index2word:
            inverse_index = {value.index: key for key, value in self.vocab.items()}
            self.index2word = [inverse_index.get(i) for i in range(len(self.vocab))]

    def word_vec(self, word, use_norm=False):
        """Get `word` representations in vector space, as a 1D numpy array.

        Parameters
        ----------
        word : str
            Input word
        use_norm : bool, optional
            If True - resulting vector will be L2-normalized (unit euclidean length).

        Returns
        -------
        numpy.ndarray
            Vector representation of `word`.

        Raises
        ------
        KeyError
            If word and all ngrams not in vocabulary.

        """
        if word in self.vocab:
            return super(FastTextKeyedVectors, self).word_vec(word, use_norm)
        elif self.bucket == 0:
            raise KeyError('cannot calculate vector for OOV word without ngrams')
        else:
            word_vec = np.zeros(self.vectors_ngrams.shape[1], dtype=np.float32)
            ngram_hashes = ft_ngram_hashes(word, self.min_n, self.max_n, self.bucket, self.compatible_hash)
            if len(ngram_hashes) == 0:
                return word_vec
            for nh in ngram_hashes:
                word_vec += self.vectors_ngrams[nh]
            result = word_vec / len(ngram_hashes)
            if use_norm:
                result /= np.sqrt(max(sum(result ** 2), EPSILON))
            return result

    def init_sims(self, replace=False):
        """Precompute L2-normalized vectors.

        Parameters
        ----------
        replace : bool, optional
            If True - forget the original vectors and only keep the normalized ones = saves lots of memory!
        """
        super(FastTextKeyedVectors, self).init_sims(replace)
        # todo: make self.vectors_norm a view over self.vectors, to avoid decompression
        # do NOT calculate vectors_ngrams_norm; using them is a mistake
        self.vectors_ngrams_norm = None
