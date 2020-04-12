import logging
import numpy as np

try:
    import pqkmeans
except ImportError:
    pqkmeans = None

from .navec_like import PQ


def quantize(matrix, qdim, centroids, sample=None, iterations=5, verbose=False):
    if not pqkmeans:
        raise ImportError('You need to install the `pqkmeans` package to perform quantization')
    encoder = pqkmeans.encoder.PQEncoder(
        iteration=iterations,
        num_subdim=qdim,
        Ks=centroids
    )

    matrix = np.array(matrix)
    vectors, dim = matrix.shape
    if sample is None:
        selection = matrix
    else:
        indexes = np.random.randint(vectors, size=sample)
        selection = matrix[indexes]

    encoder.fit(selection)
    indexes = encoder.transform(matrix).astype(PQ.index_type(centroids))
    codes = encoder.codewords

    if verbose:
        mb = 1024**2
        old_size = matrix.nbytes / mb
        new_size1 = codes.nbytes / mb
        new_size2 = indexes.nbytes / mb
        logging.info('Matrix size went from {:.1f} mb to {:.1f} + {:.1f} mb ({:.1%})'.format(
            old_size, new_size1, new_size2, (new_size1 + new_size2) / old_size
        ))

    return PQ(vectors, dim, qdim, centroids, indexes, codes)
