"""
This file re-implements the hashing functions implemented in Cython in Gensim.
"""

_MB_MASK = 0xC0
_MB_START = 0x80


def compute_ngrams_bytes_(word, min_n, max_n):
    bytez = f'<{word}>'.encode("utf-8")
    num_bytes = len(bytez)

    ngrams = []
    for i in range(num_bytes):
        if bytez[i] & _MB_MASK == _MB_START:
            continue

        j, n = i, 1
        while j < num_bytes and n <= max_n:
            j += 1
            while j < num_bytes and (bytez[j] & _MB_MASK) == _MB_START:
                j += 1
            if n >= min_n and not (n == 1 and (i == 0 or j == num_bytes)):
                ngram = bytes(bytez[i:j])
                ngrams.append(ngram)
            n += 1
    return ngrams


def ft_hash_bytes_(bytez):
    h = 2166136261
    maxint = 4294967296
    for b in bytez:
        if b > 127:
            b = ((b + 128) % 256 - 128) % maxint
        h = h ^ b
        h = h * 16777619 % maxint  # imitate uint32 overflow
    return h


def ft_ngram_hashes_(word, minn, maxn, num_buckets):
    encoded_ngrams = compute_ngrams_bytes_(word, minn, maxn)
    hashes = [ft_hash_bytes_(n) % num_buckets for n in encoded_ngrams]
    return hashes
