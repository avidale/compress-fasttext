import numpy as np

try:
    from sklearn.decomposition import TruncatedSVD
except ModuleNotFoundError:
    # SVD is not the core feature of this library, so we keep this dependency optional
    pass


class DecomposedMatrix:
    def __init__(self, compressed, components):
        self.compressed = compressed
        self.components = components

    def __getitem__(self, item):
        return np.dot(self.compressed[item], self.components)

    @property
    def shape(self):
        return self.compressed.shape[0], self.components.shape[1]

    @classmethod
    def compress(cls, data, n_components=30, fp16=True):
        model = TruncatedSVD(n_components=n_components)
        compressed = model.fit_transform(data)
        if fp16:
            compressed = compressed.astype(np.float16)
        return cls(compressed, model.components_)
