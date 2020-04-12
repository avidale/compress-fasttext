# Compress-fastText
This Python 3 package allows to compress fastText word embedding models 
(from the `gensim` package) by orders of magnitude, 
without seriously affecting their quality. It can be installed with `pip`:
```commandline
pip install compress-fasttext[full]
```
If you are not going to perform matrix decomposition or quantization,
 you can install a variety with less dependencies: 
```commandline
pip install compress-fasttext
```


This [blogpost](https://habr.com/ru/post/489474) (in Russian) gives more details about the motivation and 
methods for compressing fastText models.

### Model compression
You can use this package to compress your own fastText model (or one downloaded e.g. from 
[RusVectores](https://rusvectores.org/ru/models/)):

```python
import gensim
import compress_fasttext
big_model = gensim.models.fasttext.FastTextKeyedVectors.load('path-to-original-model')
small_model = compress_fasttext.prune_ft_freq(big_model, pq=True)
small_model.save('path-to-new-model')
```

Different compression methods include:
- matrix decomposition (`svd_ft`)
- product quantization (`quantize_ft`)
- optimization of feature hashing (`prune_ft`)
- feature selection (`prune_ft_freq`)

The recommended approach is combination of feature selection and quantization (`prune_ft_freq` with `pq=True`).

### Model usage
If you just need a tiny fastText model for Russian, you can download 
[this](https://github.com/avidale/compress-fasttext/releases/download/v0.0.1/ft_freqprune_100K_20K_pq_100.bin)
28-megabyte model. It's a compressed version of 
[ruscorpora_none_fasttextskipgram_300_2_2019](http://vectors.nlpl.eu/repository/20/181.zip) model
from [RusVectores](https://rusvectores.org/ru/models/).

If `compress-fasttext` is already installed, you can download and use this tiny model
```python
import compress_fasttext
small_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(
    'https://github.com/avidale/compress-fasttext/releases/download/v0.0.1/ft_freqprune_100K_20K_pq_100.bin'
)
print(small_model['спасибо'])
```
The class `CompressedFastTextKeyedVectors` inherits from `gensim.models.keyed_vectors.FastTextKeyedVectors`, 
but makes a few additional optimizations.

### Notes
This code is heavily based on the [navec](https://github.com/natasha/navec) package by Alexander Kukushkin and 
[the blogpost](https://medium.com/@vasnetsov93/shrinking-fasttext-embeddings-so-that-it-fits-google-colab-cd59ab75959e) 
by Andrey Vasnetsov about shrinking fastText embeddings.
