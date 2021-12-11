# Compress-fastText
This Python 3 package allows to compress fastText word embedding models 
(from the `gensim` package) by orders of magnitude, 
without seriously affecting their quality. 

**Note: gensim==4.0.0 has introduced some backward-incompatible changes:**
* With gensim<4.0.0, please use compress-fasttext<=0.0.7 
(and optionally Russian models from [our first release](https://github.com/avidale/compress-fasttext/releases/tag/v0.0.1)).
* With gensim>=4.0.0, please use compress-fasttext>=0.1.0
(and optionally Russian or English models from [our 0.1.0 release](https://github.com/avidale/compress-fasttext/releases/tag/v0.1.0)).
* Some models are supported by both versions of gensim+compress-fasttext; this should be determined experimentally.


The package can be installed with `pip`:
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

Compress a model in Gensim format:
```python
import gensim
import compress_fasttext
big_model = gensim.models.fasttext.FastTextKeyedVectors.load('path-to-original-model')
small_model = compress_fasttext.prune_ft_freq(big_model, pq=True)
small_model.save('path-to-new-model')
```

Import a model in Facebook original format and compress it:
```python
from gensim.models.fasttext import load_facebook_model
import compress_fasttext
big_model = load_facebook_model('path-to-original-model').wv
small_model = compress_fasttext.prune_ft_freq(big_model, pq=True)
small_model.save('path-to-new-model')
```
To perform this compression, you will need to `pip install gensim==3.8.3 pqkmeans` beforehand. 

Different compression methods include:
- matrix decomposition (`svd_ft`)
- product quantization (`quantize_ft`)
- optimization of feature hashing (`prune_ft`)
- feature selection (`prune_ft_freq`)

The recommended approach is combination of feature selection and quantization (`prune_ft_freq` with `pq=True`).

### Model usage
If you just need a tiny fastText model for Russian, you can download 
[this](https://github.com/avidale/compress-fasttext/releases/download/gensim-4-draft/geowac_tokens_sg_300_5_2020-100K-20K-100.bin)
21-megabyte model. It's a compressed version of 
[geowac_tokens_none_fasttextskipgram_300_5_2020](http://vectors.nlpl.eu/repository/20/214.zip) model
from [RusVectores](https://rusvectores.org/ru/models/).

If `compress-fasttext` is already installed, you can download and use this tiny model
```python
import compress_fasttext
small_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(
    'https://github.com/avidale/compress-fasttext/releases/download/gensim-4-draft/geowac_tokens_sg_300_5_2020-100K-20K-100.bin'
)
print(small_model['спасибо'])
# [ 1.22736421e-01 -3.21375653e-01 ...  4.33826083e-01] # a 300-dimensional vector
print(small_model.most_similar('котенок'))
# [('щенок', 0.659467875957489), ('кот', 0.5826340913772583), ... ]
```
The class `CompressedFastTextKeyedVectors` inherits from `gensim.models.fasttext.FastTextKeyedVectors`, 
but makes a few additional optimizations.

For English, you can use [this](https://github.com/avidale/compress-fasttext/releases/download/v0.0.4/cc.en.300.compressed.bin) tiny model, 
obtained by compressing [the model by Facebook](https://fasttext.cc/docs/en/crawl-vectors.html).

```python
import compress_fasttext
small_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(
    'https://github.com/avidale/compress-fasttext/releases/download/v0.0.4/cc.en.300.compressed.bin'
)
print(small_model['hello'])
# [ 1.84736611e-01  6.32683930e-03  4.43901886e-03 ... -2.88431027e-02]  # a 300-dimensional vector
print(small_model.most_similar('Python'))
# [('PHP', 0.5252903699874878), ('.NET', 0.5027452707290649), ('Java', 0.4897131323814392),  ... ]
```

More compressed models for 101 various languages can be found at https://zenodo.org/record/4905385. 

### Notes
This code is heavily based on the [navec](https://github.com/natasha/navec) package by Alexander Kukushkin and 
[the blogpost](https://medium.com/@vasnetsov93/shrinking-fasttext-embeddings-so-that-it-fits-google-colab-cd59ab75959e) 
by Andrey Vasnetsov about shrinking fastText embeddings.
