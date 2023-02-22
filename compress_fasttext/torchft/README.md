# compress_fasttext.torchft

This folder contains work in progress on PyTorch implementation of FastText.

The intended use of this submodule is to be able to fine-tune FastText
embeddings for an arbitrary downstream task, using them as a normal Torch module.

It is not intended for:
* __Training FastText from scratch__. In the future, we may implement it.
* __Inference__. This module is not optimized for efficient inference in any way.
It is expected that after fine-tuning, you convert the model back into Gensim format.
* __Working with sparse, quantized, or decomposed FastText__. 
The only method of compression that is compatible with `torchft` is `prune_ft` 
which fully preserves the original model format.


An expected scenario of use is the following:

(1) Load Gensim FastText model and convert it to Torch
```
import gensim
from compress_fasttext.torchft.converters import gensim_to_torch, torch_to_gensim
orig_ft = gensim.models.fasttext.FastTextKeyedVectors.load('orig_gensim_file.bin')
model, tokenizer = gensim_to_torch(orig_ft)
```
(2) Perform some fine-tuning on your data and task
```
for batch_of_texts, labels in batch_iterator:
    batch_embeds = model(**tokenizer(batch_of_texts, return_tensors='pt'))  
    # the shape is batch_size * sequence_length * embed_dim
    compute_loss(batch_embeds, labels).backward()
    optimizer.step()
    optimizer.zero_grad()
```

(3) Convert your fine-tuned model back to Gensim FastTextKeyedVectors and save them.
```
new_ft = torch_to_gensim(model, tokenizer)
new_ft.save('new_gensim_file.bin')
```

Now you can forget about fine-tuning and just use the new embeddings.