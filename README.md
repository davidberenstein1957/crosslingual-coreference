# Crosslingual Coreference
Coreference is amazing but the data required for training a model is very scarce. In our case, the available training for non-English languages also proved to be poorly annotated. Crosslingual Coreference, therefore, uses the assumption a trained model with English data and cross-lingual embeddings should work for languages with similar sentence structures.

[![Current Release Version](https://img.shields.io/github/release/pandora-intelligence/crosslingual-coreference.svg?style=flat-square&logo=github)](https://github.com/pandora-intelligence/crosslingual-coreference/releases)
[![pypi Version](https://img.shields.io/pypi/v/crosslingual-coreference.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/crosslingual-coreference/)
[![PyPi downloads](https://static.pepy.tech/personalized-badge/crosslingual-coreference?period=total&units=international_system&left_color=grey&right_color=orange&left_text=pip%20downloads)](https://pypi.org/project/crosslingual-coreference/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)

# Install

```
pip install crosslingual-coreference
```
# Quickstart
```python
from crosslingual_coreference import Predictor

text = (
    "Do not forget about Momofuku Ando! He created instant noodles in Osaka. At"
    " that location, Nissin was founded. Many students survived by eating these"
    " noodles, but they don't even know him."
)

# choose minilm for speed/memory and info_xlm for accuracy
predictor = Predictor(
    language="en_core_web_sm", device=-1, model_name="minilm"
)

print(predictor.predict(text)["resolved_text"])
# Note you can also get 'cluster_heads' and 'clusters'
# Output
#
# Do not forget about Momofuku Ando!
# Momofuku Ando created instant noodles in Osaka.
# At Osaka, Nissin was founded.
# Many students survived by eating instant noodles,
# but Many students don't even know Momofuku Ando.
```
![](https://raw.githubusercontent.com/Pandora-Intelligence/crosslingual-coreference/master/img/example_en.png)

## Models
As of now, there are two models available "spanbert", "info_xlm", "xlm_roberta", "minilm", which scored 83, 77, 74 and 74 on OntoNotes Release 5.0 English data, respectively. 
- The "minilm" model is the best quality speed trade-off for both mult-lingual and english texts. 
- The "info_xlm" model produces the best quality for multi-lingual texts.
- The AllenNLP "spanbert" model produces the best quality for english texts.

## Chunking/batching to resolve memory OOM errors

```python
from crosslingual_coreference import Predictor

predictor = Predictor(
    language="en_core_web_sm",
    device=0,
    model_name="minilm",
    chunk_size=2500,
    chunk_overlap=2,
)
```

## Use spaCy pipeline
```python
import spacy

import crosslingual_coreference

text = (
    "Do not forget about Momofuku Ando! He created instant noodles in Osaka. At"
    " that location, Nissin was founded. Many students survived by eating these"
    " noodles, but they don't even know him."
)


nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(
    "xx_coref", config={"chunk_size": 2500, "chunk_overlap": 2, "device": 0}
)

doc = nlp(text)
print(doc._.coref_clusters)
# Output
#
# [[[4, 5], [7, 7], [27, 27], [36, 36]],
# [[12, 12], [15, 16]],
# [[9, 10], [27, 28]],
# [[22, 23], [31, 31]]]
print(doc._.resolved_text)
# Output
#
# Do not forget about Momofuku Ando!
# Momofuku Ando created instant noodles in Osaka.
# At Osaka, Nissin was founded.
# Many students survived by eating instant noodles,
# but Many students don't even know Momofuku Ando.
print(doc._.cluster_heads)
# Output
# 
# {Momofuku Ando: [5, 6], 
# instant noodles: [11, 12], 
# Osaka: [14, 14], 
# Nissin: [21, 21], 
# Many students: [26, 27]} 
```

## More Examples
![](https://raw.githubusercontent.com/Pandora-Intelligence/crosslingual-coreference/master/img/example_total.png)
