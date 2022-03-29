# Crosslingual Coreference
Coreference is amazing but the data required for training a model is very scarce. In our case, the available training for non-English languages also proved to be poorly annotated. Crosslingual Coreference, therefore, uses the assumption a trained model with English data and cross-lingual embeddings should work for languages with similar sentence structures.

# Install

```
pip install crosslingual-coreference
```
# Quickstart
```python
from crosslingual_coreference import Predictor

text = "Do not forget about Momofuku Ando! He created instant noodles in Osaka. At that location, Nissin was founded. Many students survived by eating these noodles, but they don't even know him."

predictor = Predictor(language="en_core_web_sm", device=-1, model_name="info_xlm")

print(predictor.predict(text)["resolved_text"])
# Output
# 
# Do not forget about Momofuku Ando! 
# Momofuku Ando created instant noodles in Osaka. 
# At Osaka, Nissin was founded. 
# Many students survived by eating instant noodles, 
# but Many students don't even know Momofuku Ando.
```
![](https://raw.githubusercontent.com/Pandora-Intelligence/crosslingual-coreference/master/img/example_en.png)
## Use spaCy pipeline
```python
import crosslingual_coreference
import spacy

text = "Do not forget about Momofuku Ando! He created instant noodles in Osaka. At that location, Nissin was founded. Many students survived by eating these noodles, but they don't even know him."

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('xx_coref')

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
```
## Available models
As of now, there are two models available "info_xlm", "xlm_roberta", which scored 77 and 74 on OntoNotes Release 5.0 English data, respectively.
## More Examples
![](https://raw.githubusercontent.com/Pandora-Intelligence/crosslingual-coreference/master/img/example_total.png)

