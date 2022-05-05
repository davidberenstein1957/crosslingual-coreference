import spacy

import crosslingual_coreference  # noqa: F401

from .data import texts

nlp = spacy.load("nl_core_news_sm")

nlp.add_pipe("xx_coref", config={"model_name": "minilm"})

for doc in nlp.pipe(texts):
    print(doc._.coref_clusters)
    print(doc._.resolved_text)
