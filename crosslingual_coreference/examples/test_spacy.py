import crosslingual_coreference
import spacy

from .data import texts

nlp = spacy.load('nl_core_news_sm')

nlp.remove_pipe('xx_coref')
nlp.add_pipe('xx_coref', config={"model_name": "xlm_roberta"})

for doc in nlp.pipe(texts):
    print(doc._.coref_clusters)
    print(doc._.resolved_text)
