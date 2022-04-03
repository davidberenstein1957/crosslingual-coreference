from crosslingual_coreference import Predictor

from .data import texts

predictor = Predictor(language="nl_core_news_sm", chunk_size=2500, chunk_overlap=2)

print(predictor.pipe(texts))
