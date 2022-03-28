from crosslingual_coreference import Predictor

from .data import texts

predictor = Predictor(language='nl_core_news_sm')

print(predictor.pipe(texts))

predictor = Predictor(language='nl_core_news_sm', model_name='xlm_roberta')

print(predictor.pipe(texts))
