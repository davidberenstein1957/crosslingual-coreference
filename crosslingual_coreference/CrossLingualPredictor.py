import pathlib
from typing import List

import requests
import tqdm  # progress bar
from allennlp.predictors.predictor import Predictor

from .CorefResolver import CorefResolver as Resolver

MODELS = {
    'xlm_roberta': {
        'url': 'https://storage.googleapis.com/pandora-intelligence/models/crosslingual-coreference/xlm-roberta-base/model.tar.gz',
        'f1_score_ontonotes': 74,
        'file_extension': '.tar.gz'
    },
    'info_xlm': {
        'url': 'https://storage.googleapis.com/pandora-intelligence/models/crosslingual-coreference/infoxlm-base/model.tar.gz',
        'f1_score_ontonotes': 77,
        'file_extension': '.tar.gz'
    }
}


class CrossLingualPredictor(object):
    def __init__(self, language: str, device: int = -1, model_name: str = 'info_xlm') -> None:
        self.language = language
        self.filename = None
        self.device = device
        self.model_url = MODELS[model_name]['url']
        self.resolver = Resolver()
        self.download_model()
        self.set_coref_model()
    
    def download_model(self):
        """
        Download file with progressbar if file is present, otherwise pass
        """
        self.filename = self.model_url.replace('https://storage.googleapis.com/pandora-intelligence/', '')
        path = pathlib.Path(self.filename)
        if path.is_file():
            pass
        else:
            path.parent.absolute().mkdir(parents=True, exist_ok=True)
            r = requests.get(self.model_url, stream=True)
            file_size = int(r.headers['Content-Length'])
            
            chunk_size = 1024
            num_bars = int(file_size / chunk_size)

            with open(self.filename, 'wb') as fp:
                for chunk in tqdm.tqdm(
                                        r.iter_content(chunk_size=chunk_size),
                                        total = num_bars,
                                        unit = 'KB',
                                        desc = self.filename,
                                        leave = True 
                                    ):
                    fp.write(chunk)
    
    def set_coref_model(self):
        """Initialize AllenNLP coreference model """
        self.predictor = Predictor.from_path(self.filename, language=self.language, cuda_device=self.device)

    def predict(self, text: str, advanced_resolve: bool = True) -> dict:
        """predict and rsolve

        Args:
            text (str): an input text
            advanced_resolve (bool, optional): use more advanced resoled from 
            https://towardsdatascience.com/how-to-make-an-effective-coreference-resolution-model-55875d2b5f19. Defaults to True.

        Returns:
            dict: a prediciton
        """
        prediction = self.predictor.predict_json({"document": text})
        
        clusters = prediction.get("clusters")

        # Passing a document with no coreferences returns its original form
        if not clusters:
            return text
        
        doc = self.predictor._spacy(text)
        
        if advanced_resolve:
            prediction['resolved_text'] = self.resolver.replace_corefs(doc, clusters)
        else:
            prediction['resolved_text'] = self.predictor.replace_corefs(doc, clusters)
            
        return prediction
        
    def pipe(self, texts: List[str], advanced_resolve: bool = True) -> List[dict]:
        return [self.predict(text, advanced_resolve) for text in texts]
