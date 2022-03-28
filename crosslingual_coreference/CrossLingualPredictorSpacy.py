from spacy import util
from spacy.tokens import Doc

from .CrossLingualPredictor import CrossLingualPredictor as Predictor


class CrossLingualPredictorSpacy(Predictor):
    def __init__(self, language: str, device: int = -1, model_name: str = 'info_xlm') -> None:
        super().__init__(language, device, model_name)
        Doc.set_extension("coref_clusters", default=None, force=True)
        Doc.set_extension("resolved_text", default=None, force=True)

    def __call__(self, doc: Doc) -> Doc:
        """
        predict the class for a spacy Doc

        Args:
            doc (Doc): a spacy doc

        Returns:
            Doc: spacy doc with ._.cats key-class proba-value dict
        """
        prediction = super(self.__class__, self).predict(doc.text.replace("\n", " "))
        doc = self.assign_prediction_to_doc(doc, prediction)
        return doc

    def pipe(self, stream, batch_size=128):
        """
        predict the class for a spacy Doc stream

        Args:
            stream (Doc): a spacy doc

        Returns:
            Doc: spacy doc with ._.cats key-class proba-value dict
        """
        for docs in util.minibatch(stream, size=batch_size):
            texts = [doc.text.replace("\n", " ") for doc in docs]
            
            pred_results = super(self.__class__, self).pipe(texts)
            
            for doc, pred_result in zip(docs, pred_results):
                doc = self.assign_prediction_to_doc(doc, pred_result)
                
                yield doc

    @staticmethod
    def assign_prediction_to_doc(doc: Doc, prediction: dict):
        doc._.coref_clusters = prediction['clusters']
        doc._.resolved_text = prediction['resolved_text']
        return doc
        
        