from typing import Union

from spacy import util
from spacy.tokens import Doc

from .CrossLingualPredictor import CrossLingualPredictor as Predictor


class CrossLingualPredictorSpacy(Predictor):
    def __init__(
        self,
        language: str,
        device: int = -1,
        model_name: str = "minilm",
        chunk_size: Union[int, None] = None,
        chunk_overlap: int = 2,
    ) -> None:
        super().__init__(language, device, model_name, chunk_size, chunk_overlap)
        Doc.set_extension("coref_clusters", default=None, force=True)
        Doc.set_extension("resolved_text", default=None, force=True)
        Doc.set_extension("cluster_heads", default=None, force=True)

    def __call__(self, doc: Doc) -> Doc:
        """
        predict the class for a spacy Doc

        Args:
            doc (Doc): a spacy doc

        Returns:
            Doc: spacy doc with ._.cats key-class proba-value dict
        """
        prediction = super(self.__class__, self).predict(doc.text)
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
            texts = [doc.text for doc in docs]

            pred_results = super(self.__class__, self).pipe(texts)

            for doc, pred_result in zip(docs, pred_results):
                doc = self.assign_prediction_to_doc(doc, pred_result)

                yield doc

    @staticmethod
    def assign_prediction_to_doc(doc: Doc, prediction: dict):
        """
        It takes a spaCy Doc object and a prediction dictionary and adds the prediction to the Doc object

        :param doc: Doc
        :type doc: Doc
        :param prediction: The prediction returned by the model
        :type prediction: dict
        :return: The doc object with the coref_clusters and resolved_text attributes added.
        """
        doc._.coref_clusters = prediction["clusters"]
        doc._.resolved_text = prediction["resolved_text"]
        doc._.cluster_heads = prediction["cluster_heads"]
        return doc
