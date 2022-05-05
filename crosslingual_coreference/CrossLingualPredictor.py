import itertools
import pathlib
from typing import List, Union

import requests
import tqdm  # progress bar
from allennlp.predictors.predictor import Predictor
from spacy.tokens import Doc

from .CorefResolver import CorefResolver as Resolver

MODELS = {
    "xlm_roberta": {
        "url": "https://storage.googleapis.com/pandora-intelligence/models/crosslingual-coreference/xlm-roberta-base/model.tar.gz",  # noqa: B950
        "f1_score_ontonotes": 74,
        "file_extension": ".tar.gz",
    },
    "info_xlm": {
        "url": "https://storage.googleapis.com/pandora-intelligence/models/crosslingual-coreference/infoxlm-base/model.tar.gz",  # noqa: B950
        "f1_score_ontonotes": 77,
        "file_extension": ".tar.gz",
    },
    "minilm": {
        "url": (
            "https://storage.googleapis.com/pandora-intelligence/models/crosslingual-coreference/minilm/model.tar.gz"
        ),
        "f1_score_ontonotes": 74,
        "file_extension": ".tar.gz",
    },
}


class CrossLingualPredictor(object):
    def __init__(
        self,
        language: str,
        device: int = -1,
        model_name: str = "minilm",
        chunk_size: Union[int, None] = None,  # determines the # sentences per batch
        chunk_overlap: int = 2,  # determines the # of overlapping sentences per chunk
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.language = language
        self.filename = None
        self.device = device
        self.model_url = MODELS[model_name]["url"]
        self.resolver = Resolver()
        self.download_model()
        self.set_coref_model()

    def download_model(self):
        """
        It downloads the model from the url provided and saves it in the current directory
        """
        self.filename = self.model_url.replace("https://storage.googleapis.com/pandora-intelligence/", "")
        path = pathlib.Path(self.filename)
        if path.is_file():
            pass
        else:
            path.parent.absolute().mkdir(parents=True, exist_ok=True)
            r = requests.get(self.model_url, stream=True)
            file_size = int(r.headers["Content-Length"])

            chunk_size = 1024
            num_bars = int(file_size / chunk_size)

            with open(self.filename, "wb") as fp:
                for chunk in tqdm.tqdm(
                    r.iter_content(chunk_size=chunk_size),
                    total=num_bars,
                    unit="KB",
                    desc=self.filename,
                    leave=True,
                ):
                    fp.write(chunk)

    def set_coref_model(self):
        """Initialize AllenNLP coreference model"""
        self.predictor = Predictor.from_path(self.filename, language=self.language, cuda_device=self.device)

    def predict(self, text: str, advanced_resolve: bool = True) -> dict:
        """predict and rsolve

        Args:
            text (str): an input text
            advanced_resolve (bool, optional): use more advanced resoled from
            https://towardsdatascience.com/how-to-make-an-effective-coreference-resolution-model-55875d2b5f19.
            Defaults to True.

        Returns:
            dict: a prediciton
        """
        # chunk text
        doc = self.predictor._spacy(text)
        if self.chunk_size:
            chunks = self.chunk_sentencized_doc(doc)
        else:
            chunks = [text]

        # make predictions for individual chunks
        predictions = [self.predictor.predict_json({"document": chunk}) for chunk in chunks]

        # determine doc_lengths to resolve overlapping chunks
        doc_lengths = [
            sum([len(sent) for sent in list(doc_chunk.sents)[:-2]]) for doc_chunk in self.predictor._spacy.pipe(chunks)
        ]
        doc_lengths = [0] + doc_lengths[:-1]

        # convert cluster predictions to their original index in doc
        all_clusters = [pred["clusters"] for pred in predictions]
        corrected_clusters = []
        for idx, doc_clus in enumerate(all_clusters):
            corrected_clusters.append(
                [[[num + sum(doc_lengths[: idx + 1]) for num in span] for span in clus] for clus in doc_clus]
            )

        merged_clusters = self.merge_clusters(corrected_clusters)

        prediction = {
            "clusters": merged_clusters,
            "resolved_text": self.resolver.replace_corefs(doc, merged_clusters),
        }

        return prediction

    def pipe(self, texts: List[str], advanced_resolve: bool = True) -> List[dict]:
        """
        > The function takes a list of strings and returns a list of dictionaries

        :param texts: List[str]
        :type texts: List[str]
        :param advanced_resolve: If True, the model will try to resolve the ambiguity of the entities, defaults to True
        :type advanced_resolve: bool (optional)
        :return: A list of dictionaries.
        """
        return [self.predict(text, advanced_resolve) for text in texts]

    def chunk_sentencized_doc(self, doc: Doc) -> List[str]:
        """Split spacy doc object into chunks of maximum size 'chunk_size' with
        overlapping sentences between the chunks equal to 'overlap_size'.

        Args:
            doc (Doc): Spacy doc object
            chunk_size (int): maximum chunk size
            overlap_size (int): the number of sentences to overlap between chunks

        Returns:
            List[str]: List of all chunks, each merged in a string.
        """
        result = []
        temp_len = 0
        temp_chunk = []

        for sentence in doc.sents:
            s = str(sentence)
            if temp_len + len(s) + 1 <= self.chunk_size:
                temp_chunk.append(sentence)
                temp_len += len(s) + 1
            else:
                result.append(temp_chunk[:])
                overlap_sentences = temp_chunk[len(temp_chunk) - self.chunk_overlap :]
                len_overlap_sentences = sum(len(str(x)) + 1 for x in overlap_sentences) - 1
                temp_chunk = []
                temp_chunk.extend(overlap_sentences)
                temp_chunk.append(sentence)
                temp_len = len_overlap_sentences + len(s) + 1

        # Check if the last chunk is too short. If it is, add it to the previous chunk
        if len(temp_chunk) == 1 and len(temp_chunk[0]) < 3:
            result[-1].extend(temp_chunk[:])
        else:
            result.append(temp_chunk[:])

        return [" ".join([str(s) for s in chunk]) for chunk in result]

    @staticmethod
    def merge_clusters(
        clusters: List[List[List[int]]],
    ) -> List[List[List[int]]]:
        """merge overlapping cluster from different segments, based on n_overlap_sentences"""
        main_doc_clus = []
        for doc_clus in clusters:
            for clus in doc_clus:
                combine_clus = False
                for span in clus:
                    for main_clus in main_doc_clus:
                        for main_span in main_clus:
                            if main_span == span:
                                combined_clus = main_clus + clus
                                combined_clus.sort()
                                combined_clus = list(k for k, _ in itertools.groupby(combined_clus))
                                combine_clus = True
                                break
                        if combine_clus:
                            break
                    if combine_clus:
                        break
                if combine_clus:
                    main_doc_clus.append(combined_clus)
                else:
                    main_doc_clus.append(clus)

        main_doc_clus.sort()
        main_doc_clus = list(k for k, _ in itertools.groupby(main_doc_clus))
        return main_doc_clus
