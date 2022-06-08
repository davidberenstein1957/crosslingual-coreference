import warnings

import nltk

nltk.download("omw-1.4")
from typing import Union  # noqa: E402

from spacy.language import Language  # noqa: E402

from .CrossLingualPredictor import CrossLingualPredictor as Predictor  # noqa: E402
from .CrossLingualPredictorSpacy import (  # noqa: E402
    CrossLingualPredictorSpacy as SpacyPredictor,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
__all__ = ["Predictor"]


@Language.factory(
    "xx_coref",
    default_config={
        "device": -1,
        "model_name": "minilm",
        "chunk_size": None,
        "chunk_overlap": 2,
    },
)
def make_crosslingual_coreference(
    nlp: Language,
    name: str,
    device: int,
    model_name: str,
    chunk_size: Union[int, None],
    chunk_overlap: int,
):
    return SpacyPredictor(
        language=nlp.path.name.split("-")[0],
        device=device,
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
