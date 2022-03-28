from spacy.language import Language

from .CrossLingualPredictor import CrossLingualPredictor as Predictor
from .CrossLingualPredictorSpacy import \
    CrossLingualPredictorSpacy as SpacyPredictor

__all__ = ["Predictor"]

@Language.factory(
    "xx_coref",
    default_config={
        "device": -1,
        "model_name": 'info_xlm'
    },
)
def make_crosslingual_coreference(
    nlp: Language,
    name: str,
    device: int,
    model_name: str,
):  
    return SpacyPredictor(
        language=nlp.path.name.split('-')[0],
        device=device,
        model_name=model_name,
    )

