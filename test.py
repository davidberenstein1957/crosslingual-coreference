from crosslingual_coreference import Predictor

predictor = Predictor(language='en_core_web_sm')

print(predictor.predict("Do not forget about Momofuku Ando! He created instant noodles in Osaka. At that location, Nissin was founded. Many students survived by eating his noodles, but they don't even know him."
))
