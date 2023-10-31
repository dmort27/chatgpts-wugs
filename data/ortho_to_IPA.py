import pandas as pd
import epitran
import panphon

LANGUAGE = "tam"
SET = "trn"

EPITRAN_TAGS = {
    "deu": "deu-Latn",
    "eng": "eng-Latn",
    "tam": "tam-Taml",
    "tur": "tur-Latn"
}

filename = f"{LANGUAGE}.{SET}"
new_filename = f"{LANGUAGE}_ipa.{SET}"

if SET == "nonce":
    names = ['input_orth', 'morphosyn']
else:
    names = ['input_orth', 'morphosyn', 'output_orth']
data = pd.read_csv(filename, sep='\t', names=names)

epi = epitran.Epitran(EPITRAN_TAGS[LANGUAGE])
ft = panphon.FeatureTable()

lem_phon = [' '.join(ft.ipa_segs(epi.transliterate(word))) for word in data['input_orth']]
form_phon = [' '.join(ft.ipa_segs(epi.transliterate(word))) for word in data['output_orth']]

data['input_IPA'] = lem_phon
data['output_IPA'] = form_phon

data.to_csv(new_filename, sep='\t',
            columns=['input_IPA', 'output_IPA', 'morphosyn', 'input_orth', 'output_orth'],
            header=False, index=False)
