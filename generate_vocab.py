# Naive Basil Pesto - Miro Furtado & Simon Shen
import torch, spacy, pickle
from torchtext import data, datasets
from namedtensor.text import NamedField

"Generate vocab from spacy: de -> en"
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')
def tokenize_de(text):
	return [tok.text for tok in spacy_de.tokenizer(text)]
def tokenize_en(text):
	return [tok.text for tok in spacy_en.tokenizer(text)]
BOS_WORD = '<s>'
EOS_WORD = '</s>'
DE = NamedField(names=('srcSeqlen',), tokenize=tokenize_de)
EN = NamedField(names=('trgSeqlen',), tokenize=tokenize_en,
				init_token = BOS_WORD, eos_token = EOS_WORD) # only target needs BOS/EOS
MAX_LEN = 20
print("Making data")
train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN), 
										filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
										len(vars(x)['trg']) <= MAX_LEN)
MIN_FREQ = 5
print("Building vocab")
DE.build_vocab(train.src, min_freq=MIN_FREQ)
EN.build_vocab(train.trg, min_freq=MIN_FREQ)

print("Dumping")
with open('DE_EN_vocab.pkl', 'wb') as f:
    pickle.dump(DE.vocab, f)
    pickle.dump(EN.vocab, f)