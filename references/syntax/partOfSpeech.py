import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')

tokenizer = PunktSentenceTokenizer(train_text=train_text)
tokenized = tokenizer.tokenize(sample_text)

for i in tokenized:
    words = nltk.word_tokenize(i)
    tagged = nltk.pos_tag(words)
    print(tagged)