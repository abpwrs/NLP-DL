from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()
example_sentence = 'this is an example showing off stop word filtration.'
example_words = ['python', 'pythoner', 'pythoning', 'pythoned', 'pythonly']
stemmed_words = [ps.stem(w) for w in example_words]
print(stemmed_words)
stemmed_example = [ps.stem(w) for w in word_tokenize(example_sentence)]
print(stemmed_example)
