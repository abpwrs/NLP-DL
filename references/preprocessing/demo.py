from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
words =["run","running","runner","runners","cacti","happenings","happening","happen"]

print('\nlemmatizing')
lem = WordNetLemmatizer()
for word in words:
    print(word,'-->', lem.lemmatize(word))

print('\nstemming')
lem = PorterStemmer()
for word in words:
    print(word,'-->', lem.stem(word)) 