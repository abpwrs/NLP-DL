from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("cat"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("pythons"))
print(lemmatizer.lemmatize("better"))
print(lemmatizer.lemmatize("better", pos='a'))
