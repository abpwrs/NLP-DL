
# coding: utf-8

# In[1]:


import word2vec
import pandas as pd
import numpy as np
import json
import os
import string
from nltk.tokenize import word_tokenize
import math
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
STOPS = stopwords.words('english')
PROJ_NAME = "BRNN_TOXIC"
MAX_COMMENT_LENGTH = 1500
from collections import Counter
LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]


# import word2vec
# import pandas as pd
# import numpy as np
# import json
# import os
# from nltk.tokenize import word_tokenize
# import math
# import nltk
# from nltk.corpus import stopwords
# from sklearn.model_selection import train_test_split
# STOPS = stopwords.words('english')
# PROJ_NAME = "BRNN_TOXIC"
# MAX_COMMENT_LENGTH = 1500
# from collections
# LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

# In[4]:


with open("config.json",'r') as f:
    config_file = json.load(f)["BASE_CONFIG"]
with open(config_file,'r') as f:
    config = json.load(f)
data_dir=os.path.join(config["data_dir"],PROJ_NAME)
model_dir=os.path.join(config["model_dir"],PROJ_NAME)
out_dir=os.path.join(config["out_dir"],PROJ_NAME)


# In[3]:


model = word2vec.load(os.path.join(data_dir,"text.bin"))


# In[4]:


len(model.vocab),model.vectors.shape


# In[6]:


# find 50 most often negative and 50 most often positive words
df = pd.read_csv(os.path.join(data_dir,"train.csv"))
dfs = {}
null = df.copy()
for name in LABELS:
    dfs[name] = df.loc[(df[name] == 1)]
    null.drop(null[null[name]==1].index,axis=0,inplace=True)


total_words = 0
all_words = set()
i = 0
for comment in df["comment_text"]:
    if i%10000 == 0:
        print(i/len(df))
    words = word_tokenize(comment)
    if len(words) < MAX_COMMENT_LENGTH:
        for word in words:
            if word not in STOPS and word not in string.punctuation:
                total_words+=1
                all_words.add(word.lower())
    i+=1
total_words


# In[7]:


frequency_dict = {}
i=0
for c,d in dfs.items():
    print(i)
    word_freq = {}
    for class_comment in d["comment_text"]:
        for word in word_tokenize(class_comment):
            if word in all_words and word not in string.punctuation:
                if word in word_freq.keys():
                    word_freq[word] = word_freq[word]+1
                else:
                    word_freq[word]=1
    frequency_dict[c] = word_freq
    i+=1


# In[8]:


null_freq = {}
for class_comment in null["comment_text"]:
        for word in word_tokenize(class_comment):
            if word in all_words and word not in string.punctuation:
                if word in null_freq.keys():
                    null_freq[word] = null_freq[word]+1
                else:
                    null_freq[word]=1


# In[9]:


selected = []
for k,v in frequency_dict.items():
    print(k)
    i = 0
    while(i<13):
        c = Counter(v).most_common(1)
        if c[0][0] not in selected and c[0][0] in model.vocab:
            selected.append(c[0][0])
            i+=1
        v[c[0][0]]=0
i=len(selected)
while(len(selected)<100):
    c = Counter(null_freq).most_common(1)
    if c[0][0] not in selected and c[0][0] in model.vocab:
        selected.append(c[0][0])
        i+=1
    null_freq[c[0][0]]=0
selected


# In[10]:


def adjust_class_balance(df: pd.DataFrame, interested_labels, thresh):
    dfs = {}
    null = df.copy()
    for name in interested_labels:
        dfs[name] = df.loc[(df[name] == 1)]
        null.drop(null[null[name]==1].index,axis=0,inplace=True)

    print("NULL:", 100*(len(null)/len(df)))
    for name, d in dfs.items():
        print("Initial percentage of DF for", name, "is", 100*(len(d)/len(df)))

    print("Each label will now have at least", thresh*100,"% of the origional df size")
    adjusted_df = null.sample(int(thresh*len(df))) # get a subsample of null cases


    for n, d in dfs.items():
        i=0
        for times in range(math.ceil((thresh/(len(d)/len(df))+1))):
            adjusted_df = adjusted_df.append(d)
            i+=1
        print(n,"upsampled",i,"times")
    return adjusted_df
a_df = adjust_class_balance(df, LABELS, 1/(len(LABELS)+1))


# In[11]:


model["hello"]


# In[ ]:


def data_from_comment(comment):
    vec = []
    clean = [ x.lower() for x in word_tokenize(comment)]
    for sel in selected:
        if sel in clean:
            vec.append(model[sel])
        else:
            vec.append(np.zeros(100))
    return vec



a_data = []
a_labels = []
for index, row in df.iterrows():
    if index %1000 ==0:
        print(index/len(df)*100)
    included = False
    temp_data = data_from_comment(row["comment_text"])
    for counter, include in enumerate(row[LABELS].values):
        if include == 1:
            one_hot = np.zeros(len(LABELS))
            one_hot[counter] = 1
            a_labels.append(one_hot)
            a_data.append(temp_data)
            included = True

    if not included:
        a_labels.append(np.zeros(len(LABELS)))
        a_data.append(temp_data)



# In[ ]:


X = np.array(a_data)
Y = np.array(a_labels)


# In[3]:


print(X.shape,Y.shape)
np.save(os.path.join(data_dir,"X_ALL.npy"),X)
np.save(os.path.join(data_dir,"Y_ALL.npy"),Y)

