
# coding: utf-8

# In[50]:


from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import text
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
import math
import nltk
import json
import os
from nltk.corpus import stopwords
PROJ_NAME = "BRNN_TOXIC"
MAX_COMMENT_LENGTH = 1500
stops = stopwords.words('english')
LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
NUM_CLASSES = len(LABELS)


# In[15]:


with open("config.json",'r') as f:
    config_file = json.load(f)["BASE_CONFIG"]
with open(config_file,'r') as f:
    config = json.load(f)
data_dir=os.path.join(config["data_dir"],PROJ_NAME)
model_dir=os.path.join(config["model_dir"],PROJ_NAME)
out_dir=os.path.join(config["out_dir"],PROJ_NAME)


# In[16]:


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


# In[17]:


df = pd.read_csv(os.path.join(data_dir, "train.csv"))
a_df = adjust_class_balance(df, LABELS, 1/(len(LABELS)+1))


# In[35]:


tokenizer = text.Tokenizer(num_words=10000)


# In[36]:


tokenizer.fit_on_texts(a_df["comment_text"].values)


# In[39]:


encoded_docs = tokenizer.texts_to_matrix(a_df["comment_text"].values, mode='count')


# In[43]:


encoded_docs[0].shape


# In[60]:


model = Sequential()
model.add(tf.keras.layers.Dense(1000, input_shape=(10000,)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dense(500))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dense(200))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dense(NUM_CLASSES))
model.add(tf.keras.layers.Activation('sigmoid'))
model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[61]:


Y = np.array(a_df[LABELS].values)


# In[62]:


X_train, X_test, y_train, y_test = train_test_split(encoded_docs,Y,test_size=0.33)


# In[63]:


model.fit(
    x=X_train,
    y=y_train,
    batch_size=10,
    epochs=20,
    verbose=1,
    shuffle=True,
)


# In[64]:


print(model.evaluate(X_test,y_test))


# In[67]:


model.save(os.path.join(model_dir,"IE_CNN_SGD_10"))
model.save_weights(os.path.join(model_dir,"IE_CNN_SGD_10_WEIGHTS"))

