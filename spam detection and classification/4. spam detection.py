#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd

text = pd.read_csv(r"E:\courses\natural lang processing\data sets for coure 1\SMSSpamCollection", sep = "\t", names = ["label", "messages"])

text.head()


#text cleaning.

import re
import nltk
from nltk.corpus import stopwords

corpus = []

from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()

for i in range(0, len(text)):
    review = re.sub('[^a-zA-Z]', " ", text['messages'][i])
    review = review.lower()
    review = review.split()
    review = [lem.lemmatize(word) for word in review if word not in (stopwords.words("english"))]
    review = " ".join(review)
    
    corpus.append(review)





# In[15]:


text['messages'][0]


# In[10]:


# converting into vectors

from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer()
x = tf.fit_transform(corpus).toarray()

y = pd.get_dummies(text['label'])
#y =y.drop(['spam'], axis = 1 )

y = y.iloc [:, 1].values


# In[ ]:





# In[11]:


# train test split

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.20, random_state = 0)


# In[12]:


#building the model

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

nb.fit(xtrain,  ytrain)

pred = nb.predict(xtest)

#calculating accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, ytest)
accuracy


# In[ ]:




