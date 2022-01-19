#!/usr/bin/env python
# coding: utf-8

# In[173]:


import pandas as pd

df = pd.read_csv(r"E:\courses\natural lang processing\data sets for coure 1\Combined_News_DJIA.csv")


# In[174]:


df.head()


# In[175]:


df = df.drop("Date", axis=1)


# In[176]:


df = df.dropna()


# In[177]:


df.shape


# In[178]:


# creating ip df and output df

x = df.drop("Label", axis=1)
y = df["Label"]


# In[ ]:





# In[179]:


#joing the text of all columns into one paragraph and making a new column as headline

x['headlines'] = x[x.columns[1:]].apply(
    lambda x: ''.join(x.dropna().astype(str)),
    axis=1
)


# In[180]:


x = x["headlines"]


# In[181]:


x.columns=["headlines"]
x.head()


# In[182]:


#text cleaning


import nltk
from nltk.stem import PorterStemmer
sm = PorterStemmer()
from nltk.corpus import stopwords
corpus = []

for i in range(len(x)):
    
    review = review.lower()
    review = review.split()
    review = [sm.stem(word) for word in review if word not in (stopwords.words("english")) ]
    review = " ".join(review)
    corpus.append(review)

    


# In[193]:


#converting into vectors

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range = (2,2))
x = cv.fit_transform(corpus).toarray()


# In[194]:


from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.30, random_state = 0)



from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
ada = RandomForestClassifier(n_estimators = 200 , criterion = "entropy")
ada.fit(xtrain, ytrain)
pred2=ada.predict(xtest)

ada.score(xtrain, ytrain)


# In[198]:


accuracy = accuracy_score(pred2, ytest)
accuracy


# In[ ]:




