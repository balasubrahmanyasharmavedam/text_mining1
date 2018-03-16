
# coding: utf-8

# In[1]:


import sklearn 
from __future__ import print_function 
from sklearn.decomposition import TruncatedSVD 
from sklearn.feature_extraction.text import TfidfVectorizer 
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import re


# In[3]:


import os


# In[7]:


os.chdir('C:/Users/subra/Desktop/ipynb/text mining/short_reviews')


# # Knowing the files

# In[8]:


with open('positive.txt', 'r') as file:
    string1 = file.read()


# In[12]:


string1_1=string1.split('\n')


# In[13]:


len(string1_1)


# In[14]:


with open('negative.txt', 'r') as file:
    string2 = file.read()


# In[15]:


string2_1= string2.split('\n')


# In[16]:


len(string2_1)


# # Pre-processing

# In[39]:


import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import re

stopwords = nltk.corpus.stopwords
eng_stopwords = stopwords.words('english')
ps = nltk.stem.PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

def basic_preprocessing(text):
    text = text.lower() #lowering
    text = re.sub(r'\[.*?\']', '', text) #removing all instances of citation brackets found in wiki articles
    text = word_tokenize(text)
    text = [word for word in text if word not in eng_stopwords] #removing stop words
    text = [word for word in text if len(word) > 1] #removing single character tokens
    text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    text = [ps.stem(word) for word in text]

    return(text)


# In[40]:


processed_texts_pos = [basic_preprocessing(text) for text in string1_1]


# In[41]:


processed_texts_neg = [basic_preprocessing(text) for text in string2_1]


# In[42]:


clean= processed_texts_pos+processed_texts_neg


# In[43]:


clean2=[]
for i in np.arange(0,len(clean),1):
    clean[i].reverse()
    clean2.append(" ".join(clean[i]))


# In[44]:


clean2[0]


# In[46]:


targets=["positive"]*5332+["negative"]*5332


# # TF-IDF

# In[61]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3),max_features=900,min_df=1,max_df=0.97)

tfidf= tfidf_vectorizer.fit_transform(clean2)


# In[62]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(tfidf,targets)


# In[63]:


x_train.shape


# # Naive bayes model

# In[64]:


from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB()

model.fit(x_train,y_train)


# In[65]:


from sklearn.metrics import accuracy_score, confusion_matrix

test_preds = model.predict(x_test)
print(accuracy_score(y_test,test_preds))


# # KNN Model

# In[66]:


from sklearn.neighbors import KNeighborsClassifier


# In[67]:


knn=KNeighborsClassifier(n_neighbors= 11)
knn.fit(x_train,y_train)
predictions= knn.predict(x_test)


# In[68]:


test_preds1 = knn.predict(x_test)
print(accuracy_score(y_test,test_preds1))

