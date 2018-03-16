
# coding: utf-8

# In[102]:


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


# # Importing corpus

# In[103]:


from nltk.corpus import state_union

_files_all_speechs = state_union.fileids()

all_raw_speeches = []
for _file_ in _files_all_speechs:
    all_raw_speeches.append(state_union.raw(_file_))

all_categories = [x.split('-')[1].split('.')[0] for x in _files_all_speechs]

print('Number of Speeches:', len(all_raw_speeches))
print(all_categories)


# In[104]:


all_raw_speeches[1]


# # Data pre-processing [Text Cleansing]

# In[105]:


clean1=[]
for i in np.arange(0,len(all_raw_speeches),1):
    clean1.append(all_raw_speeches[i].split('\n')[4:])


# In[106]:


clean1[0][0]


# In[107]:


clean2=[]
for i in np.arange(0,len(clean1),1):
    clean1[i].reverse()
    clean2.append(" ".join(clean1[i]))


# In[108]:


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
    text = re.sub(r'\[.*?\]', '', text) #removing all instances of citation brackets found in wiki articles
    text = word_tokenize(text)
    text = [word for word in text if word not in eng_stopwords] #removing stop words
    text = [word for word in text if len(word) > 1] #removing single character tokens
    text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    text = [ps.stem(word) for word in text]

    return(text)


# In[109]:


processed_texts = [basic_preprocessing(text) for text in clean2]


# In[116]:


processed_texts[0]


# In[111]:


clean3=[]
for i in np.arange(0,len(processed_texts),1):
    processed_texts[i].reverse()
    clean3.append(" ".join(processed_texts[i]))


# In[117]:


clean3[0]


# In[113]:


targets=all_categories


# In[114]:


targets


# # TF-IDF

# In[220]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3),max_features=1000,min_df=1,max_df=0.97)

tfidf= tfidf_vectorizer.fit_transform(clean3)


# In[221]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(tfidf,targets)


# In[222]:


x_train.shape


# # Naive Bayes Model

# In[223]:


from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB()

model.fit(x_train,y_train)


# In[224]:


from sklearn.metrics import accuracy_score, confusion_matrix

test_preds = model.predict(x_test)
print(accuracy_score(y_test,test_preds))


# # KNN Model

# In[204]:


from sklearn.neighbors import KNeighborsClassifier


# In[160]:


knn=KNeighborsClassifier(n_neighbors= 11)
knn.fit(x_train,y_train)
predictions= knn.predict(x_test)


# In[161]:


test_preds1 = knn.predict(x_test)
print(accuracy_score(y_test,test_preds1))

