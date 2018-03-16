
# coding: utf-8

# In[35]:


import numpy as np


# In[16]:


import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import re

stopwords = nltk.corpus.stopwords
eng_stopwords = stopwords.words('english')
ps = nltk.stem.PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

def basic_preprocessing(text):
    
    text = word_tokenize(text)
    text = text.lower()
    text = [word for word in text if word not in eng_stopwords]
    text = [wordnet_lemmatizer.lemmatize(word) for word in text]
  

    return(text)


# In[20]:


string = 'The quick brown fox. Jumped over the lazy dog.'
senttok= sent_tokenize(string)
for sentence in senttok:
    processed_tokens = basic_preprocessing(sentence)
    print(sentence)
    print(processed_tokens)


# In[25]:


len(senttok)


# In[63]:


print('1) no of words in the text:',len(string.split()))
print("2) no.of sentences in the string:", len(senttok))
print("no.of words in sentence1:",len(senttok[0].split()))
print("no.of words in sentence2:",len(senttok[1].split()))
unique_elements, counts_elements = np.unique(string.split(), return_counts=True)
print("3) unique words in the text :")
print(np.asarray((unique_elements, counts_elements)))
print('4) avg. no of characters of a word in the text:', len(string)/len(string.split()))
print("5) words list w/o stop words in the text:", processed_tokens)
print("6) common words in the text:",)
print(np.asarray((unique_elements, counts_elements)))


# In[46]:


len(string)

