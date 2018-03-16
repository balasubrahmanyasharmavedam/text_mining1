
# coding: utf-8

# In[1]:


import os


# In[2]:


os.chdir('E:/Insofe/piazza resources/machine learning/text mining/11-03/classification_datasets/spamHam (Document Classification) (Activity)')


# In[3]:


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

