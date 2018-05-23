# -*- coding: utf-8 -*-
"""
Created on Mon May 21 17:43:24 2018

@author: andreas
"""


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import re
import time
import pandas as pd
import numpy as np
import sys
import json
from slimit import ast
from slimit.parser import Parser
from slimit.visitors import nodevisitor
import pickle
from itertools import combinations
from nltk.util import ngrams
import string
import matplotlib.pyplot as plt
import nltk
from sklearn.metrics.pairwise import cosine_distances
from sklearn import manifold
from random import randint, shuffle

#load in the scraped data

file = open("data3.pkl",'rb')
job_docs = pickle.load(file)
file.close()

#extract tf-idf and vectorise

ngramMat = ngram_vectorise(job_docs)


# remove words which unnecessary
labels_for_removal=[k for i,k in enumerate(ngramMat.columns) if "cwjobscouk" in k] + \
       [k for i,k in enumerate(ngramMat.columns) if "such" in k] + \
       [k for i,k in enumerate(ngramMat.columns) if "please" in k] + \
       [k for i,k in enumerate(ngramMat.columns) if "job" in k] + \
       [k for i,k in enumerate(ngramMat.columns) if "london" == k] + \
       [k for i,k in enumerate(ngramMat.columns) if "be" == k] + \
       [k for i,k in enumerate(ngramMat.columns) if "is" == k] + \
       [k for i,k in enumerate(ngramMat.columns) if "are" == k] + \
       [k for i,k in enumerate(ngramMat.columns) if "more" == k]

ngramMat.drop(labels_for_removal, inplace=True, axis=1)

#%% add R and C++. python filters out single-character words and punctuation.
num_docs = ngramMat.shape[0]

ngram_vectorizer = TfidfVectorizer(analyzer='char',ngram_range=(1, 3), min_df=1,sublinear_tf=True,lowercase=False)
tf = ngram_vectorizer.fit_transform(job_docs)
fnames = ngram_vectorizer.get_feature_names()
dense = tf.todense()

Cpp = [i for i,k in enumerate(fnames) if "C++" == k]
new=np.reshape(np.array(dense[:,Cpp]),num_docs)
ngramMat['C++'] = pd.Series(new,index=ngramMat.index)

R = [i for i,k in enumerate(fnames) if " R" == k]
new=np.reshape(np.array(dense[:,R]),num_docs)
ngramMat['R'] = pd.Series(new,index=ngramMat.index)

#%% remove duplicate docs

DM_docs = cosine_distances(ngramMat)

duplicates = np.zeros(num_docs)

for n in range(0,num_docs):
    doc_dupes = np.sort(np.where(DM_docs[n,:] == 0))[0][1:]
    duplicates[doc_dupes] = 1

docs_for_removal = np.where(duplicates.astype('int')==1)[0]
    
ngramMat.drop(docs_for_removal,inplace=True,axis=0)

#%%setup some constants used later

word_occurences=np.sum(ngramMat)
num_words = word_occurences.shape[0]
word_names = np.array(ngramMat.columns)

#%%plot tf-idf values for unigrams / bigrams

wLength = [len(k.split()) for i,k in enumerate(ngramMat.columns)]
unigrams = ngramMat.columns[np.where(np.array(wLength) == 1)[0]]
bigrams = ngramMat.columns[np.where(np.array(wLength) == 2)[0]]

vals=word_occurences[unigrams]

top50unigrams = vals[np.argsort(vals)][-50:]

plt.figure()
pl = plt.barh(np.arange(0,50), top50unigrams.values, 0.75)
ind_labels = np.arange(0.5,50,1)
plt.yticks(ind_labels,top50unigrams.index)
plt.tight_layout()
axes = plt.gca()
axes.set_xlim([np.floor(np.min(top50unigrams)),np.ceil(np.max(top50unigrams))])
plt.xlabel('tf-idf')

vals=word_occurences[bigrams]

top50bigrams = vals[np.argsort(vals)][-50:]

plt.figure()
pl = plt.barh(np.arange(0,50), top50bigrams.values, 0.75)
ind_labels = np.arange(0.5,50,1)
plt.yticks(ind_labels,top50bigrams.index)
plt.tight_layout()
axes = plt.gca()
axes.set_xlim([np.floor(np.min(top50bigrams)),np.ceil(np.max(top50bigrams))])
plt.xlabel('tf-idf')

#%%mds plot

mds = manifold.MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='precomputed')

top100=np.array(word_occurences[np.argsort(word_occurences.values)[-100:]].index)

X = (ngramMat[top100].T.values > 0) * 1

DM = cosine_distances(X)

best_run, final_cl=probKMeans(X,2,1000,5)

cl = final_cl[:,best_run]

fig, ax = plt.subplots()

for i, txt in enumerate(top100):
    
    ax.annotate(txt, (mds_fit[i,0],mds_fit[i,1]))

plt.scatter(mds_fit[:, 0], mds_fit[:, 1], s=(np.exp(word_occurences[top100].values/5)),c=cl, lw=0)


#%% some words cluster together. we need to know a little bit more...

for word in ['experience','scientist','science','machine','lead','ability']:
    
    labels=[k for i,k in enumerate(ngramMat.columns) if word in k]
    
    X=ngramMat[labels].values

    DM = cosine_distances(X.T > 0)
    
    Z = hierarchy.linkage(DM,method='average')
    
    T = hierarchy.fcluster(Z,1.1)
    
    node_i = np.where(np.array(labels) == word)[0][0]
    
    node_cluster = T[node_i]
    
    print(np.array(labels)[(np.where(T==node_cluster))])

#%% conditional probability of certain words
    
words = [['R','sql','python','excel','matlab','C++']]

for word_class in range(0,len(words)):
    
    plt.figure(figsize=(7, 4))
    
    
    for i,w in enumerate(words[word_class]):
        
        cond_probs = np.zeros(num_words)
        
        for n in range(0,num_words):
            
                vals = (ngramMat[ngramMat.columns[n]].values>0) * 1
            
                if np.dot(vals,vals) > 250:
                    
                    cond_probs[n]=np.dot((ngramMat[w]>0) *1,vals) / (np.dot(vals,vals))
    
        
        sb=int(['32' + str(i+1)][0])    
        ax = plt.subplot(sb)
    
        top_words =  word_names[np.argsort(cond_probs)][-16:-1]
        vals = cond_probs[np.argsort(cond_probs)[-16:-1]]
        ax.barh(np.arange(0,15), vals, 0.75)
        ax.set_xlim([np.min(vals)-0.05,np.ceil(np.max(vals))])
        ax.set_title(w)
        ind_labels = np.arange(0.5,15,1)
        plt.yticks(ind_labels,top_words)
        ax.set_ylim([0,15])
        ax.set_xlim([np.min(vals)-0.05,np.ceil(np.max(vals))])    
        
#%% NMF for clustering and plotting

model = NMF(n_components=2, init='random', random_state=0,beta_loss=1,solver='mu')
W = model.fit_transform(ngramMat.values)
H = model.components_
scores=np.dot(ngramMat.values,H.T)
means=np.tile(np.mean(scores,axis=0),(num_docs,1))
scores_centered = scores - means

words = [['R','sql','python','excel','matlab','C++'],
         ['computer science','statistics','mathematics','machine learning','economics'],
         ['deep learning','big data','data management','data analysis','predictive modelling']]
         
for word_class in range(0,len(words)):
    
    fig, ax = plt.subplots()
    
    plt.scatter(scores_centered[:, 0], scores_centered[:, 1],lw=0)       
    arrow_size, text_pos = 2.0, 2.0,
    
        # projections of the original features
    for i, v in enumerate(words[word_class]):
    
        key_i = [key_i for key_i,w in enumerate(ngramMat.columns) if w == v]
        
        ax.arrow(0, 0, arrow_size*H[0,key_i][0], arrow_size*H[1,key_i][0], 
                  head_width=0.05, head_length=0.05, linewidth=1, color='red')
        ax.text(H[0,key_i][0]*text_pos, H[1,key_i][0]*text_pos, v, color='black', 
                 ha='center', va='center', fontsize=16)    
    
    axes = plt.gca()
    #axes.set_ylim([-1,1])
    plt.grid()

