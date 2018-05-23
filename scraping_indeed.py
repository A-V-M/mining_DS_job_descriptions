# -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:41:12 2018

@author: andreas
"""
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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from random import randint, shuffle

def get_job_links(url_listings):

    page = requests.get(url_listings)
    soup = BeautifulSoup(page.content)    
    
    found = 0
    n = -1
    
    
    # parse all the script instances from the content
    # this is javascript so we need to weed it out from html
    while found == 0:
        
        n = n + 1
    
        script = soup.findAll('script')[n].string
        if script!=None:
            
            if "jobmap" in script:
                found = 1
        
    #thanks to StackExchange I got the following to isolate the 
    #relevant fields which contain job descriptions
    script = soup.findAll('script')[11].string
    data = script.split("jobmap = ", 1)[-1].rsplit(';', 1)[0]
    
    job_urls = []
    
    #parse the fields and get the urls
    parser = Parser()
    tree = parser.parse(data)
    for node in nodevisitor.visit(tree):
       if isinstance(node, ast.Assign):
           value = getattr(node.left, 'value', '')
           if value=="jk":
               jobID = getattr(node.right, 'value', '')
               jobID = re.sub('\'', '', jobID)
               new = "https://www.indeed.co.uk/viewjob?jk="+ jobID + "&q"
               job_urls.append(new)
        
    return job_urls
    
def get_job_description(job_url):
    
    try:    
        job_page = Article(url = job_url)
        job_page.download()
        job_page.parse()
        job_text = job_page.text
                    
    except ArticleException:
        print("url issues during text retrieval! no txt saved")
        job_text=[]

    except Exception:
        print("error during text retrieval! no txt saved")
        job_text=[]
    
    return job_text
    
root_url = "https://www.indeed.co.uk/jobs?q=\"data+scientist\"&start="
    
job_docs = pd.Series()

n_scraps =  range(0,110)

for scrap in n_scraps:
    
    ind = scrap * 10
    
    url_listing = root_url + str(ind)
    
    found_links = get_job_links(url_listing)
            
    adict = []
            
    for link in range(0,len(found_links)):
        try:        
            txt = get_job_description(found_links[link])
        except Exception:
            txt = []
            print("error")
            
        if txt != []:        
                        
            adict.append(txt)
                
    job_docs = pd.concat([job_docs,pd.Series(adict)],axis=0)    
   
    time_int = int(round(abs(np.random.normal(loc=18.0, scale=6.0, size=None)) + (np.random.rand() * 10)))
    time.sleep(time_int)
    
    print('another scrap for the heap...!')
       
output = open('data4.pkl', 'wb')
pickle.dump(job_docs, output)
output.close()
