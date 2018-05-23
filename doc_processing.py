# -*- coding: utf-8 -*-
"""
Created on Fri May 18 19:52:09 2018

@author: andreas
"""

def ngram_vectorise(txt):
    
    num_docs = txt.shape[0]
        
    nonacceptable_tags = ['CC','CD','DT','IN','EX','LS','MD','POS','PDT','RP','TO','WDT','WP','WRB','PRP$','PRP']               
    
    tag_combos = [item for n,item in enumerate(combinations(nonacceptable_tags[::-1],2))] + \
                 [item for n,item in enumerate(combinations(nonacceptable_tags,2))] + \
                 [tuple([i,i]) for i in nonacceptable_tags]
                    
    translator = str.maketrans('', '', string.punctuation)
    
    text_processed = [None] * num_docs
 
    for i in range(0,num_docs):
        
        processed=txt[i].lower().translate(translator).replace('\n',' ').replace('  ',' ')
        website_suffix = processed.find("save job original job" )          
        text_processed[i]=processed[:website_suffix]
           
    ngram_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1,sublinear_tf=True)        
    
    tf_idf = ngram_vectorizer.fit_transform(text_processed)
    fnames = ngram_vectorizer.get_feature_names()
    idf = ngram_vectorizer.idf_
    dense = tf_idf.todense()
    
    valid_ngrams = np.zeros([len(fnames)])
    
    for i,fname in enumerate(fnames):
        
        fname = fname.split()
        
        if len(fname) > 1:
            
            valid_ngrams[i] = tuple(dict(nltk.pos_tag(tuple(fname))).values()) not in tag_combos
            
        else:
            
            valid_ngrams[i] = nltk.pos_tag([fname[0]])[0][1] not in nonacceptable_tags

    tfidf_matrix = dense[:,np.where(valid_ngrams)[0]]
    
    labels = np.array(fnames)[np.where(valid_ngrams)]
    
    ngramMat = pd.DataFrame(data=tfidf_matrix,columns=labels)

    return ngramMat,idf
    