# -*- coding: utf8 -*-
import gensim
import re
import numpy as np
from nltk import ngrams

# =========================
# ==== Helper Methods =====

# Clean/Normalize Arabic Text
def clean_str(text):
    search = ["أ","إ","آ","ة","_","-","/",".","،"," و "," يا ",'"',"ـ","'","ى","\\",'\n', '\t','&quot;','?','؟','!']
    replace = ["ا","ا","ا","ه"," "," ","","",""," و"," يا","","","","ي","",' ', ' ',' ',' ? ',' ؟ ',' ! ']
    
    #remove tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel,"", text)
    
    #remove longation
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)
    
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')
    
    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])
    
    #trim    
    text = text.strip()

    return text

def get_vec(n_model,dim, token):
    vec = np.zeros(dim)
    is_vec = False
    if token not in n_model.wv:
        _count = 0
        is_vec = True
        for w in token.split("_"):
            if w in n_model.wv:
                _count += 1
                vec += n_model.wv[w]
        if _count > 0:
            vec = vec / _count
    else:
        vec = n_model.wv[token]
    return vec

def calc_vec(pos_tokens, neg_tokens, n_model, dim):
    vec = np.zeros(dim)
    for p in pos_tokens:
        vec += get_vec(n_model,dim,p)
    for n in neg_tokens:
        vec -= get_vec(n_model,dim,n)
    
    return vec   

## -- Retrieve all ngrams for a text in between a specific range
def get_all_ngrams(text, nrange=3):
    text = re.sub(r'[\,\.\;\(\)\[\]\_\+\#\@\!\?\؟\^]', ' ', text)
    tokens = [token for token in text.split(" ") if token.strip() != ""]
    ngs = []
    for n in range(2,nrange+1):
        ngs += [ng for ng in ngrams(tokens, n)]
    return ["_".join(ng) for ng in ngs if len(ng)>0 ]

## -- Retrieve all ngrams for a text in a specific n
def get_ngrams(text, n=2):
    text = re.sub(r'[\,\.\;\(\)\[\]\_\+\#\@\!\?\؟\^]', ' ', text)
    tokens = [token for token in text.split(" ") if token.strip() != ""]
    ngs = [ng for ng in ngrams(tokens, n)]
    return ["_".join(ng) for ng in ngs if len(ng)>0 ]

## -- filter the existed tokens in a specific model
def get_existed_tokens(tokens, n_model):
    return [tok for tok in tokens if tok in n_model.wv ]





# ============================   
# ====== N-Grams Models ======

#t_model = gensim.models.Word2Vec.load('models/full_grams_cbow_300_twitter.mdl')
#
## python 3.X
#token = clean_str(u'منشان').replace(" ", "_")
## python 2.7
## token = clean_str(u'ابو تريكه'.decode('utf8', errors='ignore')).replace(" ", "_")
#
#if token in t_model.wv:
#    most_similar = t_model.wv.most_similar( token, topn=10 )
#    for term, score in most_similar:
#        term = clean_str(term).replace(" ", "_")
#        if term != token:
#            print(term, score)

# تريكه 0.752911388874054
# حسام_غالي 0.7516342401504517
# وائل_جمعه 0.7244222164154053
# وليد_سليمان 0.7177559733390808
# ...

# =========================================
# == Get the most similar tokens to a compound query
# most similar to 
# عمرو دياب + الخليج - مصر

#pos_tokens=[ clean_str(t.strip()).replace(" ", "_") for t in ['بشار الأسد', 'مصر'] if t.strip() != ""]
#neg_tokens=[ clean_str(t.strip()).replace(" ", "_") for t in ['سوريا'] if t.strip() != ""]
#
#vec = calc_vec(pos_tokens=pos_tokens, neg_tokens=neg_tokens, n_model=t_model, dim=t_model.vector_size)
#
#most_sims = t_model.wv.similar_by_vector(vec, topn=10)
#for term, score in most_sims:
#    if term not in pos_tokens+neg_tokens:
#        print(term, score)

# راشد_الماجد 0.7094649076461792
# ماجد_المهندس 0.6979793906211853
# عبدالله_رويشد 0.6942606568336487
# ...

# ====================
# ====================




# ============================== 
# ====== Uni-Grams Models ======

#t_model = gensim.models.Word2Vec.load('models/full_uni_cbow_300_twitter.mdl')
#
## python 3.X
#token = clean_str(u'زي')
## python 2.7
## token = clean_str('تونس'.decode('utf8', errors='ignore'))
#
#most_similar = t_model.wv.most_similar( token, topn=10 )
#for term, score in most_similar:
#    print(term, score)

# ليبيا 0.8864325284957886
# الجزائر 0.8783721327781677
# السودان 0.8573237061500549
# مصر 0.8277812600135803
# ...



# get a word vector
#word_vector = t_model.wv[ token ]
#print(token, word_vector)