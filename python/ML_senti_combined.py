#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 12:08:25 2019

@author: xabuka
"""

# best Identify langauge code


from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.model_selection import train_test_split




#data_set = load_files('../corpora/Balanced_Shami/train/dialects', encoding = 'utf-8',decode_error='ignore')
#data_test = load_files('../corpora/Balanced_Shami/test', encoding = 'utf-8',decode_error='ignore')
#X_train = data_set.data
#y_train = data_set.target
#X_test = data_test.data
#y_test = data_test.target


data_set = load_files('../PalSenti/', encoding = 'utf-8',decode_error='ignore')
X_train, X_test, y_train, y_test = train_test_split(data_set.data, data_set.target, test_size=0.2, random_state=42)
print('data loaded')


# order of labels in `target_names` can be different from `categories`
target_names = data_set.target_names


def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6


data_train_size_mb = size_mb(x_train)
data_test_size_mb = size_mb(x_test)

print("%d documents - %0.3fMB (training set)" % (
    len(x_train), data_train_size_mb))
print("%d documents - %0.3fMB (test set)" % (
    len(x_test),data_test_size_mb))
print("%d categories" % len(target_names))
print()




#word_vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'word', ngram_range=(1,1))
char_vect  = TfidfVectorizer(max_features = 50000, sublinear_tf=True,norm ='l1', max_df=0.75,analyzer = 'char_wb', ngram_range=(2,5))


union = FeatureUnion([("w_v", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'word', ngram_range=(1,1)
                                 )),
                       ("c_wb", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'char_wb', ngram_range=(2,5)
                                 )),
#                       ("c_v", TfidfVectorizer(sublinear_tf=True, max_df=0.5,analyzer = 'char', ngram_range=(2,5)
#                                 ))
                       ])

#union.fit_transform(data_train.data)
X_features = union.fit_transform(X_train) #union.fit_transform(data_train.data)
#Y_train = union.transform
X_test = union.transform(X_test)#union.transform(data_test.data)

print("Combined space has", X_features.shape[1], "features")

# this is for lev only
#svm = SGDClassifier(alpha=0.001, max_iter=50,penalty="l2")
# this is for high level lev, msa, eg, na
svm = MultinomialNB(alpha=0.0001) 

svm.fit(X_features, y_train)
#pipeline = Pipeline([("features", union), ("svm", svm)])

pred = svm.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)

print("classification report:")
print(metrics.classification_report(y_test, pred,target_names=target_names))

print("confusion matrix:")
print(metrics.confusion_matrix(y_test, pred))