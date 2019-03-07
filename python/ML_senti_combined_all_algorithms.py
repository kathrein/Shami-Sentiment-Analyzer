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
X_train = X_features
#Y_train = union.transform
X_test = union.transform(X_test)#union.transform(data_test.data)

print("Combined space has", X_features.shape[1], "features")

# this is for lev only
#svm = SGDClassifier(alpha=0.001, max_iter=50,penalty="l2")
# this is for high level lev, msa, eg, na
#svm = MultinomialNB(alpha=0.001) 
#
#svm.fit(X_features, y_train)
##pipeline = Pipeline([("features", union), ("svm", svm)])
#
#pred = svm.predict(X_test)
#score = metrics.accuracy_score(y_test, pred)
#print("accuracy:   %0.3f" % score)
#
#print("classification report:")
#print(metrics.classification_report(y_test, pred,target_names=target_names))
#
#print("confusion matrix:")
#print(metrics.confusion_matrix(y_test, pred))


# mapping from integer feature name to original token string
if opts.use_hashing:
    feature_names = None
else:
    feature_names = vectorizer.get_feature_names()

if opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
    print("done in %fs" % (time() - t0))
    print()

if feature_names:
    feature_names = np.asarray(feature_names)


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


# #############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    l_acc.append(score)
#    print(metrics.classification_report(data_test.target, pred,
#     target_names=data_test.target_names)) 
    opts.print_top10 = False
    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate(target_names):
                top10 = np.argsort(clf.coef_[i])[-10:]
                for t in top10:
                    if t > len(feature_names):
                        top10list = top10.tolist()#top10.delete(t,top10[top10.tolist().index(t)])
                        top10list.remove(t)
                        print(top10list)
                print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
        print()

    opts.print_report = True
    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))

    #opts.print_cm = True
    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
        (Perceptron(max_iter=50, tol=1e-3), "Perceptron"),
        (PassiveAggressiveClassifier(max_iter=50, tol=1e-3),
         "Passive-Aggressive"),
        #(KNeighborsClassifier(n_neighbors=10), "kNN"),
        #(RandomForestClassifier(n_estimators=100), "Random forest")
        ):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                       tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50, 
                                           penalty=penalty)))# alpha 0.00001 is better

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                       penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))
results.append(benchmark(ComplementNB(alpha=.1)))

print('=' * 80)
print("Naive Bayes 2")
results.append(benchmark(MultinomialNB(alpha=.1)))
results.append(benchmark(BernoulliNB(alpha=.1)))
results.append(benchmark(ComplementNB(alpha=.01)))

print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                  tol=1e-3))),
  ('classification', LinearSVC(penalty="l2"))])))

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()

for score ,clfname in zip(l_acc,clf_names):
    print(clfname,"  %0.3f" % score)
