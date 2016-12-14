# Saving the classifier

import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


# this takes the most of the algorithm time.
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)


all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())
    
all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    
    return features
    
#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

# Training and testing sets splitted up.
training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# posterior = prior occurences * likelihoood / evidence
#classifier = nltk.NaiveBayesClassifier.train(training_set)

#if you want to save and load in between.
#classifier_f = open("naivebayes.pickle", "rb")
#classifier = pickle.load(classifier_f)
#classifier_f.close()


# Testing now.
print("Original Naive Bayes Algorithm accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)

# most valuable words when it comes to positive and negative movie reviews.
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

# GaussianNB, BernoulliNB
#GaussianNB = SklearnClassifier(GaussianNB())
#GaussianNB.train(training_set)
#print("GaussianNB accuracy percent:", (nltk.classify.accuracy(GaussianNB, testing_set))*100)

#BernoulliNB = SklearnClassifier(BernoulliNB())
#BernoulliNB.train(training_set)
#print("BernoulliNB accuracy percent:", (nltk.classify.accuracy(BernoulliNB, testing_set))*100)

# LogisticRegression, SGDClassifier
# SVC, LinearSVC, NuSVC

LogisticRegression = SklearnClassifier(LogisticRegression())
LogisticRegression.train(training_set)
print("LogisticRegression accuracy percent:", (nltk.classify.accuracy(LogisticRegression, testing_set))*100)

SGDClassifier = SklearnClassifier(SGDClassifier())
SGDClassifier.train(training_set)
print("SGDClassifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier, testing_set))*100)

SVC = SklearnClassifier(SVC())
SVC.train(training_set)
print("SVC accuracy percent:", (nltk.classify.accuracy(SVC, testing_set))*100)

LinearSVC = SklearnClassifier(LinearSVC())
LinearSVC.train(training_set)
print("LinearSVC accuracy percent:", (nltk.classify.accuracy(LinearSVC, testing_set))*100)

NuSVC = SklearnClassifier(NuSVC())
NuSVC.train(training_set)
print("NuSVC accuracy percent:", (nltk.classify.accuracy(NuSVC, testing_set))*100)

