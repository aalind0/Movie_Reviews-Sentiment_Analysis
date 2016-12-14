# Saving the classifier

import nltk
import random
from nltk.corpus import movie_reviews
import pickle

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
classifier = nltk.NaiveBayesClassifier.train(training_set)

#if you want to save and load in between.
#classifier_f = open("naivebayes.pickle", "rb")
#classifier = pickle.load(classifier_f)
#classifier_f.close()


# Testing now.
print("Naive Bayes Algorithm accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)

# most valuable words when it comes to positive and negative movie reviews.
classifier.show_most_informative_features(15)

# saving the classifier
save_classifier = open("naivebayes.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()
