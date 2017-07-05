from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
# Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, n_iter=5, random_state=42))])

import numpy as np

categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train',
                                  categories=categories, shuffle=True, random_state=42)

twenty_test = fetch_20newsgroups(subset='test',
                                 categories=categories, shuffle=True, random_state=42)
# classifier, raw data goes into ___.data and the labels go into ___.target

text_clf.fit(twenty_train.data, twenty_train.target)

docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)

print metrics.classification_report(twenty_test.target, predicted,
                                    target_names=twenty_test.target_names)

# print np.mean(predicted == twenty_test.target)

# count_vectorizer = CountVectorizer() # This supports N-grams of words or consecutive characters
# Once fitted, the vectorizer has built a dictionary of feature indices
# The index value of a word in the vocabulary is linked to its frequency in the whole training corpus
# tfidf_transformer = TfidfTransformer()
# Term frequency times inverse document frequency -- Step 1 and 2 of TWCNB Algorithm

# text_clf = text_clf.fit(twenty_train.data, twenty_train.target)



# x_train_counts = count_vectorizer.fit_transform(twenty_train.data)
# x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
# clf = MultinomialNB().fit(x_train_tfidf, twenty_train.target)
#
# docs_new = ['God is love', 'OpenGL on the GPU is fast']
# x_new_counts = count_vectorizer.transform(docs_new)
# x_new_tfidf = tfidf_transformer.transform(x_new_counts)
#
# predicted = clf.predict(x_new_tfidf)

