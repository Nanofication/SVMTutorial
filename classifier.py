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
                                           alpha=1e-3, n_iter=5))])

import numpy as np
import ReadData

categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train',
                                  categories=categories, shuffle=True, random_state=42)

twenty_test = fetch_20newsgroups(subset='test',
                                 categories=categories, shuffle=True, random_state=42)
# classifier, raw data goes into ___.data and the labels go into ___.target


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

if __name__ == "__main__":
    # text_clf.fit(twenty_train.data, twenty_train.target)
    #
    # docs_test = twenty_test.data
    # predicted = text_clf.predict(docs_test)
    #
    # print metrics.classification_report(twenty_test.target, predicted,
    #                                     target_names=twenty_test.target_names)

    # print np.mean(predicted == twenty_test.target)

    amazon_data = []
    amazon_target = []

    ReadData.openFile()
    ReadData.shuffleTrainingData()
    data_set = ReadData.TRAINING_DATA

    for data in data_set:
        amazon_target.append(data["class"])
        amazon_data.append(data["sentence"])

    train_target = amazon_target[:800]
    train_data = amazon_data[:800]

    text_clf.fit(train_data, train_target)

    test_target = amazon_target[800:]
    test_data = amazon_data[800:]

    predicted = text_clf.predict(test_data)

    count = 10
    total = 0

    for i in range(count):
        total += np.mean(predicted == test_target)

    print total/count
