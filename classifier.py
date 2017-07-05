from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB




count_vectorizer = CountVectorizer() # This supports N-grams of words or consecutive characters
# Once fitted, the vectorizer has built a dictionary of feature indices
# The index value of a word in the vocabulary is linked to its frequency in the whole training corpus
tfidf_transformer = TfidfTransformer()
# Term frequency times inverse document frequency -- Step 1 and 2 of TWCNB Algorithm

categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train',
                                  categories=categories, shuffle=True, random_state=42)



x_train_counts = count_vectorizer.fit_transform(twenty_train.data)
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
clf = MultinomialNB().fit(x_train_tfidf, twenty_train.target)

docs_new = ['God is love', 'OpenGL on the GPU is fast']
x_new_counts = count_vectorizer.transform(docs_new)
x_new_tfidf = tfidf_transformer.transform(x_new_counts)

predicted = clf.predict(x_new_tfidf)

