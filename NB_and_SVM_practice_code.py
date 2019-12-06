# Import:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from dataset import Modules

# Get training data:
module_train = Modules(subset='train', shuffle=True)

# Set NB Pipeline:
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

text_clf = text_clf.fit(module_train.description, module_train.label)
# Get testing data:
module_test = Modules(subset='test', shuffle=True)
predicted = text_clf.predict(module_test.description)
print(np.mean(predicted == module_test.label))

# Set SVM Pipeline
text_clf_svm = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),
('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter_no_change=5, random_state=42))])

text_clf_svm = text_clf_svm.fit(module_train.description, module_train.label)
predicted_svm = text_clf_svm.predict(module_test.description)
print(np.mean(predicted_svm == module_test.label))

# Set Random Forest Pipeline
text_clf_rf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),
('clf-rf', RandomForestClassifier())])

text_clf_rf = text_clf_rf.fit(module_train.description, module_train.label)
predicted_rf = text_clf_rf.predict(module_test.description)
print(np.mean(predicted_rf == module_test.label))