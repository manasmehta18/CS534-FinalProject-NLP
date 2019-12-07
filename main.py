# Runs classification algorithms on the data. Classifier parameters are optimized by the optimization scripts and best
# performing parameters are used in here.

# Import:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from data_import import Modules
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Get data:
# module_train = Modules(subset='train', shuffle=True)
module_all = Modules(subset='all', shuffle=False)

# Cast into a list
module_input = [module_all.description,module_all.label]

# Generate DataFrame
module_pd = (pd.DataFrame(module_input))
module_pd_t = module_pd.transpose()

# Initialize recording lists:
acc_MultiNB = list()
acc_GaussNB = list()
acc_SGDClassifier = list()
acc_SVC = list()
acc_rf = list()
i_rec = list()

# Loop for randomization (100 times):
for i in range(0,100):

    # Get training set (0.8 of data)
    module_train = module_pd_t.sample(frac = 0.8, random_state=i)

    # Get testing set
    module_test = module_pd_t.drop(module_train.index)

    # Seperete X and Y
    X_module_train = module_train.iloc[:,:-1].values
    Y_module_train = module_train.iloc[:,-1].values
    X_module_test = module_test.iloc[:,:-1].values
    Y_module_test = module_test.iloc[:,-1].values

    # Cast into lists
    x_train_lst = X_module_train.flatten().astype(str).tolist()
    y_train_lst = Y_module_train.flatten().tolist()
    x_test_lst = X_module_test.flatten().astype(str).tolist()
    y_test_lst = Y_module_test.flatten().tolist()

    # Set MultinomialNB Pipeline:
    text_clf_MultiNB = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer(use_idf=False)),
    ('clf-MultiNB', MultinomialNB(alpha=1e-2))])

    # MultinomialNB Training:
    text_clf_MultiNB = text_clf_MultiNB.fit(x_train_lst, y_train_lst)
    
    # MultinomialNB Testing
    predicted_MultiNB = text_clf_MultiNB.predict(x_test_lst)
    acc_MultiNB.append(np.mean(predicted_MultiNB == y_test_lst))

    # Set GaussianNB Pipeline: (Does not work)
    text_clf_GaussNB = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf-GaussNB', GaussianNB())])

    # GaussianNB Training:
    # text_clf_GaussNB = text_clf_GaussNB.fit(x_train_lst, y_train_lst)
    
    # GaussianNB Testing
    # predicted_GaussNB = text_clf_GaussNB.predict(x_test_lst)
    # acc_GaussNB.append(np.mean(predicted_GaussNB == y_test_lst))

    # Set SGDClassifier Pipeline
    text_clf_SGDClassifier = Pipeline([('vect', CountVectorizer(stop_words='english')),('tfidf', TfidfTransformer(use_idf=True)),
    ('clf-SGDClassifier', SGDClassifier(loss='hinge', penalty='l1', alpha=1e-3, n_iter_no_change=5, random_state=42))])

    # SGDClassifier Training:
    text_clf_SGDClassifier = text_clf_SGDClassifier.fit(x_train_lst, y_train_lst)

    # SGDClassifier Testing:
    predicted_SGDClassifier = text_clf_SGDClassifier.predict(x_test_lst)
    acc_SGDClassifier.append(np.mean(predicted_SGDClassifier == y_test_lst))
    
    # Set SVC Pipeline
    text_clf_SVC = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer(use_idf=True)),
    ('clf-SVC', SVC(C=2, kernel='linear'))])

    # SVC Training:
    text_clf_SVC = text_clf_SVC.fit(x_train_lst, y_train_lst)

    # SVC Testing:
    predicted_SVC = text_clf_SVC.predict(x_test_lst)
    acc_SVC.append(np.mean(predicted_SVC == y_test_lst))

    # Set Random Forest Pipeline
    text_clf_rf = Pipeline([('vect', CountVectorizer(stop_words='english')),('tfidf', TfidfTransformer(use_idf=False)),
    ('clf-rf', RandomForestClassifier(n_estimators=50, max_depth=None, bootstrap=False))])
    
    # Random Forest Training:
    text_clf_rf = text_clf_rf.fit(x_train_lst, y_train_lst)

    # Random Forest Testing
    predicted_rf = text_clf_rf.predict(x_test_lst)
    acc_rf.append(np.mean(predicted_rf == y_test_lst))

    i_rec.append(i)


# Results:
np_acc_MultiNB = np.array(acc_MultiNB)
np_acc_GaussNB = np.array(acc_GaussNB)
np_acc_SGDClassifier = np.array(acc_SGDClassifier)
np_acc_SVC = np.array(acc_SVC)
np_acc_rf = np.array(acc_rf)

# Average and standard deviation for accuracy:
print("Average accuracy for Multinomial Naive Bayes: {}".format(np.mean(acc_MultiNB)))
print("Standard deviation for Multinomial Naive Bayes: {}".format(np.std(acc_MultiNB)))
# print("Average accuracy for Gaussian Naive Bayes: {}".format(np.mean(acc_GaussNB)))
# print("Standard deviation for Gaussian Naive Bayes: {}".format(np.std(acc_GaussNB)))
print("Average accuracy for SGDClassifier: {}".format(np.mean(acc_SGDClassifier)))
print("Standard deviation for SGDClassifier: {}".format(np.std(acc_SGDClassifier)))
print("Average accuracy for Support Vector Classification: {}".format(np.mean(acc_SVC)))
print("Standard deviation for Support Vector Classification: {}".format(np.std(acc_SVC)))
print("Average accuracy for Random Forest: {}".format(np.mean(np_acc_rf)))
print("Standard deviation for Random Forest: {}".format(np.std(np_acc_rf)))

# Plots:
plt.plot(i_rec,acc_MultiNB, label='MultiNB')
# plt.plot(i_rec,acc_GaussNB, label='GaussNB')
plt.plot(i_rec,acc_SGDClassifier, label='SGDClassifier')
plt.plot(i_rec,acc_SVC, label='SVC')
plt.plot(i_rec,acc_rf, label='RF')
plt.legend()
plt.show()