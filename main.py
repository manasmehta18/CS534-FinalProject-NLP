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

def f_measure(data_class,prediction):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(0,len(data_class)):
        if data_class[i] == 1 and prediction[i] == 1:
             tp += 1
        elif data_class[i] == 0 and prediction[i] == 0:
            tn += 1
        elif data_class[i] == 0 and prediction[i] == 1:
            fp += 1
        elif data_class[i] == 1 and prediction[i] == 0:
            fn += 1
    pre = tp/(tp+fp)
    rec = tp/(tp+fn)
    f_measure = 2*pre*rec/(pre+rec)
    return f_measure

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
f_MultiNB = list()
f_GaussNB = list()
f_SGDClassifier = list()
f_SVC = list()
f_rf = list()
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
    f_MultiNB.append(f_measure(y_test_lst,predicted_MultiNB))

    # Set GaussianNB Pipeline: (Does not work)
    text_clf_GaussNB = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf-GaussNB', GaussianNB())])

    # GaussianNB Training:
    # text_clf_GaussNB = text_clf_GaussNB.fit(x_train_lst, y_train_lst)
    
    # GaussianNB Testing
    # predicted_GaussNB = text_clf_GaussNB.predict(x_test_lst)
    # acc_GaussNB.append(np.mean(predicted_GaussNB == y_test_lst))

    # Set SGDClassifier Pipeline
    text_clf_SGDClassifier = Pipeline([('vect', CountVectorizer(stop_words='english')),('tfidf', TfidfTransformer(use_idf=True)),
    ('clf-SGDClassifier', SGDClassifier(loss='hinge', penalty='l1', alpha=1e-3))])

    # SGDClassifier Training:
    text_clf_SGDClassifier = text_clf_SGDClassifier.fit(x_train_lst, y_train_lst)

    # SGDClassifier Testing:
    predicted_SGDClassifier = text_clf_SGDClassifier.predict(x_test_lst)
    acc_SGDClassifier.append(np.mean(predicted_SGDClassifier == y_test_lst))
    f_SGDClassifier.append(f_measure(y_test_lst,predicted_SGDClassifier))
    
    # Set SVC Pipeline
    text_clf_SVC = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer(use_idf=True)),
    ('clf-SVC', SVC(C=2, kernel='linear'))])

    # SVC Training:
    text_clf_SVC = text_clf_SVC.fit(x_train_lst, y_train_lst)

    # SVC Testing:
    predicted_SVC = text_clf_SVC.predict(x_test_lst)
    acc_SVC.append(np.mean(predicted_SVC == y_test_lst))
    f_SVC.append(f_measure(y_test_lst,predicted_SVC))

    # Set Random Forest Pipeline
    text_clf_rf = Pipeline([('vect', CountVectorizer(stop_words='english')),('tfidf', TfidfTransformer(use_idf=False)),
    ('clf-rf', RandomForestClassifier(n_estimators=50, max_depth=None, bootstrap=False))])
    
    # Random Forest Training:
    text_clf_rf = text_clf_rf.fit(x_train_lst, y_train_lst)

    # Random Forest Testing
    predicted_rf = text_clf_rf.predict(x_test_lst)
    acc_rf.append(np.mean(predicted_rf == y_test_lst))
    f_rf.append(f_measure(y_test_lst,predicted_rf))

    i_rec.append(i)


# Results:
np_acc_MultiNB = np.array(acc_MultiNB)
np_acc_GaussNB = np.array(acc_GaussNB)
np_acc_SGDClassifier = np.array(acc_SGDClassifier)
np_acc_SVC = np.array(acc_SVC)
np_acc_rf = np.array(acc_rf)
np_f_MultiNB = np.array(f_MultiNB)
np_f_GaussNB = np.array(f_GaussNB)
np_f_SGDClassifier = np.array(f_SGDClassifier)
np_f_SVC = np.array(f_SVC)
np_f_rf = np.array(f_rf)

# Average and standard deviation for accuracy:
print("Average accuracy for Multinomial Naive Bayes: {}".format(np.mean(np_acc_MultiNB)))
print("Standard deviation for Multinomial Naive Bayes: {}".format(np.std(np_acc_MultiNB)))
print("Average f-measure for Multinomial Naive Bayes: {}".format(np.mean(np_f_MultiNB)))
print("Standard deviation of f-measure for Multinomial Naive Bayes: {}".format(np.std(np_f_MultiNB)))
# print("Average accuracy for Gaussian Naive Bayes: {}".format(np.mean(acc_GaussNB)))
# print("Standard deviation for Gaussian Naive Bayes: {}".format(np.std(acc_GaussNB)))
print("Average accuracy for SGDClassifier: {}".format(np.mean(np_acc_SGDClassifier)))
print("Standard deviation for SGDClassifier: {}".format(np.std(np_acc_SGDClassifier)))
print("Average f-measure for SGDClassifier: {}".format(np.mean(np_f_SGDClassifier)))
print("Standard deviation of f-measure for SGDClassifier: {}".format(np.std(np_f_SGDClassifier)))
print("Average accuracy for Support Vector Classification: {}".format(np.mean(np_acc_SVC)))
print("Standard deviation for Support Vector Classification: {}".format(np.std(np_acc_SVC)))
print("Average f-measure for SVC: {}".format(np.mean(np_f_SVC)))
print("Standard deviation of f-measure for SVC: {}".format(np.std(np_f_SVC)))
print("Average accuracy for Random Forest: {}".format(np.mean(np_acc_rf)))
print("Standard deviation for Random Forest: {}".format(np.std(np_acc_rf)))
print("Average f-measure for Random Forest: {}".format(np.mean(np_f_rf)))
print("Standard deviation of f-measure for Random Forest: {}".format(np.std(np_f_rf)))

# Plots:
plt.figure(1)
plt.plot(i_rec,acc_MultiNB, label='MultiNB')
plt.title("Accuracy: MultiNB")
# plt.plot(i_rec,acc_GaussNB, label='GaussNB')
plt.figure(2)
plt.plot(i_rec,acc_SGDClassifier, label='SGDClassifier')
plt.title("Accuracy: SGDClassifier")
plt.figure(3)
plt.plot(i_rec,acc_SVC, label='SVC')
plt.title("Accuracy: SVC")
plt.figure(4)
plt.plot(i_rec,acc_rf, label='RF')
plt.title("Accuracy: RF")
plt.figure(5)
plt.plot(i_rec,f_MultiNB, label='MultiNB')
plt.title("f-measure: MultiNB")
# plt.plot(i_rec,acc_GaussNB, label='GaussNB')
plt.figure(6)
plt.plot(i_rec,f_SGDClassifier, label='SGDClassifier')
plt.title("f-measure: SGDClassifier")
plt.figure(7)
plt.plot(i_rec,f_SVC, label='SVC')
plt.title("f-measure: SVC")
plt.figure(8)
plt.plot(i_rec,f_rf, label='RF')
plt.title("f-measure: RF")
plt.show()