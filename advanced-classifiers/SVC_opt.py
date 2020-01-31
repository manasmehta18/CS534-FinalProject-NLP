# Import:
import csv
from sklearn.svm import SVC
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

x = list()
y = list()


with open("../word_embedding/vectorized_data.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        vector = row[:-1]
        y.append(float(row[-1]))
        for i in range(0,len(row)-1):
            vector[i] = (float(row[i]))
        x.append(vector)

 # Cast into lists
x_train_lst = x[0:int(0.8*len(x))]
y_train_lst = y[0:int(0.8*len(y))]

# Set SGDClassifier Pipeline:
text_clf_SVC = Pipeline([('clf-SVC',SVC())])

# Define optimization parameters:
parameters = {'clf-SVC__C': [0.1, 1, 2, 5],'clf-SVC__kernel':['rbf', 'linear']}

# Conduct GridSearch:
gs_clf = GridSearchCV(text_clf_SVC, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(x_train_lst, y_train_lst)
print(gs_clf.best_score_)
print(gs_clf.best_params_)