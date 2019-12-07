# Import:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
import numpy as np
from data_import import Modules
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Get data:
module_all = Modules(subset='all', shuffle=False)

# Cast into a list
module_input = [module_all.description,module_all.label]

# Generate DataFrame
module_pd = (pd.DataFrame(module_input))
module_pd_t = module_pd.transpose()

# Get training set (0.8 of data)
module_train = module_pd_t.sample(frac = 0.8, random_state=1)

# Get testing set
module_test = module_pd_t.drop(module_train.index)

# Seperete X and Y
X_module_train = module_train.iloc[:,:-1].values
Y_module_train = module_train.iloc[:,-1].values

# Cast into lists
x_train_lst = X_module_train.flatten().astype(str).tolist()
y_train_lst = Y_module_train.flatten().tolist()

# Set SGDClassifier Pipeline:
text_clf_SGDClassifier = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),
('clf-SGDClassifier', SGDClassifier())])

# Define optimization parameters:
parameters = {'vect__stop_words': ['english', None],'vect__ngram_range': [(1, 1), (1, 2)],
'tfidf__norm':['l1', 'l2'], 'tfidf__use_idf': (True, False),
'clf-SGDClassifier__alpha': (1e-2, 1e-3),'clf-SGDClassifier__loss':['hinge', 'log'],'clf-SGDClassifier__penalty':['l2', 'l1', 'elasticnet']}

# Conduct GridSearch:
gs_clf = GridSearchCV(text_clf_SGDClassifier, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(x_train_lst, y_train_lst)
print(gs_clf.best_score_)
print(gs_clf.best_params_)