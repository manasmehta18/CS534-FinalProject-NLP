# Import:
import csv
import random
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# function to compute f-measure:
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


# Initialize lists for vectors and labels
x = list()
y = list()

# Reading vectors and labels from csv file
with open("../word_embedding/vectorized_data.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        vector = row[:-1]
        y.append(float(row[-1]))
        for i in range(0,len(row)-1):
            vector[i] = (float(row[i]))
        x.append(vector)

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
    
    # Shuffle vectors and labels with the same seed
    random.Random(i).shuffle(x)
    random.Random(i).shuffle(y)

    # Cast into lists (seperating the training and testing by 0.8-0.2)
    x_train_lst = x[0:int(0.8*len(x))]
    y_train_lst = y[0:int(0.8*len(y))]
    x_test_lst = x[int(0.8*len(x)):len(x)]
    y_test_lst = y[int(0.8*len(y)):len(y)]

    # SGDClassifier:
    clf_SGDClassifier = SGDClassifier(alpha=1e-4,loss='hinge',penalty='l2')
    clf_SGDClassifier.fit(x_train_lst,y_train_lst)

    predicted_SGDClassifier = clf_SGDClassifier.predict(x_test_lst)
    acc_SGDClassifier.append(np.mean(predicted_SGDClassifier == y_test_lst))
    f_SGDClassifier.append(f_measure(y_test_lst,predicted_SGDClassifier))

    # SVC:
    clf_SVC = SVC(C=5,kernel='linear')
    clf_SVC.fit(x_train_lst,y_train_lst)

    predicted_SVC = clf_SVC.predict(x_test_lst)
    acc_SVC.append(np.mean(predicted_SVC == y_test_lst))
    f_SVC.append(f_measure(y_test_lst,predicted_SVC))

    # Random Forest:
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=None, bootstrap=False)
    clf_rf.fit(x_train_lst,y_train_lst)

    predicted_rf = clf_rf.predict(x_test_lst)
    acc_rf.append(np.mean(predicted_rf == y_test_lst))
    f_rf.append(f_measure(y_test_lst,predicted_rf))

    i_rec.append(i)


# f-measure results:
np_f_SGDClassifier = np.array(f_SGDClassifier)
np_f_SVC = np.array(f_SVC)
np_f_rf = np.array(f_rf)
print("Average f-measure for SGDClassifier: {}".format(np.mean(np_f_SGDClassifier)))
print("Standard deviation of f-measure for SGDClassifier: {}".format(np.std(np_f_SGDClassifier)))
print("Average f-measure for SVC: {}".format(np.mean(np_f_SVC)))
print("Standard deviation of f-measure for SVC: {}".format(np.std(np_f_SVC)))
print("Average f-measure for Random Forest: {}".format(np.mean(np_f_rf)))
print("Standard deviation of f-measure for Random Forest: {}".format(np.std(np_f_rf)))

# Plots:
matplotlib.rcParams['axes.linewidth'] = 3
plt.figure(1)
plt.plot(i_rec,f_SGDClassifier, 'b', label='SGDClassifier', linewidth=2)
plt.title("f-measure: SGDClassifier")
plt.xlabel('Seed')
plt.ylabel('f-measure')
plt.figure(2)
plt.plot(i_rec,f_SVC, 'b', label='SVC', linewidth=2)
plt.title("f-measure: SVC")
plt.xlabel('Seed')
plt.ylabel('f-measure')
plt.figure(3)
plt.plot(i_rec,f_rf, 'b', label='RF', linewidth=2)
plt.title("f-measure: RF")
plt.xlabel('Seed')
plt.ylabel('f-measure')
plt.show()