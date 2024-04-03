#!/usr/bin/env python
# coding: utf-8

# # MLDM Lab week 3: Neural Networks - Perceptron and Multi-Layer Perceptron (MLP)

# <h3> <font color="blue"> Introduction </h3>
# 
# In this lab session, we explore Perceptron and Multi-Layer Perceptron (MLP) using `sklearn`.
# 
# We revisit the Iris and the Breast Cancer datasets from previous lab sessions and train Perceptron and MLP classifiers using these datasets. Please see the information regarding these datasets from previous two lab sessions.

# <h3> <font color="blue"> Lab goals</font> </h3>
# <p> 1.  Learn how to create and use a Perceptron. </p>
# <p> 2.  Learn how to create and use a MLP. </p>
# <p> 3.  Learn how to create ROC curves to evaluate and compare models. </p>

# ## <font color="blue"> Training and evaluating a Perceptron 
# In this experiment we re-visit the Iris dataset, however, instead of loading the dataset from a file or url, we load the dataset directly from the scikit-learn datasets. In this dataset, the third column represents the petal length, and the fourth column the petal width of the flower examples and the classes are already converted to integer labels where 0=Iris-Setosa, 1=Iris-Versicolor, 2=Iris-Virginica.

# In[3]:


from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

print('Class labels:', np.unique(y))


# Preparing the training and test data by splitting the main dataset into 70% training and 30% test stes:

# In[4]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)


# In[5]:


print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))


# The features can be standardised using the `StandardScaler`. See <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">`class sklearn.preprocessing.StandardScaler`</a> for more details.

# In[6]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# ### <font color="blue"> Training a Perceptron 
#     
# We train a Perceptron on the Iris dataset. In the first experiment we use simple and basic parameters. 
#     
# More information about Perceptron parameters can be found from <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html"> `sklearn.linear_model.Perceptron` </a>.

# In[7]:


from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)


# **Note** You can replace `Perceptron(n_iter, ...)` by `Perceptron(max_iter, ...)` in scikit-learn >= 0.19. The `n_iter` parameter is used here deriberately, because some people still use scikit-learn 0.18.

# In[8]:


y_pred = ppn.predict(X_test_std)
print('Misclassified examples: %d' % (y_test != y_pred).sum())


# In[9]:


print('Accuracy(test set): %.3f' % accuracy_score(y_test, y_pred))
print('Accuracy (standardised test set): %.3f' % ppn.score(X_test_std, y_test))


# ## <font color="blue"> Training and evaluating a Multi-Layer Perceptron (MLP)
# In this experiment we re-visit the breast cancer dataset, however, instead of loading the dataset from scikit-learn datasets, we load the dataset from a csv file. Please download (and unzip) the breast cancer dataset from SurreyLearn and copy the file 'breast_cancer_data.csv' into your Jupyter working directory.

# In[10]:


import pandas as pd
breast_cancer = pd.read_csv('breast_cancer_data.csv')
# Dataset Shape
print (breast_cancer.shape)


# In[11]:


# First Few Columns and Head Details
breast_cancer.head()


# <p> We need to specify the features which are used for machine learning. We need to remove the features which should be excluded, e.g. ID. We also need to specify the target feature, i.e. diagnosis class.
#     Feature Engineering is the way of extracting features from data and transforming them into formats that are suitable for Machine Learning algorithms.</p> Here we are going to remove certain unwanted features.

# In[12]:


# Features "id" and "Unnamed: 32" should be removed
feature_names = breast_cancer.columns[2:-1]
X = breast_cancer[feature_names]
# the target feature, i.e. diagnosis class
y = breast_cancer.diagnosis


# <p>The traget feature in this dataset is included as a text value so it should be converted into a numerical value.</p>

# In[13]:


from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
# M -> 1 and B -> 0
y = class_le.fit_transform(breast_cancer.diagnosis.values)


# In[14]:


breast_cancer[feature_names]


# ### <font color="blue"> Training a Multi-Layer Perceptron (MLP)
# We train a Multi-Layer Perceptron (MLP) on the breast cancer dataset. In the first experiment we use simple parameters which we will try to optimise later. 
#     
# More information about MLP parameters can be found from <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html"> `sklearn.neural_network.MLPClassifier` </a>.

# In[15]:


# Training and testing data preperation.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Initializing the classifier
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(random_state=0, activation='logistic', hidden_layer_sizes=(5,), solver='adam', max_iter=100)
clf.fit(X_train, y_train)
clf_predict = clf.predict(X_test)
# Accuracy factors
print('acc for training data: {:.3f}'.format(clf.score(X_train, y_train)))
print('acc for test data: {:.3f}'.format(clf.score(X_test, y_test)))
print('MLP Classification report:\n\n', classification_report(y_test, clf_predict))


# We can improve the accuracy of the MLP by tuning the parameters, e.g. increasing the 'hidden_layer_sizes' or 'max_iter'. Below we increase hidden_layer_sizes from 5 to 10.

# In[16]:


clf = MLPClassifier(random_state=0, activation='logistic', solver='adam', hidden_layer_sizes=(10,), max_iter=100)
clf.fit(X_train, y_train)
clf_predict = clf.predict(X_test)

# Accuracy factors
print('acc for training data: {:.3f}'.format(clf.score(X_train, y_train)))
print('acc for test data: {:.3f}'.format(clf.score(X_test, y_test)))
print('MLP Classification report:\n\n', classification_report(y_test, clf_predict))


# <h3><font color="red">Exercise 1 </font> </h3>
# <p>Repeat the experiment above and try to further improve the accuracy of the MLP by increasing 'max_iter'</p>
# 
# <p>Use the code cell below to write your code for Excercise 1</p>

# In[20]:


# Answer to Exercise 1 
# Grid Search: In order to find the best max_iter value, Grid Search has been implemented to evaluate all possible combinations
# with 5 cross-validation. A code has been run to find the best value of the parameter for cv but not included here since it is 
# not the subject of the experiment. In this section with the best max_iteration value highest accuracy for the model is found 
# for 2 different hidden layer values: 5 and 10. Also for a fair comparison i compared the 2 model for max_iter=200

from sklearn.model_selection import GridSearchCV

# Define the parameter grid for grid search
param_grid = {'max_iter': range(100, 1000, 10)}  # Set the range of max_iter values to search between 100 to 1000 with 10 step size

# Define MLPClassifier objects for 5 and 10 hidden layer sizes
clf_5_hidden = MLPClassifier(random_state=0, activation='logistic', solver='adam', hidden_layer_sizes=(5,))
clf_10_hidden = MLPClassifier(random_state=0, activation='logistic', solver='adam', hidden_layer_sizes=(10,))

# Do grid search for 5 hidden layers
grid_search_5_hidden = GridSearchCV(clf_5_hidden, param_grid, cv=5)
grid_search_5_hidden.fit(X_train, y_train)

# Do grid search for 10 hidden layers
grid_search_10_hidden = GridSearchCV(clf_10_hidden, param_grid, cv=5)
grid_search_10_hidden.fit(X_train, y_train)

# Get the best parameter values for 5 and 10 hidden layers
best_max_iter_5_hidden = grid_search_5_hidden.best_params_['max_iter']
best_max_iter_10_hidden = grid_search_10_hidden.best_params_['max_iter']

print("Best max_iter for 5 hidden layers:", best_max_iter_5_hidden)
print("Best max_iter for 10 hidden layers:", best_max_iter_10_hidden)

# Train both models with the best max_iter value
best_clf_5_hidden = MLPClassifier(random_state=0, activation='logistic', solver='adam', hidden_layer_sizes=(5,), max_iter=best_max_iter_5_hidden)
best_clf_5_hidden.fit(X_train, y_train)

best_clf_10_hidden = MLPClassifier(random_state=0, activation='logistic', solver='adam', hidden_layer_sizes=(10,), max_iter=best_max_iter_10_hidden)
best_clf_10_hidden.fit(X_train, y_train)

# Predict using the best models
clf_predict_5_hidden = best_clf_5_hidden.predict(X_test)
clf_predict_10_hidden = best_clf_10_hidden.predict(X_test)

# Check if max_iter=200 is the best value for 5 hidden layers
if 200 in param_grid['max_iter']:
    clf_5_hidden_200 = MLPClassifier(random_state=0, activation='logistic', solver='adam', hidden_layer_sizes=(5,), max_iter=200)
    clf_5_hidden_200.fit(X_train, y_train)
    clf_predict_5_hidden_200 = clf_5_hidden_200.predict(X_test)
    print('Accuracy for 5 hidden layers with max_iter=200 on training data: {:.3f}'.format(clf_5_hidden_200.score(X_train, y_train)))
    print('Accuracy for 5 hidden layers with max_iter=200 on test data: {:.3f}'.format(clf_5_hidden_200.score(X_test, y_test)))
    print('MLP Classification report for 5 hidden layers with max_iter=200:\n\n', classification_report(y_test, clf_predict_5_hidden_200))

# Check if max_iter=200 is the best value for 10 hidden layers
if 200 in param_grid['max_iter']:
    clf_10_hidden_200 = MLPClassifier(random_state=0, activation='logistic', solver='adam', hidden_layer_sizes=(10,), max_iter=200)
    clf_10_hidden_200.fit(X_train, y_train)
    clf_predict_10_hidden_200 = clf_10_hidden_200.predict(X_test)
    print('Accuracy for 10 hidden layers with max_iter=200 on training data: {:.3f}'.format(clf_10_hidden_200.score(X_train, y_train)))
    print('Accuracy for 10 hidden layers with max_iter=200 on test data: {:.3f}'.format(clf_10_hidden_200.score(X_test, y_test)))
    print('MLP Classification report for 10 hidden layers with max_iter=200:\n\n', classification_report(y_test, clf_predict_10_hidden_200))
    
# Accuracy factors
print('Accuracy for 5 hidden layers on training data with the best max_iter: {:.3f}'.format(best_clf_5_hidden.score(X_train, y_train)))
print('Accuracy for 5 hidden layers on test data: {:.3f}'.format(best_clf_5_hidden.score(X_test, y_test)))
print('MLP Classification report for 5 hidden layers:\n\n', classification_report(y_test, clf_predict_5_hidden))

print('Accuracy for 10 hidden layers on training datawith the best max_iter: {:.3f}'.format(best_clf_10_hidden.score(X_train, y_train)))
print('Accuracy for 10 hidden layers on test data: {:.3f}'.format(best_clf_10_hidden.score(X_test, y_test)))
print('MLP Classification report for 10 hidden layers:\n\n', classification_report(y_test, clf_predict_10_hidden))

# On the comparison of the models, for the max_iter=200 and max_iter=100 for hidden layer=5  
# For 5 hidden layers and max_iter=200: Training Accuracy: 0.906 Test Accuracy: 0.881
# For 10 hidden layers and max_iter=100: Training Accuracy: 0.627 Test Accuracy: 0.629
# In the first model which is pre-coded at the beginning of the lab sheet, with a max_iter value of 100 and 5 hidden layers, 
# the model achieved an accuracy of 0.627 on the training data and an accuracy of 0.629 on the test data and the model struggled 
# to classify the minority class (class 1) with a precision, recall, and F1-score of 0.00, showing poor performance.
# The overall performance of the model was modest, with a weighted average F1-score of 0.49.
# The second model with 5 hidden layers using a max_iter value of 200. performed significantly better, achieving an accuracy 
# of 0.906 on the training data and an accuracy of 0.881 on the test data.
# The classification report shows improved precision, recall, and F1-scores for both classes, 
# indicating better performance in classifying instances from both classes. The overall performance of this model was higher, 
# with a weighted average F1-score of 0.88.
# Comparing the two scenarios, it is seen that increasing the max_iter value from 100 to 200 and using 5 hidden layers 
# significantly improved the performance of the model. The second model achieved higher accuracies and better classification 
# results, showing the importance of model optimization in achieving better predictive capabilities.

# On the comparison of the models, for the max_iter=200 and max_iter=100 for hidden layer=10  
# In the first model, where the model had 10 hidden layers and a max_iter value of 100, the accuracy on the training data 
# was 0.904 and on the test data was 0.909. The classification report shows a precision, recall, and F1-score of 0.89 for 
#class 0, shpwing good performance in identifying negative instances. For class 1, the precision, recall, and F1-score were
# 0.95, 0.79, and 0.87, respectively, showing good performance. The overall performance of the model was good, with a weighted 
# average F1-score of 0.91.
# In the second model, the model had the same architecture with 10 hidden layers, but with a higher max_iter value of 200. 
#This model performed slightly worse, achieving an accuracy of 0.925 on the training data and 0.916 on the test data. 
#The classification report shows similar precision, recall, and F1-scores for both classes, indicating consistent performance 
#in classifying instances from both classes. The overall performance of this model was also high, with a weighted average
#F1-score of 0.92.
# Comparing the two scenarios, it appears that increasing the max_iter value from 100 to 200 had a marginal impact on the 
# model's performance. The second model achieved slightly lower accuracies but maintained similar precision, recall, and 
# F1-scores as the first scenario. Both models showed good overall performance, with similar F1-scores of around 0.91-0.92.

# I also wanted to see the best accuracy for the most optimal max_iter:
# For 5 hidden layer the model achieved an accuracy of 0.918 on the training data and 0.909 on the test data. 
# The classification report shows a precision, recall, and F1-score of 0.91
# For 10 hidden layer the model achieved an accuracy of 0.930 on the training data and an accuracy of 0.944 on the test data. 
# The classification report shows high precision, recall, and F1-scores of 0.94.


# Overall, the models show that models with higher iteration models can achieve high accuracy and perform well.


# <h3><font color="red">Exercise 2 </font> </h3>
# <p>Repeat the experiment above using a Perceptron on the breast cancer dataset and compare the accuracy with the results from MLP.</p>
# 
# <p>Use the code cell below to write your code for Excercise 2</p>

# In[32]:


# Answer to Exercise 2
#To find the best parameter a Grid Search is used for the best max_iter value. The experiment is re-experimented with perceptron
# with the values of 100,200 and best max_iter value. 

# Define the parameter grid for grid search
param_grid = {'max_iter': [100, 200,60]}

# Define Perceptron object
perceptron = Perceptron(random_state=0)

# Do grid search
grid_search = GridSearchCV(perceptron, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best max_iter value
best_max_iter = grid_search.best_params_['max_iter']

print("Best max_iter:", best_max_iter)

# Train models for different max_iter values
perceptron_100 = Perceptron(random_state=0, max_iter=100)
perceptron_200 = Perceptron(random_state=0, max_iter=200)
best_perceptron = Perceptron(random_state=0, max_iter=best_max_iter)

perceptron_100.fit(X_train, y_train)
perceptron_200.fit(X_train, y_train)
best_perceptron.fit(X_train, y_train)

# Predict using the models
clf_predict_100 = perceptron_100.predict(X_test)
clf_predict_200 = perceptron_200.predict(X_test)
clf_predict_best = best_perceptron.predict(X_test)

# Calculate accuracy for each max_iter value
accuracy_100 = perceptron_100.score(X_test, y_test)
accuracy_200 = perceptron_200.score(X_test, y_test)
accuracy_best = best_perceptron.score(X_test, y_test)

# Print the accuracy
print('Accuracy with max_iter=100:', accuracy_100)
print('Accuracy with max_iter=200:', accuracy_200)
print('Accuracy with best max_iter from grid search:', accuracy_best)

# Training accuracy
train_accuracy_100 = perceptron_100.score(X_train, y_train)
train_accuracy_200 = perceptron_200.score(X_train, y_train)
train_accuracy_best = best_perceptron.score(X_train, y_train)

print('Training accuracy with max_iter=100:', train_accuracy_100)
print('Training accuracy with max_iter=200:', train_accuracy_200)
print('Training accuracy with best max_iter from grid search:', train_accuracy_best)

# Classification report
print('Classification report with max_iter=100:')
print(classification_report(y_test, clf_predict_100))

print('Classification report with max_iter=200:')
print(classification_report(y_test, clf_predict_200))

print('Classification report with best max_iter from grid search:')
print(classification_report(y_test, clf_predict_best))

# It is seen that the Perceptron models with different max_iter values (100 and 200) and the best max_iter obtained from thr
# grid search showed similar performance. They achieve an accuracy of approximately 0.853.
# Overall, the models perform similar performance between different max_iter values.

#On the comparison with MLP, it is seen that for every max_iter and hidden layer, MLP and Perceptron has different comparisons. 
# For example for the MLP of 5 hidden layer and 100 max_iter the accuracy is 0.63 overall so it is recommened to use Perceptron
# but for a model which has 10 hidden layers and 560 iterations the accuracy is 0.94. In this case it is recommend to use MLP.


# <h3> <font color="blue"> Receiver Operating Characteristic (ROC) curves </font> </h3>
# 
# ROC curves are one of the most useful ways to evaluate and compare learning algorithms. Below we compare the performance of two MLP classifiers (i.e. MLP1 and MLP2) with different parameters

# In[23]:


# roc curve and auc
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# generate a random prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# fit first model(MLP 1)
clf1 = MLPClassifier(random_state=0, activation='logistic', hidden_layer_sizes=(5,), max_iter=100)
clf1.fit(X_train, y_train)
# fit second model(MLP 2)
clf2 = MLPClassifier(random_state=0, activation='logistic', hidden_layer_sizes=(10,), max_iter=100)
clf2.fit(X_train, y_train)


# predict probabilities for different models
lr_probs1 = clf1.predict_proba(X_test)
lr_probs2 = clf2.predict_proba(X_test)

# keep probabilities for the positive outcome only
lr_probs1 = lr_probs1[:, 1]
lr_probs2 = lr_probs2[:, 1]

# calculate accuracy score for random prediction model
ns_auc = roc_auc_score(y_test, ns_probs)

# calculate accuracy score different MLP models
lr_auc1 = roc_auc_score(y_test, lr_probs1)
lr_auc2 = roc_auc_score(y_test, lr_probs2)

# summarize scores
print('Baseline (random guess): ROC AUC=%.3f' % (ns_auc))
print('MLP 1 (hidden layer sizes=5): ROC AUC=%.3f' % (lr_auc1))
print('MLP 2 (hidden layer sizes=10): ROC AUC=%.3f' % (lr_auc2))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr1, lr_tpr1, _ = roc_curve(y_test, lr_probs1)
lr_fpr2, lr_tpr2, _ = roc_curve(y_test, lr_probs2)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Baseline (random guess)')
pyplot.plot(lr_fpr1, lr_tpr1, marker='.', label='MLP 1 (hidden layer sizes=5)')
pyplot.plot(lr_fpr2, lr_tpr2, marker='.', label='MLP 2 (hidden layer sizes=10)')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


# <h3><font color="red">Exercise 3 </font> </h3>
# <p>Repeat the experiment above and define a new MLP classifier (named MLP 3) try to further improve the accuracy of the MLP 3 by changing the parameters, e.g. increasing 'hidden_layer_sizes' and 'max_iter'. What is the max accuracy you can get by tuning the parameters ?</p>
# 
# <p>Use the code cell below to write your code for Excercise 3</p>

# In[27]:


# Experiment ie re-experimented with 3 different hidden layer sizes which are 5,10,15 and a Grid Search applied in order to 
#find the best max_iteration.

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Predict the majority class
majority_class = max(set(y_train), key=list(y_train).count)
ns_probs = [0 if cls == majority_class else 1 for cls in y_test]

# Define the limited parameter ranges
hidden_layer_sizes_range = [(5,), (10,), (15,)]  # Limited range for hidden_layer_sizes
max_iter_range = range(100, 1000, 10)  # Limited range for max_iter

# Create classifier as MLP3 named
MLP3 = MLPClassifier(random_state=0, activation='logistic')

# Define the parameter grid
param_grid = {
    'hidden_layer_sizes': hidden_layer_sizes_range,
    'max_iter': max_iter_range
}

# the GridSearchCV 
grid_search = GridSearchCV(MLP3, param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

# Get the best parameters and best accuracy
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters: ", best_params)
print("Best Accuracy: ", best_score)

# Fit the best model using the best parameters
best_MLP3 = MLPClassifier(random_state=0, activation='logistic', **best_params)
best_MLP3.fit(X_train, y_train)

# Predict probabilities for the best model
lr_probs = best_MLP3.predict_proba(X_test)[:, 1]

# Calculate AUC score for the best model
lr_auc = roc_auc_score(y_test, lr_probs)

# Summarize scores
print('Baseline (majority class): ROC AUC=%.3f' % roc_auc_score(y_test, ns_probs))
print('Best MLP3: ROC AUC=%.3f' % lr_auc)

# Calculate ROC curve for the best model
fpr, tpr, _ = roc_curve(y_test, lr_probs)

# Plot the ROC curve for the best model
pyplot.plot(fpr, tpr, marker='.', label='Best MLP3')
pyplot.plot([0, 1], [0, 1], linestyle='--', label='Baseline (majority class)')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()

# The best parameters found by the model for the best accuracy is 15 hidden layers and 560 max_iterations. Also, the best
# accuracy is 0.9412038303693571 and the ROC AUC equals to 0.978 with these parameters. On the comparison with the model above 
# the area is 0.022 bigger.
#


# <h3><font color="red">Exercise 4 </font> </h3>
# <p>Repeat the experiment in Exercise 3 but instead of a new MLP classifier define a Perceptron (named P 1) on the same dataset (i.e. breast cancer) and compare the result with MLP 1 and MLP 2. What do you think might be the reason for the difference between the performance of the MLP classifiers and the Perceptron on this dataset? Try to chnage the Perceptron parameters (reduce the learning rate, Eta) and repeat the experiment. </p>
# 
# <p>Use the code cell below to write your code for Exercise 4</p>

# In[37]:


# Answer to Exercise 4 
# Hint: Unlike MLP, Perceptron doesn't have 'predict_proba' but you can implement it using CalibratedClassifierCV as shown below
#
# ppn = Perceptron(eta0=0.01, random_state=0, max_iter=100)
# ppn = CalibratedClassifierCV(ppn)
#
# and then you can call ppn.predict_proba
########


from sklearn.calibration import CalibratedClassifierCV
eta_values = [0.0001, 0.001, 0.01, 0.1, 1.0]

# Iterate over the eta values
for eta in eta_values:
    # Create a Perceptron model with the current eta value
    ppn = Perceptron(eta0=eta, random_state=0, max_iter=100) #max_iter stayed the same since MLP1 AND MLP2 has max_iter=100
    P1 = CalibratedClassifierCV(ppn)

    # Fit the Perceptron model on the training data
    P1.fit(X_train, y_train)

    # Predict probabilities for the positive outcome using the Perceptron model
    P1_probs = P1.predict_proba(X_test)[:, 1]

    # Calculate the ROC AUC score for the Perceptron model
    P1_auc = roc_auc_score(y_test, P1_probs)
    print(f'Perceptron (eta={eta}): ROC AUC = {P1_auc:.3f}')
    P1_fpr, P1_tpr, _ = roc_curve(y_test, P1_probs)
    pyplot.plot(P1_fpr, P1_tpr, marker='.', label=f'Perceptron (eta={eta})')

# Axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# Show the legend
pyplot.legend()

# Show the plot
pyplot.show()

# Baseline (random guess): ROC AUC=0.500
# MLP 1 (hidden layer sizes=5): ROC AUC=0.842
# MLP 2 (hidden layer sizes=10): ROC AUC=0.956
# p1 Perceptron (eta=0.0001): ROC AUC = 0.932
# p1 Perceptron (eta=0.001): ROC AUC = 0.932
# p1 Perceptron (eta=0.01): ROC AUC = 0.908
# p1 Perceptron (eta=0.1): ROC AUC = 0.500
# p1 Perceptron (eta=1.0): ROC AUC = 0.500

# On the comparison of MLP1, MLP2, AND P1 with different eta's it is seen that MLP2 has the highest ROC AUC score. 
# Since MLP2 has 10 hidden layer and MLP1 has 5, it is understandable that higher hidden layer effects the model positvely and 
# boosts to score.  
# MLP1 is also performed good on the comparison of P1 which eta=0.1 and 1.0 but when the eta is lower than 0.01 
# Perceptron outperformed  MLP1. It is seen that P1 with eta values of 0.1 and 1.0 perform poorly with ROC AUC scores 
# of 0.5 and showed that the values are same with the Base model. For the Learning rate it is seen that when the rate gets higher 
# the model's accuracy starts to get lower. The increase on the step size led to high convergence and effected the model poorly.
# For the eta = 0.001 model performs the best amongs the other learning rate values. It is also observed that at one point the decrease
# on the learning rate does not increase the performance and ROC-AUC score. 

# Overall, with two model has different hyperparameters, different model complexity, non-linearity and linearty(perceptron) 
# it is hard to recommend one of the models. According to the model, problem and optimal hyperparameter choice
#  both of the models is recommendable. 




# <h3><font color="red">Save your notebook after completing the exercises and submit it to SurreyLearn (Assessments -> Assignments -> Lab Exercises - Week 3) as a python notebook file in ipynb formt. </h3>
# <h3><font color="red">Deadline: 4:00pm Thursday 29 Feb  </h3> 
