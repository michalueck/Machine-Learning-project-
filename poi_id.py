#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import pickle
sys.path.append("../tools/")
import pandas as pd
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

df = pd.DataFrame.from_dict(data_dict, orient='index')

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 
'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 
'director_fees'] # (Units = USD) 
email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 
'from_this_person_to_poi', 'poi', 
'shared_receipt_with_poi'] # units = number of emails messages; except ‘email_address’, which is a text string

### Task 2: Remove outliers
# convert to numpy.nan
df.replace(to_replace='NaN', value=np.nan, inplace=True)

# Nan value treatment 
df1 = df.replace(to_replace=np.nan, value=0, inplace = True)
df1= df.fillna(0).copy(deep=True)
df1.columns = list(df.columns.values)
# drop row for 'THE TRAVEL AGENCY IN THE PARK'
df2= df1.drop(['THE TRAVEL AGENCY IN THE PARK'])

# drop row for 'LOCKHART EUGENE E'
df3= df2.drop(['LOCKHART EUGENE E'])
# drop column email address
df3= df3.drop(['email_address'], axis = 1)

#correcting the negative entries 
df4 = df3.apply(lambda x: abs(x))

df4['salary'].idxmax()
df4.drop('TOTAL', inplace = True)
df4.plot.scatter(x = 'salary', y = 'total_payments')

poi_pay = pd.DataFrame(df3[df3['poi']==True]['total_payments'])
print (poi_pay)

### Task 3: Create new feature(s)
#conversion to dictionary 
my_dataset = df4.to_dict('index')
# Create new features 'salary_of_total_payment' and 'salary_of_total_stock_value'
df4['salary_of_total_payments'] = 0.0
df4['salary_of_total_stock_value'] = 0.0
df4.loc[df3['total_payments'] != 0.0,'salary_of_total_payments'] = df3['salary'] / df3['total_payments'] * 100
df4.loc[df3['total_stock_value'] != 0.0,'salary_of_total_stock_value'] = df3['salary'] / df3['total_stock_value'] * 100

df4.loc[df4['from_this_person_to_poi'] != 0.0,'f_from'] = df4['from_this_person_to_poi'] / df4['from_messages'] * 100
df4.loc[df4['from_poi_to_this_person'] != 0.0,'f_to'] = df4['from_poi_to_this_person'] / df4['to_messages'] * 100 
df4['f_from'].fillna(0, inplace=True)
df4['f_to'].fillna(0, inplace=True)
df4.head()
 
df5= (df4.nlargest(10,['total_payments'], keep='first'))
x = (df5[:]['salary_of_total_stock_value'])
y= (df5[:]['salary_of_total_payments'])


#complete list of my features before feature selection 
features_list = ['poi','salary', 'to_messages','deferral_payments','total_payments', 'loan_advances','bonus',  'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses','from_poi_to_this_person', 'exercised_stock_options', 'from_messages','other', 'from_this_person_to_poi', 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees', 'salary_of_total_payments', 'salary_of_total_stock_value', 'f_from', 'f_to'] 


### Store to my_dataset for easy export below.
#my_dataset = data_dict
my_dataset=df4.to_dict('index')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
my_dataset=df4.to_dict('index')
features_list =['poi', 'salary', 'bonus', 'from_poi_to_this_person', 'from_this_person_to_poi', 'salary_of_total_payments','f_from', 'f_to'] 


import os
os.chdir("./")
import feature_format
import numpy as np

data = feature_format.featureFormat (my_dataset, features_list, sort_keys = True)
## Extract features and labels from dataset for local testing
labels, features= feature_format.targetFeatureSplit (data)
#which columns shall be used for prediciton 
X =features
y =labels

# Feature importances By SelectKBest

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import KFold
#%matplotlib inline
import matplotlib.pyplot as plt


# Perform feature selection

predictors = features_list
selector = SelectKBest(f_classif, k=5)
selector.fit(df4[predictors],df4["poi"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

indices = np.argsort(scores)[::-1]

sorted_important_features=[]
for i in indices:
    sorted_important_features.append(predictors[i])

# Plot the scores.  
plt.figure()
plt.title("Feature Importances")
plt.bar(range(np.size(predictors)), scores[indices],
       color="seagreen", yerr=np.std([indices]), align="center")
plt.xticks(range(np.size(predictors)), sorted_important_features, rotation='vertical')

plt.xlim([-1, np.size(predictors)])
plt.show()


from sklearn.feature_selection import VarianceThreshold

#Use Variance Threshold to select features with the highest variance

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(df4)
indices = sel.get_support()

discarded = []
for i in range(0, len(indices)):
    if indices[i] == False:
        discarded.append(list(df4.columns.values)[i])
        
  ### deploying feature selection
from sklearn import cross_validation
from time import time
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

### use KFold for split and validate algorithm
from sklearn.cross_validation import KFold
kf=KFold(len(labels),3)
for train_indices, test_indices in kf:
    #make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]

from sklearn.tree import DecisionTreeClassifier

t0 = time()

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
print ('accuracy before tuning ', score)

print ("Decision tree algorithm time:", round(time()-t0, 3), "s")
     
importances = clf.feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]
print ('Feature Ranking: ')
#for i in range(7):
    #print ("{} feature {} ({})".format(i+1,features_list[i+1],importances[indices[i]]))

# Get names of indexes for which column salary of total payments is 0
indexNames = df4[ df4['salary_of_total_payments'] == 0.0 ].index
 
# Delete these row indexes from dataFrame
df4.drop(indexNames , inplace=True)

df_final= df4[['poi', 'salary', 'bonus','expenses', 'from_poi_to_this_person', 'from_this_person_to_poi', 'salary_of_total_payments','f_from', 'f_to']]
df_final.head()

from sklearn.model_selection import train_test_split

# Welche Spalten sollen zur Vorhersage verwendet werden
X = df_final[["salary", "bonus"]].values

y = df_final["poi"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.4) 

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

from helper import plot_classifier

# test data plot
plot_classifier(model, X_train, y_train, proba = False, xlabel = "feature", ylabel = "label")
# test data plot
plot_classifier(model, X_test, y_test, proba = False, xlabel = "feature", ylabel = "label")



# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
X_train = features_train 
X_test = features_test
y_train = labels_train 
y_test = labels_test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
#from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

clfs = {
    "logistic regression": LogisticRegression(),
    "svc": SVC(),
    #"neural network": MLPClassifier(),
    "decision tree": DecisionTreeClassifier(),
    "random forest": RandomForestClassifier(),
    "gradient boosting": GradientBoostingClassifier()
}

for key, clf in clfs.items():
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(key + ": " + str(score))
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.model_selection import StratifiedShuffleSplit
X = np.array(features)
y = np.array(labels)
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
sss.get_n_splits(X, y)

#print(sss)       

for train_index, test_index in sss.split(X, y):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# pipeline.set_params(knn__n_neighbors = 1)
from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(pipeline, param_grid = {
    "knn__n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
})
clf.fit(X_train, y_train)

print(clf.best_params_)

print(clf.best_score_)

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
from time import time
t1 = time()

pred = clf.predict(features_test)
print ("Testing time:", round(time()-t1,3),"s")

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

accuracy = accuracy_score(labels_test,pred)

print("Accuracy is equal to %0.4F %%" % (accuracy*100))
print("Precision is : ",precision_score(pred, labels_test))
print ("Recall is    : ",recall_score(pred, labels_test))
print ("f1-score is  : ",f1_score(pred, labels_test))


# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )
