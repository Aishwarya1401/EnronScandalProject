#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")
import pandas as pd
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt


Features = ['poi', 'email_address', 'salary', 'to_messages', 'deferral_payments', 
                'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 
                'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 
                'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 
                'director_fees', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']

FinancialFeatures = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                    'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                    'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 
                    'director_fees']
EmailFeatures = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                  'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

df_enron = pd.DataFrame.from_dict(data_dict, orient='index')
df_enron = df_enron.replace('NaN', np.nan)

df_enron[FinancialFeatures] = df_enron[FinancialFeatures].fillna(0)

#there are many ways to impute the unfilled data , found median to be the best.

df_enron[EmailFeatures] = df_enron[EmailFeatures].fillna(df_enron[EmailFeatures].median())


# Removing outlier Total and entries whose to and from message are less
df_enron = df_enron.drop('TOTAL')
                                 

### Task 3: Create new feature(s)
### Adding the feature to the dataset
df_enron['fraction_of_messages_to_poi'] = df_enron.from_this_person_to_poi / df_enron.from_messages
df_enron['fraction_of_messages_from_poi'] = df_enron.from_poi_to_this_person / df_enron.to_messages
 
my_dataset = df_enron.to_dict('index')

temp_features_list = ['poi', 'salary', 'to_messages', 'total_payments','exercised_stock_options',
                      'director_fees','bonus', 'total_stock_value','deferred_income','deferral_payments',
                      'shared_receipt_with_poi','from_messages','shared_receipt_with_poi',
                      'fraction_of_messages_to_poi',
                      'fraction_of_messages_from_poi','loan_advances']
#Feature Selection using SelectkBest I found exercised_stock_options,total_stock_value, bonus,salary,
#fraction_of_messages_to_poi,deferred_income,total_payments,shared_receipt_with_poi,loan_advances to be the best

     
### Extract features and labels from dataset for local testing


"""
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
features = min_max_scaler.fit_transform(features) 

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

pre_selected_list = []
features_by_importance = []
for k in range(1,15):
    selector = SelectKBest(f_classif, k=k)
    selector = selector.fit(features,labels)

    features_list_wo_poi = [i for i in features_list if i!="poi"] ### features list without poi

    ### Print features chosen by SelectKBest
    selected_list = [features_list_wo_poi[i] for i in range(len(features_list_wo_poi)) if selector.get_support()[i]]

    print "K:", k
    for i in selected_list:
        if i not in pre_selected_list:
            print "\t", i
            features_by_importance.append(i)

    pre_selected_list = selected_list


# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()

clf = DecisionTreeClassifier(criterion='gini', splitter='random', min_samples_split=5)

from sklearn.pipeline import Pipeline
pipe = Pipeline(steps=[('minmaxer', min_max_scaler), ('clf', clf)])

# Setting  up the print format
PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

#Set up the arrays for precision, recall and F1
precision_list = []
recall_list = []
f1_list = []

# Calculate scores for each K value
for i in range(len(features_by_importance)):
    selected_features_list = features_by_importance[:(i+1)]
    selected_features_list.insert(0,'poi')
    selected_data = featureFormat(my_dataset, selected_features_list)

    # Split the data into labels and features
    labels, features = targetFeatureSplit(selected_data)

    cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0

    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )

            features_test.append( features[jj])
            labels_test.append( labels[jj] )


        # fit the classifier using training set, and test on test set
        pipe.fit(features_train, labels_train)
        try:
            print clf.best_params_
            for k,v in clf.best_params_.iteritems():
                if k in best_params_collector:
                    best_params_collector[k].append(v)
                else:
                    best_params_collector[k] = [v]
        except:
            pass

        predictions = pipe.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            # Assign prediction either 0 or 1
            if prediction < .5:
                prediction = 0
            else:
                prediction = 1

            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)

        print selected_features_list[-1]
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    except:
        print "Precision or recall may be undefined due to a lack of true positive predicitons.\n"



#%matplotlib inline
# Set the size of the plot
plt.rcParams["figure.figsize"] = (10,6)

# Set up the x-axis
k_values = range(1,len(recall_list)+1)

# Draw Salary histogram
plt.plot(k_values, precision_list, k_values, recall_list, k_values, f1_list)

x = [1,18]
y = [.3,.3]
plt.plot(x,y)

plt.xlim([1,18])

plt.legend(['precision','recall','f1'])
plt.xlabel("K")
plt.ylabel("Scores")
plt.title("Scores for each K value")
plt.show()
"""


features_list =['poi', 'salary', 'total_payments','exercised_stock_options',
                'bonus', 'total_stock_value','deferred_income',
                'shared_receipt_with_poi',  'from_messages',  'shared_receipt_with_poi',
                'fraction_of_messages_to_poi','loan_advances'
       ]
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, make_scorer
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('Selection',SelectKBest(k=3)),
    ('Pca', PCA(n_components=3,iterated_power='auto',svd_solver='auto',whiten=True)),
    ('clas',DecisionTreeClassifier(random_state =42,min_samples_split=2,splitter='random')),
    ])
    
    
    
parameters = {
        #'scaler__with_centering': (True,False),
        #'vect__max_features': (None, 5000, 10000, 50000),
        #'scaler__with_scaling': (True,False),  # unigrams or bigrams
        #'tfidf__norm': ('l1', 'l2'),
        #'Pca__n_components': ([1,9]),
        #'Pca__whiten': (True, False),
        #'Pca__random_state': ([1,100]),
        #'Pca__svd_solver':('auto', 'full', 'arpack', 'randomized'),
        #'clas__n_neighbors':([1,5])
        #'clas__kernel':('rbf','poly','sigmoid','linear'),
        #'clas__C':([1,100]),
        #'clas__gamma':([0,1])
        #'clas__splitter':('best','random'),
        #'clas__max_features':('log2','sqrt'),
        #'clas__criterion': ('gini', 'entropy'),
        #'random_state':range(1,50),
        #'clas__min_samples_leaf':range(1,5,50),
        #'max_depth': range(1, 10),
        #'n_estimators': range(1,100,10)
        #'clas__min_samples_split':range(2,9)
    
    }
    
# After running GridSearch CV I found the parameters mentioned the brackets of pipeline 
    
gs = GridSearchCV(pipeline, parameters,scoring='f1',cv=StratifiedShuffleSplit(labels,1000,
                                                                              random_state = 42)
                                                                                ,n_jobs=-1)


t0 = time()
gs.fit(features, labels)
print("done in %0.3fs" % (time() - t0))
clf= gs.best_estimator_
     

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

