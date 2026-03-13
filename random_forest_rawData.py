# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 16:18:21 2021

@author: panka
"""
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_curve, auc, accuracy_score

#from ipypublish import nb_setup
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import random

from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek

df = pd.read_csv('timeseries_data.csv')

df = df.dropna(axis=0, how='any')

df = df[df.location != 'HMU']

# define the list of the desired columns for the final dataframe
columns = ['contestant_class',
                 'contestant',
                 'location',
                 'track',
                 'event_class',
                 'time_s',
                 'distance',
                 'cumulative_distance',
                 'speed_mps',
                 'acceleration',
                 'accelerator',
                 'gear',
                 'brake',
                 'steering',
                 'headway_s',
                 'lane_offset_m'
                 ]

# define the list of datasets that will be used to generate the final dataframe
df = df[[c for c in columns]]
# create the final dataframe 
df = df.loc[(df['speed_mps'] != -0.0) & (df['cumulative_distance'] != 0)]


# + rural
# df1 = pd.read_csv('timeseries_data.csv')
# df2 = pd.read_csv('df_ntu_rural.csv')

# df1 = df1.dropna(axis=0, how='any')
# df2 = df2.dropna(axis=0, how='any')

# df1 = df1[df1.location != 'HMU']

# # define the list of the desired columns for the final dataframe
# columns = ['contestant_class',
#                   'contestant',
#                   'location',
#                   'track',
#                   'event_class',
#                   'time_s',
#                   'distance',
#                   'cumulative_distance',
#                   'speed_mps',
#                   'acceleration',
#                   'accelerator',
#                   'gear',
#                   'brake',
#                   'steering',
#                   'headway_s',
#                   'lane_offset_m'
#                   ]

# # define the list of datasets that will be used to generate the final dataframe
# dataf = [df1[[c for c in columns]],
#           df2[[c for c in columns]]
#                     ]
# # create the final dataframe 
# df = pd.concat(dataf, ignore_index=True)
# df = df.loc[(df['speed_mps'] != -0.0) & (df['cumulative_distance'] != 0)]


#####encoding
def encode_and_bind(original_dataframe, feature_to_encode):
    """
    takes as input a dataframe and a feature, applies one-hot encoding to the
    feature, drops it from the original dataframe, appends the encoded columns
    and returns the resulting dataframe
    """
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    result_dataframe = pd.concat([original_dataframe, dummies], axis=1)
    # result_dataframe = result_dataframe.drop([feature_to_encode], axis=1)
    
    return result_dataframe

# define the list of features to be encoded
features_to_encode = ['event_class']

# apply one-hot encoding
for feature in features_to_encode:
    df = encode_and_bind(df, feature)
    
    
df_dict = {g: d for g, d in df.groupby('contestant')}
#dictu = {key: df.loc[value] for key, value in df2.groupby("contestant").groups.items()}



keys = list(df_dict.keys())
random.shuffle(keys)

suffled_timeseries_data = dict()
for key in keys:
    suffled_timeseries_data.update({key: df_dict[key]})

for df in suffled_timeseries_data.keys():
    # drop columns
    suffled_timeseries_data[df] = suffled_timeseries_data[df].drop([
                                                                    #'contestant_class',
                                                                    'location',
                                                                    'track',
                                                                    'event_class', 'time_s'],
                                                                   axis=1
                                                                   )    
    
    
    

# define the y variable we want to predict
Y_col='contestant_class'

# define test size as 40% of the dataframe
test_size = int(len(suffled_timeseries_data.keys()) * 0.3)

# split data to test and train
test_set = list(suffled_timeseries_data.keys())[:test_size]
train_set = list(suffled_timeseries_data.keys())[test_size:]
 
train_df = pd.DataFrame()

for train in train_set:
    train_df = pd.concat([train_df,suffled_timeseries_data[train]])
    
    
IDcol = 'contestant'
target = 'contestant_class'

predictors = [x for x in train_df.columns if x not in [target, IDcol]]
train_df['contestant_class'] = train_df['contestant_class'].map(lambda x: 1 if x == 'Patient' else 0)
train_df['contestant_class'] = train_df['contestant_class'].astype(int)
train_df['contestant_class'].value_counts()



# define the undersampling method
sample = SMOTEENN()
# transform the dataset
x_train,  y_train = sample.fit_resample(train_df[predictors],  train_df['contestant_class'])

y_train.value_counts()

# =============================================================================
# from imblearn.ensemble import BalancedRandomForestClassifier
# 
# clf = BalancedRandomForestClassifier(n_estimators = 250, max_depth= 10, bootstrap=True, class_weight='balanced_subsample',\
#                              sampling_strategy = 1.0, criterion= 'entropy' , n_jobs = -1)
# clf = clf.fit(x_train,y_train)
# 
# =============================================================================

clf = RandomForestClassifier(n_estimators=250, max_depth= None, bootstrap=False, class_weight='balanced_subsample',
                             criterion= 'gini', n_jobs = -1)
clf = clf.fit(x_train,y_train)

#Predict training set
dtrain_predictions = clf.predict(x_train)
dtrain_predprob = clf.predict_proba(x_train)[:,1]

#Print model report
print ("\nModel Report")
print ("Accuracy : %.4g" % metrics.accuracy_score(y_train.values, dtrain_predictions))
print ("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, dtrain_predprob))
# Model Report
# Accuracy : 1
# AUC Score (Train): 1.000000

test_df = pd.DataFrame()

for test in test_set:
    test_df = test_df.append(suffled_timeseries_data[test])
    
#predictors = [x for x in test_df.columns if x not in [target, IDcol]]

test_df['contestant_class'] = test_df['contestant_class'].map(lambda x: 1 if x == 'Patient' else 0)
test_df['contestant_class'] = test_df['contestant_class'].astype(int)

dtest_predictions = clf.predict(test_df[predictors])
test_df['predicted'] = clf.predict_proba(test_df[predictors])[:,1]


contestant_score = test_df.groupby(IDcol, as_index=False)['predicted', 'contestant_class'].mean()
contestant_score['Y_pred_mapped'] = np.where(contestant_score['predicted'] >= 0.5, 1, 0)
contestant_score['accuracy'] = np.where(contestant_score['Y_pred_mapped'] == contestant_score['contestant_class'], 1, 0)



print ('AUC Score (Test): %f' % metrics.roc_auc_score(test_df['contestant_class'], test_df['predicted']))
# AUC Score (Test): 0.556988 staro
print ('AUC Score (Test): %f' % metrics.roc_auc_score(contestant_score['contestant_class'], contestant_score['predicted']))
# AUC Score (Test): 0.685714 grouped




################         tree      ##########################################


###############      roc curve    ###########################


fpr, tpr, _ = metrics.roc_curve(contestant_score['contestant_class'], contestant_score['predicted'])

auc_score = metrics.auc(fpr, tpr)

# clear current figure
plt.clf()

plt.title('ROC Curve')
plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(auc_score))

# it's helpful to add a diagonal to indicate where chance 
# scores lie (i.e. just flipping a coin)
plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.legend(loc='lower right')
plt.show()

##############          confusion matrix         ############################


from sklearn.metrics import confusion_matrix
import itertools

matrix = confusion_matrix(contestant_score['contestant_class'], contestant_score['Y_pred_mapped'])

plt.clf()

# place labels at the top
plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position('top')

# plot the matrix per se
plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)

# plot colorbar to the right
plt.colorbar()

fmt = 'd'

# write the number of predictions in each bucket
thresh = matrix.max() / 2.
for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):

    # if background is dark, use a white number, and vice-versa
    plt.text(j, i, format(matrix[i, j], fmt),
         horizontalalignment="center",
         color="white" if matrix[i, j] > thresh else "black")

class_names = ['control','patient']
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.tight_layout()
plt.ylabel('True label',size=14)
plt.xlabel('Predicted label',size=14)
plt.show()