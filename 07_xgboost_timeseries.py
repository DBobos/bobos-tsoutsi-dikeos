

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import random
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_curve, auc, accuracy_score
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import roc_auc_score

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


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
                 #'distance',
                 'cumulative_distance',
                 'speed_mps',
                 'acceleration',
                 #'accelerator',
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


# =============================================================================
# # + rural
# df1 = pd.read_csv('timeseries_data.csv')
# df2 = pd.read_csv('df_ntu_rural.csv')
# 
# df1 = df1.dropna(axis=0, how='any')
# df2 = df2.dropna(axis=0, how='any')
# 
# df1 = df1[df1.location != 'HMU']
# 
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
# 
# # define the list of datasets that will be used to generate the final dataframe
# dataf = [df1[[c for c in columns]],
#           df2[[c for c in columns]]
#                     ]
# # create the final dataframe 
# df = pd.concat(dataf, ignore_index=True)
# df = df.loc[(df['speed_mps'] != -0.0) & (df['cumulative_distance'] != 0)]
# =============================================================================

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

# shuffle data
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
                                                                    'event_class'
                                                                    ],
                                                                   axis=1
                                                                   )    



# create dictionaries with control and patient dataframes separately
# to define the composition of test data 
suffled_keys = list(suffled_timeseries_data.keys())
import re
pattern_control = re.compile('^CG.*')
pattern_patient = re.compile('^DP.*')
keys_control = [k for k in suffled_keys if pattern_control.match(k)]
keys_patient = [k for k in suffled_keys if pattern_patient.match(k)]
df_dict_control = {k: suffled_timeseries_data[k] for k in keys_control}
df_dict_patient = {k: suffled_timeseries_data[k] for k in keys_patient}
   

# define the y variable we want to predict
Y_col='contestant_class'

# define test size as 40% of the dataframe
train_size = int(len(suffled_timeseries_data.keys()) * 0.7)
# 40% of the train set will be control group
train_size_control = int(train_size * 0.4)
train_size_patient = train_size - train_size_control

train_set_control = list(df_dict_control.keys())[:train_size_control]
train_set_patient = list(df_dict_patient.keys())[:train_size_patient]

test_set_control = list(df_dict_control.keys())[train_size_control:]
test_set_patient = list(df_dict_patient.keys())[train_size_patient:]

# split data to test and train
train_set = train_set_control + train_set_patient
test_set = test_set_control + test_set_patient
 
train_df = pd.DataFrame()

for train in train_set:
    train_df = train_df.append(suffled_timeseries_data[train])
    
     
IDcol = 'contestant'
target = 'contestant_class'
predictors = [x for x in train_df.columns if x not in [target, IDcol]]


train_df['contestant_class'] = train_df['contestant_class'].map(lambda x: 1 if x == 'Patient' else 0)
train_df['contestant_class'] = train_df['contestant_class'].astype(int)
train_df['contestant_class'].value_counts()






# sampling train dataset
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import RandomUnderSampler

# define the undersampling method
undersample = RandomUnderSampler(sampling_strategy='majority')
# transform the dataset

x_train,  y_train = undersample.fit_resample(train_df[predictors],  train_df['contestant_class'])

y_train.value_counts()

# model   
    
alg = XGBClassifier(
        learning_rate = 1,
        n_estimators = 250,
        max_depth = None,
        #min_child_weight=1,
        gamma=0,
        subsample=1,
        colsample_bytree=0.8,
        colsample_bynode=0.8,
        objective= 'binary:logistic',
        eval_metric = 'auc',
        scale_pos_weight = 1,
        reg_alpha=10,
        nthread=-1,
        seed=120,
        use_label_encoder=False)


#Fit the algorithm on the data
alg.fit(x_train, y_train)

#Predict training set
dtrain_predictions = alg.predict(x_train)
dtrain_predprob = alg.predict_proba(x_train)[:,1]

#Print model report
print ("\nModel Report")
print ("Accuracy : %.4g" % metrics.accuracy_score(y_train.values, dtrain_predictions))
print ("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, dtrain_predprob))
print ()


test_df = pd.DataFrame()

for test in test_set:
    test_df = test_df.append(suffled_timeseries_data[test])
    

predictors_test = [x for x in test_df.columns if x not in [target, IDcol]]




test_df['contestant_class'] = test_df['contestant_class'].map(lambda x: 1 if x == 'Patient' else 0)
test_df['contestant_class'] = test_df['contestant_class'].astype(int)
test_df['contestant_class'].value_counts()
dtest_predictions = alg.predict(test_df[predictors_test])
test_df['predicted'] = alg.predict_proba(test_df[predictors_test])[:,1]

contestant_score = test_df.groupby(IDcol, as_index=False)['predicted', 'contestant_class'].mean()
contestant_score['Y_pred_mapped'] = np.where(contestant_score['predicted'] >= 0.5, 1, 0)
contestant_score['accuracy'] = np.where(contestant_score['Y_pred_mapped'] == contestant_score['contestant_class'], 1, 0)


print ('AUC Score (Test): %f' % metrics.roc_auc_score(test_df['contestant_class'], test_df['predicted']))



####################### grid search for optimization ############################

#Grid seach on subsample and max_features
#Choose all predictors except target & IDcols
param_test1 = {
    'max_depth':range(3,10,2),
    'min_child_weight':range(1,6,2),
    'gamma':[i/10.0 for i in range(0,5)],
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)],
    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}


gsearch7 = GridSearchCV(estimator = alg, param_grid = param_test1, scoring='roc_auc',n_jobs=4, cv=5)



gsearch7.fit(train_df[predictors],train_df[target])


print(sorted(gsearch7.cv_results_.keys()))

################         feature importance      ##############################


feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
feat_scores = pd.Series(alg.get_booster().get_score(importance_type='weight'))
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')


################         tree      ##########################################

import matplotlib.pyplot as plt
import graphviz
fig, ax = plt.subplots(figsize=(50, 50))
xgb.plot_tree(alg, num_trees=4, ax=ax, rankdir='LR')
plt.show()


###############      roc curve    ###########################


# fpr, tpr, _ = metrics.roc_curve(y_train, dtrain_predprob)

# auc_score = metrics.auc(fpr, tpr)

# # clear current figure
# plt.clf()

# plt.title('ROC Curve')
# plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(auc_score))

# # it's helpful to add a diagonal to indicate where chance 
# # scores lie (i.e. just flipping a coin)
# plt.plot([0,1],[0,1],'r--')

# plt.xlim([-0.1,1.1])
# plt.ylim([-0.1,1.1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')

# plt.legend(loc='lower right')
# plt.show()

##############          confusion matrix         ############################


from sklearn.metrics import confusion_matrix
import itertools

matrix = confusion_matrix(test_df['contestant_class'], test_df['predicted'].round())



# Confusion matrixes for tree 
matrix_Train=confusion_matrix(y_train.values, dtrain_predictions.round())
matrix_Test=confusion_matrix(test_df['contestant_class'], test_df['predicted'].round())


#Evaluation for tree 2
print ('\tClassifier Evaluation')

print ('Accuracy Train=', accuracy_score(y_train.values, dtrain_predictions.round(), normalize=True))
print ('Accuracy Test=', accuracy_score(test_df['contestant_class'], test_df['predicted'].round(), normalize=True))

print ('train: Conf matrix Decision Tree')
print (matrix_Train)
print(matrix_Test)


# Measures of performance: Precision, Recall, F1 for tree 
print ('Tree train:  Macro Train Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_train.values,  dtrain_predictions.round(), average='macro'))
print ('Tree test:  Macro Test Precision, recall, f1-score')
print (precision_recall_fscore_support(test_df['contestant_class'], test_df['predicted'].round(), average='macro'))
print ()

print(classification_report(test_df['contestant_class'], test_df['predicted'].round()))


# use feature importance for feature selection
from numpy import loadtxt
from numpy import sort
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel


y_test = test_df['predicted'].round()

test_df = test_df.drop(['predicted','contestant_class', 'contestant'], axis=1)
#using the feature immportance
# Fit model using each importance as a threshold
thresholds = sort(alg.feature_importances_)
for thresh in thresholds:
 # select features using threshold
 selection = SelectFromModel(alg, threshold=thresh, prefit=True)
 select_X_train = selection.transform(x_train)

 selection_model = XGBClassifier(learning_rate = 1,
 n_estimators = 250,
 max_depth = 15,
 #min_child_weight=1,
 gamma=0,
 subsample=1,
 colsample_bytree=0.8,
 colsample_bynode=0.8,
 objective= 'binary:logistic',
 eval_metric = 'auc',
 scale_pos_weight = 1,
 reg_alpha=10,
 nthread=-1,
 use_label_encoder=False)
 selection_model.fit(select_X_train, y_train)

 select_X_test = selection.transform(test_df)
 predictions = selection_model.predict(select_X_test)
 accuracy = accuracy_score(y_test, predictions)
 print("Train Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))








# plt.clf()

# # place labels at the top
# plt.gca().xaxis.tick_top()
# plt.gca().xaxis.set_label_position('top')

# # plot the matrix per se
# plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)

# # plot colorbar to the right
# plt.colorbar()

# fmt = 'd'

# # write the number of predictions in each bucket
# thresh = matrix.max() / 2.
# for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):

#     # if background is dark, use a white number, and vice-versa
#     plt.text(j, i, format(matrix[i, j], fmt),
#          horizontalalignment="center",
#          color="white" if matrix[i, j] > thresh else "black")

# class_names = ['control','patient']
# tick_marks = np.arange(len(class_names))
# plt.xticks(tick_marks, class_names, rotation=45)
# plt.yticks(tick_marks, class_names)
# plt.tight_layout()
# plt.ylabel('True label',size=14)
# plt.xlabel('Predicted label',size=14)
# plt.show()



#modelfit(alg, train_df, test_df, predictors)
######FUNCTION TO FIT AND TEST A MODEL##########

# =============================================================================
# def modelfit(alg, train_df, test_df, predictors):
# # , useTrainCV=True, cv_folds=5, early_stopping_rounds=50
# # =============================================================================
# #     if useTrainCV:
# #         xgb_param = alg.get_xgb_params()
# #         xgtrain = xgb.DMatrix(train_df[predictors].values, label=train_df[target].values)
# #         xgtest = xgb.DMatrix(test_df[predictors].values)
# #         cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,  
# #                           metrics='auc', early_stopping_rounds=early_stopping_rounds)
# #         alg.set_params(n_estimators=cvresult.shape[0])
# # =============================================================================
# 
#     #Fit the algorithm on the data
#     alg.fit(train_df[predictors], train_df['contestant_class'],eval_metric='auc')
#     
#     #Predict training set
#     dtrain_predictions = alg.predict(train_df[predictors])
#     dtrain_predprob = alg.predict_proba(train_df[predictors])[:,1]
#     
#     
#     #Print model report
#     print ("\nModel Report")
#     print ("Accuracy : %.4g" % metrics.accuracy_score(train_df['contestant_class'].values, dtrain_predictions))
#     print ("AUC Score (Train): %f" % metrics.roc_auc_score(train_df['contestant_class'], dtrain_predprob))
# 
#     #Predict on testing data
#     test_df['contestant_class'] = test_df['contestant_class'].map(lambda x: 1 if x == 'Patient' else 0)
#     test_df['contestant_class'] = test_df['contestant_class'].astype(int)
#    
#     dtest_predprob = alg.predict_proba(test_df[predictors])[:,1]
#     column_series = pd.Series(dtest_predprob)
#     test_df = test_df.assign(predicted = column_series.values)
#     print ('AUC Score (Test): %f' % metrics.roc_auc_score(test_df['contestant_class'], test_df['predicted']))
# 
# 
#     contestant_score = test_df.groupby(IDcol, as_index=False)['predicted', 'contestant_class'].mean()
#     contestant_score['Y_pred_mapped'] = np.where(contestant_score['predicted'] >= 0.5, 1, 0)
#     contestant_score['accuracy'] = np.where(contestant_score['Y_pred_mapped'] == contestant_score['contestant_class'], 1, 0)
#         
#     feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
#     feat_imp.plot(kind='bar', title='Feature Importances')
#     plt.ylabel('Feature Importance Score')
#     
#     return contestant_score, test_df
# =============================================================================





# """TIME SERIES HOMOGENEITY TEST"""
# import pyhomogeneity as hg
# import scipy.stats as st
# h_control_set = train_set_control + test_set_control
# h_patient_set = test_set_patient + test_set_patient

# control_df = pd.DataFrame()

# for control in h_control_set:
#     control_df = control_df.append(suffled_timeseries_data[control])
# control_df = control_df.drop(columns=['contestant_class', 'contestant', 'event_class_Before events','event_class_During events','event_class_Inbetween events'])
# patient_df = pd.DataFrame()
# for patient in h_patient_set:
#     patient_df = patient_df.append(suffled_timeseries_data[patient])
# patient_df = patient_df.drop(columns=['contestant_class', 'contestant', 'event_class_Before events','event_class_During events','event_class_Inbetween events'])   


# df_agg = pd.read_csv('aggregated_data.csv')


    


# df_agg = df_agg.drop(columns=['contestant', 'location'])

# c_agg_df = df_agg[df_agg['contestant_class'] == 'Control'] 
# p_agg_df = df_agg[df_agg['contestant_class'] == 'Patient'] 


# c_agg_df = c_agg_df.drop(columns=['contestant_class'])
# p_agg_df = p_agg_df.drop(columns=['contestant_class'])

# from scipy.stats import levene
# from scipy import stats
 
# # define alpha
# alpha = 0.05
# # now we pass the groups and center value
# # from the following
# # ('trimmed mean', 'mean', 'median')
# w_stats, p_value = levene(control_df['speed_mps'], patient_df['speed_mps'],
#                           center='mean')
# print(w_stats, p_value) 
# if p_value > alpha:
#     print("We do not reject the null hypothesis")
# else:
#     print("Reject the Null Hypothesis")
# # (31381.4026745654 0.0
# # Reject the Null Hypothesis)


# w_stats, p_value = levene(control_df['lane_offset_m'], patient_df['lane_offset_m'],
#                           center='mean')
# print(w_stats, p_value) 
# if p_value > alpha:
#     print("We do not reject the null hypothesis")
# else:
#     print("Reject the Null Hypothesis")
# # 6860.804626879147 0.0
# # Reject the Null Hypothesis


