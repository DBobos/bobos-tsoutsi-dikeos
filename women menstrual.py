import pandas as pd
import numpy as np
from factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer import FactorAnalyzer
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from factor_analyzer.factor_analyzer import calculate_kmo
import pingouin as pg
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.tree import export_text
from sklearn import tree
import graphviz
from dtreeviz.trees import dtreeviz 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_curve, auc, accuracy_score
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
# getting the requested variables from the extra data file
file = pd.read_csv('Extra-Data.csv')[['BeckTotalScore', 'SOFAS_score', 'EPWORTHTotalScore','HADSAnxietyTotalScore', 'HADSDepressionTotalScore',\
              'FSSTotalScore', 'SD1', 'SD5','SDTotalScore', 'AISTotalScore', 'AIS_7', 'AIS_8', 'StopBangTotalScore',\
              'RLSTotalScore','Aggression', 'Dislike_of_driving', 'Hazard_monitoring', 'Thrill_seeking', 'Proneness_to_fatigue',\
               'Violations', 'Errors', 'Lapses', 'DrvQ3','DrvQ_8', 'DrQ_11', 'Φύλο', 'Ηλικία', 'BMI', 'never_married_cohab',\
              'Επίπεδο_εκπαίδευσης', 'Επαγγελματική_κατάσταση','Ωράριο', 'Φάρμακα', 'Πιθανά_υπναγωγά', 'MENSTRUAL_CYCLE_11', 'ID']]

file.rename({'ID': 'contestant'}, axis=1, inplace=True)

df = pd.read_csv('aggregated_data.csv')
# removing control contestants that crashed
df = df[(df.contestant != 'CG2')]
df = df[(df.contestant != 'CG5')]
# merging on the contestants
df_extra = pd.merge(file, df, left_on='contestant', right_on='contestant', how='outer')
    

df_extra.contestant_class.replace(('Control', 'Patient'), (0, 1), inplace=True)
# NTU to 0 and HMU to 1
df_extra.location.replace(('NTU', 'HMU'), (0, 1), inplace=True)



df_cycle = df_extra[['MENSTRUAL_CYCLE_11', 'Φύλο', 'contestant_class']]

# get only women
# regression with mentstrual cycle and depression

df_cycle = df_cycle[df_cycle['Φύλο'] == 2]
df_cycle = df_cycle.replace('#NULL!',np.nan)

df_cycle = df_cycle.fillna(0)
df_cycle['MENSTRUAL_CYCLE_11'] = df_cycle['MENSTRUAL_CYCLE_11'].astype(str).astype(int)
df_cycle = df_cycle[df_cycle['MENSTRUAL_CYCLE_11'] < 99]


""" REGRESSION"""

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

X = df_cycle
y = df_cycle['contestant_class']

del df_cycle['contestant_class']


 

#split to test and train data

 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, shuffle=True)

 

#Linear Regression
lreg = linear_model.LinearRegression()
 

lreg.fit(X_train, y_train)



ep_predict = lreg.predict(X_test)

 

#Model evaluation
print("'Coefficients: \n ", lreg.coef_)
print('Intercept: \n',lreg.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, ep_predict))
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, ep_predict))
r2=r2_score(y_test, ep_predict)
print("R2=", r2_score(y_test, ep_predict))
vif=1/(1-r2)
print (vif)






# get only women contestants


df_extra = df_extra[df_extra['Φύλο'] == 2]
df_extra = df_extra.replace('#NULL!',np.nan)

df_extra = df_extra.fillna(0)
df_extra['MENSTRUAL_CYCLE_11'] = df_extra['MENSTRUAL_CYCLE_11'].astype(str).astype(int)
df_extra = df_extra[df_extra['MENSTRUAL_CYCLE_11'] < 99]

""" RANDOM FOREST"""


feature_cols = list(df_extra.columns)
X = df_extra[feature_cols] # Features


y = df_extra['contestant_class']

del df_extra['contestant_class']
del df_extra['location']
del df_extra['contestant']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, shuffle=True)


clfRFT1 = RandomForestClassifier(n_estimators=5, max_depth=None)
clfRFT1.fit(X_train, y_train)
Y_train_pred_RFT1=clfRFT1.predict(X_train)
Y_test_pred_RFT1=clfRFT1.predict(X_test)


# Confusion matrixes for tree 
confMatrixTrainDT=confusion_matrix(y_train, Y_train_pred_RFT1)
confMatrixTestDT=confusion_matrix(y_test, Y_test_pred_RFT1)



print ()

print ('Accuracy Train=', accuracy_score(y_train, Y_train_pred_RFT1, normalize=True))
print ('Accuracy Test=', accuracy_score(y_test, Y_test_pred_RFT1, normalize=True))


confMatrixTrainRFT1=confusion_matrix(y_train, Y_train_pred_RFT1)
confMatrixTestRFT1=confusion_matrix(y_test, Y_test_pred_RFT1)
print ('train: Conf matrix ')
print(confMatrixTrainRFT1)
print ('test: Conf matrix ')
print(confMatrixTestRFT1)

print ('Random Forest Train: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_train, Y_train_pred_RFT1, average='macro'))
print ('Random Forest Test: Macro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, Y_test_pred_RFT1, average='macro'))
print ('\n')




pr_y_test_pred_RFT1=clfRFT1.predict_proba(X_test)
fprRFT1, tprRFT1, thresholdsRFT1 = roc_curve(y_test, pr_y_test_pred_RFT1[:,1])





# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, shuffle=True)







#Decision tree
clfDT =  tree.DecisionTreeClassifier(criterion='gini', max_depth=5)

#Classifier training                 
clfDT.fit(X_train, y_train)

#  Test the trained model on the training set
Y_train_pred_DT=clfDT.predict(X_train)

# Test the trained model on the test set
Y_test_pred_DT=clfDT.predict(X_test)


# Confusion matrixes for tree 
confMatrixTrainDT=confusion_matrix(y_train, Y_train_pred_DT)
confMatrixTestDT=confusion_matrix(y_test, Y_test_pred_DT)


#Evaluation for tree 2
print ('\tClassifier Evaluation')

print ('Accuracy Train=', accuracy_score(y_train, Y_train_pred_DT, normalize=True))
print ('Accuracy Test=', accuracy_score(y_test, Y_test_pred_DT, normalize=True))



print ('train, test: Conf matrix Decision Tree')
print (confMatrixTrainDT)
print ('test, test: Conf matrix Decision Tree')
print(confMatrixTestDT)

print ('Tree train:  Macro Train Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_train, Y_train_pred_DT, average='macro'))
print ('Tree test:  Macro Test Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, Y_test_pred_DT, average='macro'))
print ()



pr_y_test_pred_DT1=clfDT.predict_proba(X_test)

fprDT1, tprDT1, thresholdsDT1 = roc_curve(y_test, pr_y_test_pred_DT1[:,1])









r = export_text(clfDT,feature_cols)
print(r)



y = [str(i) for i in y]
y_str = ['Control' if x=='0' else 'Patient' for x in y]

dot_data = tree.export_graphviz(clfDT, out_file=None, 
                                feature_names=feature_cols,  
                                class_names=y,
                                filled=True)



# Draw graph
graph = graphviz.Source(dot_data, format="png") 
graph




lw=2
plt.figure(10)

plt.plot(fprRFT1,tprRFT1,color='green')

plt.plot(fprDT1,tprDT1,color='blue')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve Athens mentrsual cycle and class using  Random Forest ')
plt.grid(True)
plt.legend(loc="lower right")
plt.show()
plt.savefig('athens_Roc_curve_SVM_Decision.png')




# with driving data
df_extra = pd.merge(file, df, left_on='contestant', right_on='contestant', how='outer')

df_extra.contestant_class.replace(('Control', 'Patient'), (0, 1), inplace=True)
# NTU to 0 and HMU to 1
df_extra.location.replace(('NTU', 'HMU'), (0, 1), inplace=True)
# get only women
df_extra = df_extra[df_extra['Φύλο'] == 2]
df_extra = df_extra.replace('#NULL!',np.nan)
df_extra = df_extra.fillna(0)
df_extra['MENSTRUAL_CYCLE_11'] = df_extra['MENSTRUAL_CYCLE_11'].astype(str).astype(int)
df_extra = df_extra[df_extra['MENSTRUAL_CYCLE_11'] < 99]





feature_cols = list(df_extra.columns)
X = df_extra[feature_cols] # Features


y = df_extra['contestant_class']

del df_extra['contestant_class']
del df_extra['contestant']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, shuffle=True)


clfRFT1 = RandomForestClassifier(n_estimators=5, max_depth=None)
clfRFT1.fit(X_train, y_train)
Y_train_pred_RFT1=clfRFT1.predict(X_train)
Y_test_pred_RFT1=clfRFT1.predict(X_test)


# Confusion matrixes for tree 
confMatrixTrainDT=confusion_matrix(y_train, Y_train_pred_RFT1)
confMatrixTestDT=confusion_matrix(y_test, Y_test_pred_RFT1)



print ()

print ('Accuracy Train=', accuracy_score(y_train, Y_train_pred_RFT1, normalize=True))
print ('Accuracy Test=', accuracy_score(y_test, Y_test_pred_RFT1, normalize=True))


confMatrixTrainRFT1=confusion_matrix(y_train, Y_train_pred_RFT1)
confMatrixTestRFT1=confusion_matrix(y_test, Y_test_pred_RFT1)
print ('train: Conf matrix ')
print(confMatrixTrainRFT1)
print ('test: Conf matrix ')
print(confMatrixTestRFT1)

print ('Random Forest Train: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_train, Y_train_pred_RFT1, average='macro'))
print ('Random Forest Test: Macro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, Y_test_pred_RFT1, average='macro'))
print ('\n')




pr_y_test_pred_RFT1=clfRFT1.predict_proba(X_test)
fprRFT1, tprRFT1, thresholdsRFT1 = roc_curve(y_test, pr_y_test_pred_RFT1[:,1])

lw=2
plt.figure(10)

plt.plot(fprRFT1,tprRFT1,color='green')


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve Athens mentrsual cycle and class using  Random Forest ')
plt.grid(True)
plt.legend(loc="lower right")
plt.show()
