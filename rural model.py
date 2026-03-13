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


df = pd.read_csv('aggregated_data_rural.csv')




""" Models"""


del df['location']


df = df[(df.contestant != 'CG2')]
df = df[(df.contestant != 'CG5')]
del df['contestant']

df.contestant_class.replace(('Control', 'Patient'), (0, 1), inplace=True)

y = df['contestant_class']


del df['contestant_class']



feature_cols = list(df.columns)
X = df[feature_cols] # Features


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, shuffle=True)



#Define Support vector machine1
clfSVM1= svm.SVC(kernel='rbf', probability=True)

#train the classifiers
clfSVM1.fit(X_train, y_train)

#test the trained model on the test set
Y_train_pred_SVM1=clfSVM1.predict(X_train)
Y_test_pred_SVM1=clfSVM1.predict(X_test)
print ('Accuracy Train=', accuracy_score(y_train, Y_train_pred_SVM1, normalize=True))
print ('Accuracy Test=', accuracy_score(y_test, Y_test_pred_SVM1, normalize=True))
#Accuracy Test= 0.7894736842105263

confMatrixTrainSVM1=confusion_matrix(y_train, Y_train_pred_SVM1)
confMatrixTestSVM1=confusion_matrix(y_test, Y_test_pred_SVM1, labels=None)

print ('Conf matrix Support Vector Classifier')
print('Train conf matrix', confMatrixTrainSVM1)

print ('Test conf matrix',confMatrixTestSVM1)

# Measures of performance: Precision, Recall, F1
print ('SVM1: Train Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_train, Y_train_pred_SVM1, average='micro'))
print ('SVM1: Test Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, Y_test_pred_SVM1, average='micro'))
print ('\n')

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_test_pred_SVM1)))




pr_y_test_pred_SVM1=clfSVM1.predict_proba(X_test)



fprSVM1, tprSVM1, thresholdsSVM1 = roc_curve(y_test, pr_y_test_pred_SVM1[:,1])



""" PENALIZING ALGORITHM ATHENS"""

svc_model = SVC(class_weight='balanced', probability=True)

svc_model.fit(X_train, y_train)

svc_predict = svc_model.predict(X_test)# check performance
print('ROCAUC score:',roc_auc_score(y_test, svc_predict))
print('Accuracy score:',accuracy_score(y_test, svc_predict))
print('F1 score:',f1_score(y_test, svc_predict))

print(classification_report(y_test, Y_test_pred_SVM1))





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, shuffle=True)
clfDT =  tree.DecisionTreeClassifier(criterion='gini', max_depth=5)

#Classifier training                 
clfDT.fit(X_train, y_train)

#  Test the trained model on the training set
Y_train_pred_DTA=clfDT.predict(X_train)

# Test the trained model on the test set
Y_test_pred_DTA=clfDT.predict(X_test)


# Confusion matrixes for tree 
confMatrixTrainDTA=confusion_matrix(y_train, Y_train_pred_DTA)
confMatrixTestDTA=confusion_matrix(y_test, Y_test_pred_DTA)


#Evaluation for tree 2
print ('\tClassifier Evaluation')

print ('Accuracy Train=', accuracy_score(y_train, Y_train_pred_DTA, normalize=True))
print ('Accuracy Test=', accuracy_score(y_test, Y_test_pred_DTA, normalize=True))

print ('train: Conf matrix Decision Tree')
print (confMatrixTrainDTA)
print ('test: Conf matrix Decision Tree')
print(confMatrixTestDTA)

# Measures of performance: Precision, Recall, F1 for tree 
print ('Tree train:  Macro Train Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_train, Y_train_pred_DTA, average='macro'))
print ('Tree test:  Macro Test Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, Y_test_pred_DTA, average='macro'))
print ()



print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_test_pred_DTA)))
pr_y_test_pred_DTA=clfDT.predict_proba(X_test)

fprDTA, tprDTA, thresholdsDTA = roc_curve(y_test, pr_y_test_pred_DTA[:,1])













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
fprRFT1, tprRFT1, thresholdsRFT1 = roc_curve(y_test, pr_y_test_pred_RFT1[:,1], pos_label='1')





""" PLOTTING """
r = export_text(clfDT,feature_cols)
print(r)

with open("decistion_tree.log", "w") as fout:
    fout.write(r)
    
y = [str(i) for i in y]
y_str = ['Control' if x=='0' else 'Patient' for x in y]

dot_data = tree.export_graphviz(clfDT, out_file=None, 
                                feature_names=feature_cols,  
                                class_names=y_str,
                                filled=True)

# Draw graph
graph = graphviz.Source(dot_data, format="png") 
graph



lw=2
plt.figure(10)
plt.plot(fprSVM1,tprSVM1,color='blue')
plt.plot(fprDTA,tprDTA,color='red')
plt.plot(fprRFT1,tprRFT1,color='green')

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve Athens rural using SVMClassification , Decision Tree and Random Forest ')
plt.grid(True)
plt.legend(loc="lower right")
plt.show()
plt.savefig('athens_Roc_curve_SVM_Decision.png')
