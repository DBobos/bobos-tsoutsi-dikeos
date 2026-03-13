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
# remove crashed control contestants
df = df[(df.contestant != 'CG2')]
df = df[(df.contestant != 'CG5')]
# mrerging the datasets 
df_extra = pd.merge(file, df, left_on='contestant', right_on='contestant', how='outer')
    

df_extra.contestant_class.replace(('Control', 'Patient'), (0, 1), inplace=True)
# NTU to 0 and HMU to 1
df_extra.location.replace(('NTU', 'HMU'), (0, 1), inplace=True)
df_extra = df_extra.replace('#NULL!',np.nan)


# df_extra.to_csv('aggragated_and_extra.csv', index=True)


# FIXING UNANSWERED QUESTION FOR CRETE DATA ON RLSTOTALSCORE AND MENSTRUAL CYCLE_11 FOR MALE HAVING 99 AS INPUT

df_extra = df_extra[(df_extra.contestant != 'DP26')] # HAS NOT ANSWERED THE QUESTIONNAIRES

df_extra = df_extra.replace(np.nan,'50')



df_extra['RLSTotalScore'] = df_extra['RLSTotalScore'].astype(int)

df_extra['RLSTotalScore'] = df_extra['RLSTotalScore'] + 1

df_extra['RLSTotalScore']= df_extra['RLSTotalScore'].replace(51, 0)

df_extra['RLSTotalScore'] = df_extra['RLSTotalScore'].astype(str)


df_extra['MENSTRUAL_CYCLE_11'] = df_extra['MENSTRUAL_CYCLE_11'].astype(int)

df_extra['MENSTRUAL_CYCLE_11'] = df_extra['MENSTRUAL_CYCLE_11'] + 1


df_extra['MENSTRUAL_CYCLE_11']= df_extra['MENSTRUAL_CYCLE_11'].replace(100, 0)

df_extra['MENSTRUAL_CYCLE_11'] = df_extra['MENSTRUAL_CYCLE_11'].astype(str)


df2 = df_extra.copy()

df3 = df_extra.copy()


""" DECISION TREE """






del df_extra['contestant']
y = df_extra['contestant_class']
# y_target = df['contestant_class'] # Target variable
del df_extra['contestant_class']




feature_cols = list(df_extra.columns)
X = df_extra[feature_cols] # Features












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

print(classification_report(y_test, Y_test_pred_DT))





print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_test_pred_DT)))

pr_y_test_pred_DT=clfDT.predict_proba(X_test)

fprDT, tprDT, thresholdsDT = roc_curve(y_test, pr_y_test_pred_DT[:,1])

""" PENALIZING ALGORITHM"""

svc_model = SVC(class_weight='balanced', probability=True)

svc_model.fit(X_train, y_train)

svc_predict = svc_model.predict(X_test)# check performance
print('ROCAUC score:',roc_auc_score(y_test, svc_predict))
print('Accuracy score:',accuracy_score(y_test, svc_predict))
print('F1 score:',f1_score(y_test, svc_predict))








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


""" PLOTTING """
r = export_text(clfDT,feature_cols)
print(r)

with open("decistion_tree_extra.log", "w") as fout:
    fout.write(r)
    

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
plt.plot(fprDT,tprDT,color='blue')
plt.plot(fprRFT1,tprRFT1,color='green')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve using DecisionTreeClassification, Random Forest with decision extra data depths=5')
plt.grid(True)
plt.legend(loc="lower right")
plt.show()













""" FACTORIZATION"""


del df2['contestant']
del df2['location']

df2 = df2.apply(lambda col:pd.to_numeric(col, errors='coerce'))


y = df2['contestant_class']
df2 = df2.drop('contestant_class',axis = 1)



chi2,p = calculate_bartlett_sphericity(df2)
print("Chi squared value : ",chi2)
print("p value : ",p)


kmo_all,kmo_model=calculate_kmo(df2)
print(kmo_model)

# heatmap of correlation matrix
plt.figure(figsize=(100,50))
c= df2.corr()

sns.heatmap(c)


#Creating the Correlation matrix and Selecting the Upper trigular matrix
c= df2.corr().abs()
c.to_csv('correlation matrix.csv')
upper_tri = c.where(np.triu(np.ones(c.shape),k=1).astype(np.bool))
print(upper_tri)

#Droping the column with high correlation
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
print(); print(to_drop)


df_dropped = df2.drop((i for i in to_drop), axis=1)

print(); print(df_dropped.head())

#To figure out how many factors we would need, we can look at eigenvalues, which is a measure of how much of the variance of the variables does a factor explain. 


# Create factor analysis object and perform factor analysis
fa = FactorAnalyzer()
fa.fit(df_dropped,10)
# Check Eigenvalues
ev, v = fa.get_eigenvalues()
ev
plt.scatter(range(1,df_dropped.shape[1]+1),ev)
plt.plot(range(1,df_dropped.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

plt.savefig('eigenvalues_of_factors.png')





# Create factor analysis object and perform factor analysis
 #Varimax rotation, which maxiizes the sum of the variance of squared loadings while ensuring that the factors created are not correlated (orthogonality)
fa = FactorAnalyzer(n_factors=10, rotation='varimax')
fa.fit(df_dropped)

print(pd.DataFrame(fa.loadings_,index=df_dropped.columns))


print(pd.DataFrame(fa.get_factor_variance(),index=['Variance','Proportional Var','Cumulative Var']))

# we see the plot and decide for 10 factors as the eigen value variance explanation drops after the 10th
print(pd.DataFrame(fa.get_communalities(),index=df_dropped.columns,columns=['Communalities']))




# creating a dataframe using the factored data
new_variables = fa.fit_transform(df_dropped)




df_factored = pd.DataFrame(new_variables)


df_factored['y'] = y
# loss of class 1
df_factored = df_factored.fillna(1)










""" DECISION TREE USING FACTORS"""

df_factored2 = df_factored.copy()



y = df_factored2['y']


del df_factored2['y']



feature_cols = list(df_factored2.columns)
X = df_factored2[feature_cols] # Features












# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, shuffle=True)





#Decision tree 2 with factors
clfDT1 =  tree.DecisionTreeClassifier(criterion='gini', max_depth=5)

#Classifier training                 
clfDT1.fit(X_train, y_train)

#  Test the trained model on the training set
Y_train_pred_DT1=clfDT1.predict(X_train)

# Test the trained model on the test set
Y_test_pred_DT1=clfDT1.predict(X_test)


# Confusion matrixes for tree 2
confMatrixTrainDT1=confusion_matrix(y_train, Y_train_pred_DT1)
confMatrixTestDT1=confusion_matrix(y_test, Y_test_pred_DT1)


#Evaluation for tree 2
print ('\tClassifier Evaluation')

print ('Accuracy Train=', accuracy_score(y_train, Y_train_pred_DT1, normalize=True))
print ('Accuracy Test=', accuracy_score(y_test, Y_test_pred_DT1, normalize=True))
print ('train, test: Conf matrix Decision Tree')
print (confMatrixTrainDT1)
print(confMatrixTestDT1)




# Measures of performance: Precision, Recall, F1 for tree 1
print ('Tree train:  Macro Train Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_train, Y_train_pred_DT1, average='macro'))
print ('Tree test:  Macro Test Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, Y_test_pred_DT1, average='macro'))
print ()



print(classification_report(y_test, Y_test_pred_DT1))


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_test_pred_DT1)))


pr_y_test_pred_DT1=clfDT1.predict_proba(X_test)

fprDT1, tprDT1, thresholdsDT1 = roc_curve(y_test, pr_y_test_pred_DT1[:,1])


""" PENALIZING ALGORITHM"""

svc_model = SVC(class_weight='balanced', probability=True)

svc_model.fit(X_train, y_train)

svc_predict = svc_model.predict(X_test)# check performance
print('ROCAUC score:',roc_auc_score(y_test, svc_predict))
print('Accuracy score:',accuracy_score(y_test, svc_predict))
print('F1 score:',f1_score(y_test, svc_predict))







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




""" PLOTTING """
r = export_text(clfDT1,feature_cols)
print(r)

with open("decistion_tree_extra.log", "w") as fout:
    fout.write(r)
    

y = [str(i) for i in y]
y_str = ['Control' if x=='0' else 'Patient' for x in y]

dot_data = tree.export_graphviz(clfDT1, out_file=None, 
                                feature_names=feature_cols,  
                                class_names=y,
                                filled=True)



# Draw graph
graph = graphviz.Source(dot_data, format="png") 
graph


lw=2
plt.figure(10)
plt.plot(fprDT1,tprDT1,color='blue')
plt.plot(fprRFT1,tprRFT1,color='green')
 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve using DecisionTreeClassification, Random Forest factored data max depths=5')
plt.grid(True)
plt.legend(loc="lower right")
plt.show()






""" PREDICTING WITH ONLY EXTRA DATA AND CLASS"""


file.rename({'ID': 'contestant'}, axis=1, inplace=True)

df = pd.read_csv('aggregated_data.csv')

df = df[(df.contestant != 'CG2')]
df = df[(df.contestant != 'CG5')]

df_solo = df[['contestant','contestant_class']]

df_extra_solo = pd.merge(file, df_solo, left_on='contestant', right_on='contestant', how='outer')



df_extra_solo = df_extra_solo[(df_extra_solo.contestant != 'DP26')] # HAS NOT ANSWERED THE QUESTIONNAIRES
df_extra_solo.contestant_class.replace(('Control', 'Patient'), (0, 1), inplace=True)

df_extra_solo = df_extra_solo.replace('#NULL!',np.nan)
df_extra_solo = df_extra_solo.replace(np.nan,'50')



df_extra_solo['RLSTotalScore'] = df_extra_solo['RLSTotalScore'].astype(int)

df_extra_solo['RLSTotalScore'] = df_extra_solo['RLSTotalScore'] + 1

df_extra_solo['RLSTotalScore']= df_extra_solo['RLSTotalScore'].replace(51, 0)

df_extra_solo['RLSTotalScore'] = df_extra_solo['RLSTotalScore'].astype(str)


df_extra_solo['MENSTRUAL_CYCLE_11'] = df_extra_solo['MENSTRUAL_CYCLE_11'].astype(int)

df_extra_solo['MENSTRUAL_CYCLE_11'] = df_extra_solo['MENSTRUAL_CYCLE_11'] + 1


df_extra_solo['MENSTRUAL_CYCLE_11']= df_extra_solo['MENSTRUAL_CYCLE_11'].replace(100, 0)

df_extra_solo['MENSTRUAL_CYCLE_11'] = df_extra_solo['MENSTRUAL_CYCLE_11'].astype(str)






""" DECISION TREE """






del df_extra_solo['contestant']
y = df_extra_solo['contestant_class']
# y_target = df['contestant_class'] # Target variable
del df_extra_solo['contestant_class']




feature_cols = list(df_extra.columns)
X = df_extra[feature_cols] # Features












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


print(classification_report(y_test, Y_test_pred_DT))






print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_test_pred_DT)))
pr_y_test_pred_DT=clfDT.predict_proba(X_test)

fprDT, tprDT, thresholdsDT = roc_curve(y_test, pr_y_test_pred_DT[:,1])

""" PENALIZING ALGORITHM"""

svc_model = SVC(class_weight='balanced', probability=True)

svc_model.fit(X_train, y_train)

svc_predict = svc_model.predict(X_test)# check performance
print('ROCAUC score:',roc_auc_score(y_test, svc_predict))
print('Accuracy score:',accuracy_score(y_test, svc_predict))
print('F1 score:',f1_score(y_test, svc_predict))








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


""" PLOTTING """
r = export_text(clfDT,feature_cols)
print(r)

with open("decistion_tree_extra.log", "w") as fout:
    fout.write(r)
    

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
plt.plot(fprDT,tprDT,color='blue')
plt.plot(fprRFT1,tprRFT1,color='green')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve using DecisionTreeClassification, Random Forest with only extra data depths=5')
plt.grid(True)
plt.legend(loc="lower right")
plt.show()
























"""SVM + DECISION TREE + RANDOM FOREST FOR ATHENS"""



df_athens = df3[df3['location'] == 0]

y = df_athens['contestant_class']
del df_athens['contestant']

del df_athens['contestant_class']

del df_athens['location']

feature_cols = list(df_athens.columns)
X = df_athens[feature_cols] # Features


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
fprRFT1, tprRFT1, thresholdsRFT1 = roc_curve(y_test, pr_y_test_pred_RFT1[:,1])





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
plt.title('ROC Curve Athens extra using SVMClassification , Decision Tree and Random Forest ')
plt.grid(True)
plt.legend(loc="lower right")
plt.show()
plt.savefig('athens_Roc_curve_SVM_Decision.png')







"""SVM , DECISION TREE, RANDOM FOREST FOR CRETE"""

df_crete = df3[df3['location'] == 1]
del df_crete['contestant']
y = df_crete['contestant_class']

del df_crete['contestant_class']

del df_crete['location']

feature_cols = list(df_crete.columns)
X = df_crete[feature_cols] # Features


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, shuffle=True)



#Define Support vector machine1
clfSVM2= svm.SVC(kernel='rbf', probability=True)

#train the classifiers
clfSVM2.fit(X_train, y_train)

#test the trained model on the test set
Y_train_pred_SVM2=clfSVM2.predict(X_train)
Y_test_pred_SVM2=clfSVM2.predict(X_test)
print('Accuracy Train=', accuracy_score(y_train, Y_train_pred_SVM2, normalize=True))
print ('Accuracy Test=', accuracy_score(y_test, Y_test_pred_SVM2, normalize=True))


confMatrixTrainSVM2=confusion_matrix(y_train, Y_train_pred_SVM2)
confMatrixTestSVM2=confusion_matrix(y_test, Y_test_pred_SVM2, labels=None)

print ('Conf matrix Support Vector Classifier')
print('Train conf matrix', confMatrixTrainSVM2)

print ('Test conf matrix',confMatrixTestSVM2)

# Measures of performance: Precision, Recall, F1
print ('SVM1: Train Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_train, Y_train_pred_SVM2, average='micro'))
print ('SVM1: Test Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, Y_test_pred_SVM2, average='micro'))
print ('\n')

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_test_pred_SVM2)))





pr_y_test_pred_SVM2=clfSVM2.predict_proba(X_test)
fprSVM2, tprSVM2, thresholdsSVM2 = roc_curve(y_test, pr_y_test_pred_SVM2[:,1])





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, shuffle=True)
clfDT =  tree.DecisionTreeClassifier(criterion='gini', max_depth=5)

#Classifier training                 
clfDT.fit(X_train, y_train)

#  Test the trained model on the training set
Y_train_pred_DTC=clfDT.predict(X_train)

# Test the trained model on the test set
Y_test_pred_DTC=clfDT.predict(X_test)


# Confusion matrixes for tree 
confMatrixTrainDTC=confusion_matrix(y_train, Y_train_pred_DTC)
confMatrixTestDTC=confusion_matrix(y_test, Y_test_pred_DTC)


#Evaluation for tree 2
print ('\tClassifier Evaluation')

print ('Accuracy Train=', accuracy_score(y_train, Y_train_pred_DTC, normalize=True))
print ('Accuracy Test=', accuracy_score(y_test, Y_test_pred_DTC, normalize=True))

print ('train: Conf matrix Decision Tree')
print (confMatrixTrainDTC)
print ('test: Conf matrix Decision Tree')
print(confMatrixTestDTC)

# Measures of performance: Precision, Recall, F1 for tree 
print ('Tree train:  Macro Train Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_train, Y_train_pred_DTC, average='macro'))
print ('Tree test:  Macro Test Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, Y_test_pred_DTC, average='macro'))
print ()

print(classification_report(y_test, Y_test_pred_DTC))


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_test_pred_DTC)))

pr_y_test_pred_DTC=clfDT.predict_proba(X_test)

fprDTC, tprDTC, thresholdsDTC = roc_curve(y_test, pr_y_test_pred_DTC[:,1])






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
plt.plot(fprSVM2,tprSVM2,color='blue')
plt.plot(fprDTC,tprDTC,color='red')
plt.plot(fprRFT1,tprRFT1,color='green')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve Crete extra using SVMClassification , Decision Tree and Random Forest ')
plt.grid(True) 
plt.legend(loc="lower right")
plt.show()





""" PENALIZING ALGORITHM CRETE"""

svc_model = SVC(class_weight='balanced', probability=True)

svc_model.fit(X_train, y_train)

svc_predict = svc_model.predict(X_test)# check performance
print('ROCAUC score:',roc_auc_score(y_test, svc_predict))
print('Accuracy score:',accuracy_score(y_test, svc_predict))
print('F1 score:',f1_score(y_test, svc_predict))


