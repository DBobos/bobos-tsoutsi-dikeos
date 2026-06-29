import pandas as pd
import numpy as np
#from factor_analyzer import calculate_bartlett_sphericity
#from factor_analyzer import FactorAnalyzer
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
#from factor_analyzer.factor_analyzer import calculate_kmo
#import pingouin as pg
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.tree import export_text
from sklearn import tree
#import graphviz
#from dtreeviz.trees import dtreeviz 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_curve, auc, accuracy_score
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
seed = (10)

df = pd.read_csv('aggregated_data.csv')






# one hot encode
# control to 0 and patient to 1
df.contestant_class.replace(('Control', 'Patient'), (0, 1), inplace=True)
# NTU to 0 and HMU to 1
df.location.replace(('NTU', 'HMU'), (0, 1), inplace=True)
# removing crashed control contestants
df = df[(df.contestant != 'CG2')]
df = df[(df.contestant != 'CG5')]
df = df[(df.contestant != 'CG7')]
# copy the dataframe
df2 = df.copy()
df3 = df.copy()

y = df['contestant_class']
df = df.drop('contestant',axis = 1)

# y_target = df['contestant_class'] # Target variable
del df['contestant_class']




feature_cols = list(df.columns)
X = df[feature_cols] # Features








""" SIMPLE DECISION TREE """



# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, shuffle=True)







#Decision tree
clfDT =  tree.DecisionTreeClassifier(criterion='gini', max_depth=5)

#Classifier training                 
clfDT.fit(X_train, y_train)

#  Train the trained model on the training set
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


from sklearn.metrics import ConfusionMatrixDisplay


ConfusionMatrixDisplay.from_estimator(clfDT, X_train , y_train)
plt.show()

ConfusionMatrixDisplay.from_estimator(clfDT, X_test, y_test)
plt.show()

print (confMatrixTrainDT)
print(confMatrixTestDT)

# Measures of performance: Precision, Recall, F1 for tree 
print ('Tree train:  Macro Train Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_train, Y_train_pred_DT, average='macro'))
print ('Tree test:  Macro Test Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, Y_test_pred_DT, average='macro'))
print ()

print(classification_report(y_test, Y_test_pred_DT))







print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_test_pred_DT)))

pr_y_test_pred_DT=clfDT.predict_proba(X_test)
#print(pr_y_test_pred_DT)
fprDT, tprDT, thresholdsDT = roc_curve(y_test, pr_y_test_pred_DT[:,1])


""" PENALIZING ALGORITHM"""

svc_model = SVC(class_weight='balanced', probability=True)

svc_model.fit(X_train, y_train)

svc_predict = svc_model.predict(X_test)# check performance
print('ROCAUC score:',roc_auc_score(y_test, svc_predict))
print('Accuracy score:',accuracy_score(y_test, svc_predict))
print('F1 score:',f1_score(y_test, svc_predict))




joblib.dump(clfDT, "agg_Decision_tree.joblib")


"""RANDOM FOREST"""





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, shuffle=True)


clfRFT1 = RandomForestClassifier(n_estimators=200, class_weight='balanced',max_depth=2)
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

print(classification_report(y_test, Y_test_pred_RFT1))

pr_y_test_pred_RFT1=clfRFT1.predict_proba(X_test)
fprRFT1, tprRFT1, thresholdsRFT1 = roc_curve(y_test, pr_y_test_pred_RFT1[:,1])


#joblib.dump(clfRFT1, "agg_random_forest.joblib")




importances = clfRFT1.feature_importances_

sorted_indices = np.argsort(importances)[::-1]
print(*X_train.columns[sorted_indices][:5], sep = "\n")

# urban_lane_offset_m_std
# motorway_steering_std
# urban_headway_s_std
# motorway_brake_std
# urban_brake_mean

from sklearn.inspection import permutation_importance
# r = permutation_importance(clfRFT1, X_test, y_test,
#                            n_repeats=30,
#                            random_state=0)

# for i in r.importances_mean.argsort()[::-1]:
#     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
#         print(f"{X.columns[i]:<8}"
#               f"{r.importances_mean[i]:.3f}"
#               f" +/- {r.importances_std[i]:.3f}")

# Feature importance based on mean decrease in impurity
# urban_speed_mps_mean0.028 +/- 0.005
scoring = ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error']
r_multi = permutation_importance(
    clfRFT1, X_test, y_test, n_repeats=30, random_state=0, scoring=scoring)

for metric in r_multi:
    print(f"{metric}")
    r = r_multi[metric]
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"    {X.columns[i]:<8}"
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")


del X['location']
sort = clfRFT1.feature_importances_.argsort()

fig, ax = plt.subplots(figsize=(16, 9))

ax.barh(X.columns[sort], clfRFT1.feature_importances_[sort], height=0.7)

plt.xlabel("Feature Importance")
plt.ylabel("Factors")




# permutation importance 
#  r2 scores
#     urban_speed_mps_mean0.123 +/- 0.023
# neg_mean_absolute_percentage_error
#     urban_speed_mps_mean 124385132565470.828 +/- 23097739255942.531
# neg_mean_squared_error
#     urban_speed_mps_mean0.028 +/- 0.005


# """ PLOTTING """
# r = export_text(clfDT,feature_cols)
# print(r)

# with open("decistion_tree.log", "w") as fout:
#     fout.write(r)
    
# y = [str(i) for i in y]
# y_str = ['Control' if x=='0' else 'Patient' for x in y]

# dot_data = tree.export_graphviz(clfDT, out_file=None, 
#                                 feature_names=feature_cols,  
#                                 class_names=y_str,
#                                 filled=True)



# # Draw graph
# graph = graphviz.Source(dot_data, format="png") 
# graph

# graph.render("decision_tree_graphivz")








# regr = DecisionTreeRegressor(max_depth=5, random_state=1)
# model = regr.fit(X, y)

# text_representation = tree.export_text(regr)
# print(text_representation)
# with open("decistion_tree_max_depth=4.log", "w") as fout:
#     fout.write(text_representation)
    
    
    
# fig = plt.figure(figsize=(25,20))
# _ = tree.plot_tree(regr, feature_names=feature_cols, filled=True)


# fig.savefig('decision_max_5.png')


# dot_data2 = tree.export_graphviz(regr, out_file=None, 
#                                 feature_names=feature_cols,  
#                                 filled=True)
# graph2 = graphviz.Source(dot_data2, format="png") 
# graph2.render("decision_tree_max_depth=5")






# lw=2
# plt.figure(10)
# plt.plot(fprDT,tprDT,color='blue')

# plt.plot(fprRFT1,tprRFT1,color='green')

# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.title('ROC Curve Decision Tree and Random Forest ')
# plt.grid(True)
# plt.legend(loc="lower right")
# plt.show()
# plt.savefig('athens_Roc_curve_SVM_Decision.png')











# """SVM + DECISION TREE  FOR ATHENS"""



# df_athens = df3[df3['location'] == 0]
# y = df_athens['contestant_class']
# del df_athens['contestant']

# del df_athens['contestant_class']

# del df_athens['location']

# feature_cols = list(df_athens.columns)
# X = df_athens[feature_cols] # Features


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, shuffle=True)



# #Define Support vector machine1
# clfSVM1= svm.SVC(kernel='rbf', probability=True)

# #train the classifiers
# clfSVM1.fit(X_train, y_train)

# #test the trained model on the test set
# Y_train_pred_SVM1=clfSVM1.predict(X_train)
# Y_test_pred_SVM1=clfSVM1.predict(X_test)

# print('Accuracy Train=', accuracy_score(y_train, Y_train_pred_SVM1, normalize=True))
# print ('Accuracy Test=', accuracy_score(y_test, Y_test_pred_SVM1, normalize=True))

# confMatrixTrainSVM1=confusion_matrix(y_train, Y_train_pred_SVM1)
# confMatrixTestSVM1=confusion_matrix(y_test, Y_test_pred_SVM1, labels=None)

# print ('Conf matrix Support Vector Classifier')
# print('Train conf matrix', confMatrixTrainSVM1)

# print ('Test conf matrix',confMatrixTestSVM1)


# # Measures of performance: Precision, Recall, F1
# print ('SVM1: Train Precision, recall, f1-score')
# print ( precision_recall_fscore_support(y_train, Y_train_pred_SVM1, average='micro'))
# print ('SVM1: Test Precision, recall, f1-score')
# print (precision_recall_fscore_support(y_test, Y_test_pred_SVM1, average='micro'))
# print ('\n')

# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_test_pred_SVM1)))



# pr_y_test_pred_SVM1=clfSVM1.predict_proba(X_test)



# fprSVM1, tprSVM1, thresholdsSVM1 = roc_curve(y_test, pr_y_test_pred_SVM1[:,1])



# """ PENALIZING ALGORITHM ATHENS"""

# svc_model = SVC(class_weight='balanced', probability=True)

# svc_model.fit(X_train, y_train)

# svc_predict = svc_model.predict(X_test)# check performance
# print('ROCAUC score:',roc_auc_score(y_test, svc_predict))
# print('Accuracy score:',accuracy_score(y_test, svc_predict))
# print('F1 score:',f1_score(y_test, svc_predict))

# print(classification_report(y_test, Y_test_pred_SVM1))






# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, shuffle=True)
# clfDT =  tree.DecisionTreeClassifier(criterion='gini', max_depth=5)

# #Classifier training                 
# clfDT.fit(X_train, y_train)

# #  Test the trained model on the training set
# Y_train_pred_DTA=clfDT.predict(X_train)

# # Test the trained model on the test set
# Y_test_pred_DTA=clfDT.predict(X_test)


# # Confusion matrixes for tree 
# confMatrixTrainDTA=confusion_matrix(y_train, Y_train_pred_DTA)
# confMatrixTestDTA=confusion_matrix(y_test, Y_test_pred_DTA)


# #Evaluation for tree 2
# print ('\tClassifier Evaluation')

# print ('Accuracy Train=', accuracy_score(y_train, Y_train_pred_DTA, normalize=True))
# print ('Accuracy Test=', accuracy_score(y_test, Y_test_pred_DTA, normalize=True))

# print ('train: Conf matrix Decision Tree')
# print (confMatrixTrainDTA)
# print ('test: Conf matrix Decision Tree')
# print(confMatrixTestDTA)


# # Measures of performance: Precision, Recall, F1 for tree 
# print ('Tree train:  Macro Train Precision, recall, f1-score')
# print ( precision_recall_fscore_support(y_train, Y_train_pred_DTA, average='macro'))
# print ('Tree test:  Macro Test Precision, recall, f1-score')
# print (precision_recall_fscore_support(y_test, Y_test_pred_DTA, average='macro'))
# print ()


# print(classification_report(y_test, Y_test_pred_DTA))

# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_test_pred_DTA)))
# pr_y_test_pred_DTA=clfDT.predict_proba(X_test)

# fprDTA, tprDTA, thresholdsDTA = roc_curve(y_test, pr_y_test_pred_DTA[:,1])















# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, shuffle=True)


# clfRFT1 = RandomForestClassifier(n_estimators=5, max_depth=None)
# clfRFT1.fit(X_train, y_train)
# Y_train_pred_RFT1=clfRFT1.predict(X_train)
# Y_test_pred_RFT1=clfRFT1.predict(X_test)


# # Confusion matrixes for tree 
# confMatrixTrainDT=confusion_matrix(y_train, Y_train_pred_RFT1)
# confMatrixTestDT=confusion_matrix(y_test, Y_test_pred_RFT1)



# print ()

# print ('Accuracy Train=', accuracy_score(y_train, Y_train_pred_RFT1, normalize=True))
# print ('Accuracy Test=', accuracy_score(y_test, Y_test_pred_RFT1, normalize=True))


# confMatrixTrainRFT1=confusion_matrix(y_train, Y_train_pred_RFT1)
# confMatrixTestRFT1=confusion_matrix(y_test, Y_test_pred_RFT1)
# print ('train: Conf matrix ')
# print(confMatrixTrainRFT1)
# print ('test: Conf matrix ')
# print(confMatrixTestRFT1)

# print ('Random Forest Train: Macro Precision, recall, f1-score')
# print ( precision_recall_fscore_support(y_train, Y_train_pred_RFT1, average='macro'))
# print ('Random Forest Test: Macro Precision, recall, f1-score')
# print (precision_recall_fscore_support(y_test, Y_test_pred_RFT1, average='macro'))
# print ('\n')






# pr_y_test_pred_RFT1=clfRFT1.predict_proba(X_test)
# fprRFT1, tprRFT1, thresholdsRFT1 = roc_curve(y_test, pr_y_test_pred_RFT1[:,1])





# """ PLOTTING """
# r = export_text(clfDT,feature_cols)
# print(r)

# with open("decistion_tree.log", "w") as fout:
#     fout.write(r)
    
# y = [str(i) for i in y]
# y_str = ['Control' if x=='0' else 'Patient' for x in y]

# dot_data = tree.export_graphviz(clfDT, out_file=None, 
#                                 feature_names=feature_cols,  
#                                 class_names=y_str,
#                                 filled=True)


# graph2 = graphviz.Source(dot_data, format="png") 
# graph2.render("decision_tree_max_depth=5")

# lw=2
# plt.figure(10)
# plt.plot(fprSVM1,tprSVM1,color='blue')
# plt.plot(fprDTA,tprDTA,color='red')
# plt.plot(fprRFT1,tprRFT1,color='green')

# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.title('ROC Curve Athens using SVMClassification , Decision Tree and Random Forest ')
# plt.grid(True)
# plt.legend(loc="lower right")
# plt.show()
# plt.savefig('athens_Roc_curve_SVM_Decision.png')







# """SVM AND DECISION TREE FOR CRETE"""

# df_crete = df3[df3['location'] == 1]

# del df_crete['contestant']
# y = df_crete['contestant_class']

# del df_crete['contestant_class']

# del df_crete['location']

# feature_cols = list(df_crete.columns)
# X = df_crete[feature_cols] # Features


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, shuffle=True)



# #Define Support vector machine1
# clfSVM2= svm.SVC(kernel='rbf', probability=True)

# #train the classifiers
# clfSVM2.fit(X_train, y_train)

# #test the trained model on the test set
# Y_train_pred_SVM2=clfSVM2.predict(X_train)
# Y_test_pred_SVM2=clfSVM2.predict(X_test)
# print('Accuracy Train=', accuracy_score(y_train, Y_train_pred_SVM2, normalize=True))
# print ('Accuracy Test=', accuracy_score(y_test, Y_test_pred_SVM2, normalize=True))

# confMatrixTrainSVM2=confusion_matrix(y_train, Y_train_pred_SVM2)
# confMatrixTestSVM2=confusion_matrix(y_test, Y_test_pred_SVM2, labels=None)

# print ('Conf matrix Support Vector Classifier')
# print('Train conf matrix', confMatrixTrainSVM2)

# print ('Test conf matrix',confMatrixTestSVM2)


# # Measures of performance: Precision, Recall, F1
# print ('SVM1: Train Precision, recall, f1-score')
# print ( precision_recall_fscore_support(y_train, Y_train_pred_SVM2, average='micro'))
# print ('SVM1: Test Precision, recall, f1-score')
# print (precision_recall_fscore_support(y_test, Y_test_pred_SVM2, average='micro'))
# print ('\n')

# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_test_pred_SVM2)))




# pr_y_test_pred_SVM2=clfSVM2.predict_proba(X_test)
# fprSVM2, tprSVM2, thresholdsSVM2 = roc_curve(y_test, pr_y_test_pred_SVM2[:,1])





# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, shuffle=True)
# clfDT =  tree.DecisionTreeClassifier(criterion='gini', max_depth=5)

# #Classifier training                 
# clfDT.fit(X_train, y_train)

# #  Test the trained model on the training set
# Y_train_pred_DTC=clfDT.predict(X_train)

# # Test the trained model on the test set
# Y_test_pred_DTC=clfDT.predict(X_test)


# # Confusion matrixes for tree 
# confMatrixTrainDTC=confusion_matrix(y_train, Y_train_pred_DTC)
# confMatrixTestDTC=confusion_matrix(y_test, Y_test_pred_DTC)


# #Evaluation for tree 2
# print ('\tClassifier Evaluation')

# print ('Accuracy Train=', accuracy_score(y_train, Y_train_pred_DTC, normalize=True))
# print ('Accuracy Test=', accuracy_score(y_test, Y_test_pred_DTC, normalize=True))

# print ('train: Conf matrix Decision Tree')
# print (confMatrixTrainDTC)
# print ('test: Conf matrix Decision Tree')
# print(confMatrixTestDTC)

# # Measures of performance: Precision, Recall, F1 for tree 
# print ('Tree train:  Macro Train Precision, recall, f1-score')
# print ( precision_recall_fscore_support(y_train, Y_train_pred_DTC, average='macro'))
# print ('Tree test:  Macro Test Precision, recall, f1-score')
# print (precision_recall_fscore_support(y_test, Y_test_pred_DTC, average='macro'))
# print ()

# print(classification_report(y_test, Y_test_pred_DTC))


# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_test_pred_DTC)))
# pr_y_test_pred_DTC=clfDT.predict_proba(X_test)

# fprDTC, tprDTC, thresholdsDTC = roc_curve(y_test, pr_y_test_pred_DTC[:,1])




# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, shuffle=True)


# clfRFT1 = RandomForestClassifier(n_estimators=5, max_depth=None)
# clfRFT1.fit(X_train, y_train)
# Y_train_pred_RFT1=clfRFT1.predict(X_train)
# Y_test_pred_RFT1=clfRFT1.predict(X_test)


# # Confusion matrixes for tree 
# confMatrixTrainDT=confusion_matrix(y_train, Y_train_pred_RFT1)
# confMatrixTestDT=confusion_matrix(y_test, Y_test_pred_RFT1)



# print ()

# print ('Accuracy Train=', accuracy_score(y_train, Y_train_pred_RFT1, normalize=True))
# print ('Accuracy Test=', accuracy_score(y_test, Y_test_pred_RFT1, normalize=True))


# confMatrixTrainRFT1=confusion_matrix(y_train, Y_train_pred_RFT1)
# confMatrixTestRFT1=confusion_matrix(y_test, Y_test_pred_RFT1)
# print ('train: Conf matrix ')
# print(confMatrixTrainRFT1)
# print ('test: Conf matrix ')
# print(confMatrixTestRFT1)

# print ('Random Forest Train: Macro Precision, recall, f1-score')
# print ( precision_recall_fscore_support(y_train, Y_train_pred_RFT1, average='macro'))
# print ('Random Forest Test: Macro Precision, recall, f1-score')
# print (precision_recall_fscore_support(y_test, Y_test_pred_RFT1, average='macro'))
# print ('\n')






# pr_y_test_pred_RFT1=clfRFT1.predict_proba(X_test)
# fprRFT1, tprRFT1, thresholdsRFT1 = roc_curve(y_test, pr_y_test_pred_RFT1[:,1])



# lw=2
# plt.figure(10)
# plt.plot(fprSVM2,tprSVM2,color='blue')
# plt.plot(fprDTC,tprDTC,color='red')
# plt.plot(fprRFT1,tprRFT1,color='green')
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.title('ROC Curve Crete using SVMClassification , Decision Tree and Random Forest ')
# plt.grid(True) 
# plt.legend(loc="lower right")
# plt.show()





# """ PENALIZING ALGORITHM CRETE"""

# svc_model = SVC(class_weight='balanced', probability=True)

# svc_model.fit(X_train, y_train)

# svc_predict = svc_model.predict(X_test)# check performance
# print('ROCAUC score:',roc_auc_score(y_test, svc_predict))
# print('Accuracy score:',accuracy_score(y_test, svc_predict))
# print('F1 score:',f1_score(y_test, svc_predict))







# r = export_text(clfDT,feature_cols)
# print(r)

# with open("decistion_tree.log", "w") as fout:
#     fout.write(r)
    
# y = [str(i) for i in y]
# y_str = ['Control' if x=='0' else 'Patient' for x in y]

# dot_data = tree.export_graphviz(clfDT, out_file=None, 
#                                 feature_names=feature_cols,  
#                                 class_names=y_str,
#                                 filled=True)


# graph2 = graphviz.Source(dot_data, format="png") 









# """ FACTORIZATION"""



# y = df2['contestant_class']
# df2 = df2.drop('contestant_class',axis = 1)
# del df2['location']
# del df2['contestant']
# # # Get one hot encoding of columns B
# # one_hot = pd.get_dummies(df['contestant'])
# # # Drop column B as it is now encoded
# # df = df.drop('contestant',axis = 1)
# # # Join the encoded df
# # df = df.join(one_hot)

# # # label encoded contestant column
# # le = preprocessing.LabelEncoder()

# # df2['contestant'] = le.fit_transform(df2['contestant'])


# chi2,p = calculate_bartlett_sphericity(df2)
# print("Chi squared value : ",chi2)
# print("p value : ",p)


# kmo_all,kmo_model=calculate_kmo(df2)
# print(kmo_model)

# # heatmap of correlation matrix
# plt.figure(figsize=(20,10))
# c= df2.corr()

# sns.heatmap(c)


# #Creating the Correlation matrix and Selecting the Upper trigular matrix
# c= df2.corr().abs()
# c.to_csv('correlation matrix.csv')
# upper_tri = c.where(np.triu(np.ones(c.shape),k=1).astype(np.bool))
# print(upper_tri)

# #Droping the column with high correlation
# to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
# print(); print(to_drop)
# #['urban_accelerator_std', 'urban_brake_std', 'urban_orientation_std']

# df_dropped = df2.drop((i for i in to_drop), axis=1)

# print(); print(df_dropped.head())

# #To figure out how many factors we would need, we can look at eigenvalues, which is a measure of how much of the variance of the variables does a factor explain. 


# # Create factor analysis object and perform factor analysis
# fa = FactorAnalyzer()
# fa.fit(df_dropped,10)
# # Check Eigenvalues
# ev, v = fa.get_eigenvalues()
# ev
# plt.scatter(range(1,df_dropped.shape[1]+1),ev)
# plt.plot(range(1,df_dropped.shape[1]+1),ev)
# plt.title('Scree Plot')
# plt.xlabel('Factors')
# plt.ylabel('Eigenvalue')
# plt.grid()
# plt.show()

# plt.savefig('eigenvalues_of_factors.png')





# # Create factor analysis object and perform factor analysis
#  #Varimax rotation, which maxiizes the sum of the variance of squared loadings while ensuring that the factors created are not correlated (orthogonality)
# fa = FactorAnalyzer(n_factors=5, rotation='varimax')
# fa.fit(df_dropped)

# print(pd.DataFrame(fa.loadings_,index=df_dropped.columns))


# print(pd.DataFrame(fa.get_factor_variance(),index=['Variance','Proportional Var','Cumulative Var']))

# print(pd.DataFrame(fa.get_communalities(),index=df_dropped.columns,columns=['Communalities']))





# # creating a dataframe using the factored data
# new_variables = fa.fit_transform(df_dropped)




# df_factored = pd.DataFrame(new_variables)


# df_factored['y'] = y

# # loss of 2  class 0 
# df_factored = df_factored.fillna(0)


# """REGRESSION"""
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Ridge, Lasso
# from sklearn import linear_model
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# X = df3
# y = df3['contestant_class']


# #Remove high correlation variables
# # ['urban_accelerator_mean', 'urban_accelerator_std', 'urban_brake_std', 'urban_orientation_std', 'urban_steering_mean']
# del df3['urban_accelerator_mean']
# del df3['urban_accelerator_std']
# del df3['urban_brake_std']
# del df3['urban_orientation_std']
# del df3['urban_steering_mean']
# del df3['contestant']

# del df3['contestant_class']

 

# #split to test and train data

 

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, shuffle=True)

 

# #Linear Regression
# lreg = linear_model.LinearRegression()
 

# lreg.fit(X_train, y_train)



# ep_predict = lreg.predict(X_test)

 

# #Model evaluation
# print("'Coefficients: \n ", lreg.coef_)
# print('Intercept: \n',lreg.intercept_)
# print("Mean squared error: %.2f" % mean_squared_error(y_test, ep_predict))
# print("Mean absolute error: %.2f" % mean_absolute_error(y_test, ep_predict))
# r2=r2_score(y_test, ep_predict)
# print("R2=", r2_score(y_test, ep_predict))
# vif=1/(1-r2)
# print (vif)




# #Lasso and Ridge




# regr2 = linear_model.LinearRegression()
# regr3 = linear_model.LinearRegression()
# regr3 = Ridge(alpha=0.4, normalize=True)
# regr2=Lasso(alpha=0.5, normalize=True)
# regr2.fit(X_train, y_train)
# regr3.fit(X_train, y_train)



# # Make predictions using the testing set, linear
# df_y_pred2 = regr2.predict(X_test)
# df_y_pred3 = regr3.predict(X_test)



# # The coefficients
# print ('Coefficients Linear Regression Lasso: \n', regr2.coef_)
# print ('Interecept: ',regr2.intercept_)

# # The mean squared error
# "Mean squared error: %.2f" % mean_squared_error(y_test, df_y_pred2)
# print ('R2 score Linear: %.2f' % r2_score(y_test, df_y_pred2))
# print ("Mean absolute error Linear: %.2f" % mean_absolute_error(y_test, df_y_pred2))
# print ("Mean squared error Linear: %.2f" % mean_squared_error(y_test, df_y_pred2))
# r2 = r2_score(y_test, df_y_pred2)
# vif=1/(1-r2)
# print (vif)



# # The coefficients
# print ('Coefficients Linear Regression Ridge: \n', regr3.coef_)
# print ('Interecept: ',regr3.intercept_)

# # The mean squared error
# "Mean squared error: %.2f" % mean_squared_error(y_test, df_y_pred3)
# # Explained variance score: 1 is perfect prediction
# print ('R2 score Linear: %.2f' % r2_score(y_test, df_y_pred3))
# print ("Mean absolute error Linear: %.2f" % mean_absolute_error(y_test, df_y_pred3))
# print ("Mean squared error Linear: %.2f" % mean_squared_error(y_test, df_y_pred3))
# r2 = r2_score(y_test, df_y_pred3)
# vif=1/(1-r2)
# print ('vif:',vif)




# import statsmodels.api as sm

# x, y = np.array(X), np.array(y)

# x = sm.add_constant(x)


# model = sm.OLS(y, x)


# results = model.fit()


# print(results.summary())

# print('coefficient of determination:', results.rsquared)
# print('adjusted coefficient of determination:', results.rsquared_adj)
# print('regression coefficients:', results.params)




# print('predicted response:', results.fittedvalues, sep='\n')

# print('predicted response:', results.predict(x), sep='\n')



# """ REGRESSION USING FACTORED DATASET"""


# y = df_factored['y']

# del df_factored['y']

# X = df_factored


# x, y = np.array(X), np.array(y)

# x = sm.add_constant(x)


# model = sm.OLS(y, x)


# results = model.fit()


# print(results.summary())




# """ DECISION TREE USING FACTORS"""








# feature_cols = list(df_factored.columns)
# X = df_factored[feature_cols] # Features












# # Split dataset into training set and test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, shuffle=True)





# #Decision tree 2 with factors
# clfDT1 =  tree.DecisionTreeClassifier(criterion='gini', max_depth=5)

# #Classifier training                 
# clfDT1.fit(X_train, y_train)

# #  Test the trained model on the training set
# Y_train_pred_DT1=clfDT1.predict(X_train)

# # Test the trained model on the test set
# Y_test_pred_DT1=clfDT1.predict(X_test)


# # Confusion matrixes for tree 2
# confMatrixTrainDT1=confusion_matrix(y_train, Y_train_pred_DT1)
# confMatrixTestDT1=confusion_matrix(y_test, Y_test_pred_DT1)


# #Evaluation for tree 2
# print ('\tClassifier Evaluation')

# print ('Accuracy Train=', accuracy_score(y_train, Y_train_pred_DT1, normalize=True))
# print ('Accuracy Test=', accuracy_score(y_test, Y_test_pred_DT1, normalize=True))

# print ('train: Conf matrix Decision Tree')
# print (confMatrixTrainDT1)
# print ('test: Conf matrix Decision Tree')
# print(confMatrixTestDT1)


# # Measures of performance: Precision, Recall, F1 for tree 1
# print ('Tree train:  Macro Train Precision, recall, f1-score')
# print ( precision_recall_fscore_support(y_train, Y_train_pred_DT1, average='macro'))
# print ('Tree test:  Macro Test Precision, recall, f1-score')
# print (precision_recall_fscore_support(y_test, Y_test_pred_DT1, average='macro'))
# print ()


# print(classification_report(y_test, Y_test_pred_DT1))

# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_test_pred_DT1)))
# pr_y_test_pred_DT1=clfDT1.predict_proba(X_test)
# print(pr_y_test_pred_DT1)
# fprDT1, tprDT1, thresholdsDT1 = roc_curve(y_test, pr_y_test_pred_DT1[:,1])


# """ PENALIZING ALGORITHM"""

# svc_model = SVC(class_weight='balanced', probability=True)

# svc_model.fit(X_train, y_train)

# svc_predict = svc_model.predict(X_test)# check performance
# print('ROCAUC score:',roc_auc_score(y_test, svc_predict))
# print('Accuracy score:',accuracy_score(y_test, svc_predict))
# print('F1 score:',f1_score(y_test, svc_predict))






# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, shuffle=True)


# clfRFT1 = RandomForestClassifier(n_estimators=5, max_depth=None)
# clfRFT1.fit(X_train, y_train)
# Y_train_pred_RFT1=clfRFT1.predict(X_train)
# Y_test_pred_RFT1=clfRFT1.predict(X_test)


# # Confusion matrixes for tree 
# confMatrixTrainDT=confusion_matrix(y_train, Y_train_pred_RFT1)
# confMatrixTestDT=confusion_matrix(y_test, Y_test_pred_RFT1)



# print ()

# print ('Accuracy Train=', accuracy_score(y_train, Y_train_pred_RFT1, normalize=True))
# print ('Accuracy Test=', accuracy_score(y_test, Y_test_pred_RFT1, normalize=True))


# confMatrixTrainRFT1=confusion_matrix(y_train, Y_train_pred_RFT1)
# confMatrixTestRFT1=confusion_matrix(y_test, Y_test_pred_RFT1)
# print ('train: Conf matrix ')
# print(confMatrixTrainRFT1)
# print ('test: Conf matrix ')
# print(confMatrixTestRFT1)

# print ('Random Forest Train: Macro Precision, recall, f1-score')
# print ( precision_recall_fscore_support(y_train, Y_train_pred_RFT1, average='macro'))
# print ('Random Forest Test: Macro Precision, recall, f1-score')
# print (precision_recall_fscore_support(y_test, Y_test_pred_RFT1, average='macro'))
# print ('\n')






# pr_y_test_pred_RFT1=clfRFT1.predict_proba(X_test)
# fprRFT1, tprRFT1, thresholdsRFT1 = roc_curve(y_test, pr_y_test_pred_RFT1[:,1])





# """ PLOTTING FACTORIZED"""
# r = export_text(clfDT1,feature_cols)
# print(r)

# with open("decistion_tree_factored.log", "w") as fout:
#     fout.write(r)
    
# y = [str(i) for i in y]
# y_str = ['Control' if x=='0' else 'Patient' for x in y]

# dot_data = tree.export_graphviz(clfDT1, out_file=None, 
#                                 feature_names=feature_cols,  
#                                 class_names=y_str,
#                                 filled=True)



# # Draw graph
# graph = graphviz.Source(dot_data, format="png") 
# graph

# graph.render("decision_tree_factored_graphivz")








# regr = DecisionTreeRegressor(max_depth=6, random_state=1)
# model = regr.fit(X, y)

# text_representation = tree.export_text(regr)
# print(text_representation)
# with open("decistion_tree_max_depth=4_factored.log", "w") as fout:
#     fout.write(text_representation)
    
    
    
# fig = plt.figure(figsize=(25,20))
# _ = tree.plot_tree(regr, feature_names=feature_cols, filled=True)


# fig.savefig('decision_max_4_factored.png')


# dot_data2 = tree.export_graphviz(regr, out_file=None, 
#                                 feature_names=feature_cols,  
#                                 filled=True)
# graph2 = graphviz.Source(dot_data2, format="png") 
# graph2.render("decision_tree_max_depth=5_factored")






# lw=2
# plt.figure(10)
# plt.plot(fprDT,tprDT,color='blue')
# plt.plot(fprRFT1, tprRFT1,color='green')
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.title('ROC Curve using DecisionTreeClassification, Random Forest with decision and factor max depths=5')
# plt.grid(True)
# plt.legend(loc="lower right")
# plt.show()






