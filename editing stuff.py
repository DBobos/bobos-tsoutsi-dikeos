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


df = pd.read_csv('p005_RTPvsControl zoi.csv')
print(df.shape)
df2 = pd.read_csv('p005_RTPvsCCL4 zoi.csv')
print(df2.shape)
df3 = pd.read_csv('p005_CCL4vsControl zoi.csv')
print(df3.shape)




statsig_df = pd.merge(df, df2, how='outer',
                  #left_on=['Accession']
                #  right_on=['Accession']
                  )




statsig_df = pd.merge(statsig_df, df3, how='outer', 
                #  left_on=['Accession'],
               #   right_on=['Accession']
                  )

print(df.columns)
print(df2.columns)
print(df3.columns)

a = df.columns.tolist() + df2.columns.tolist() + df3.columns.tolist()

duplicates = [i for i in set(a) if a.count(i) > 1]
print(duplicates)


print(statsig_df.columns)

statsig_df.to_csv ('statsig_df zoi.csv')

dfstat = pd.read_csv('differentially expressed all.csv')
print(dfstat.shape)
#number of columns should be 

