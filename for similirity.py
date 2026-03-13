
import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.tree import export_text
from sklearn import tree

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_curve, auc, accuracy_score
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from GSimPy import data
file = pd.read_csv('aggregated_data.csv')
