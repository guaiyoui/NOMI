import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import scipy.sparse
from scipy.stats import zscore
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from scipy.spatial.distance import cosine

def conf_matrix(y_test, pred_test):    
    
    # Creating a confusion matrix
    con_mat = confusion_matrix(y_test, pred_test)
    con_mat = pd.DataFrame(con_mat, range(2), range(2))
   
    #Ploting the confusion matrix
    plt.figure(figsize=(6,6))
    sns.set(font_scale=1.5) 
    sns.heatmap(con_mat, annot=True, annot_kws={"size": 16}, fmt='g', cmap='Blues', cbar=False)

# def preparation(data, categorical_variables, numerical_variables, target_column, splitting_ratio):
  
#     X = data.loc[:, originaldata.columns!=target_column]
#     t = data[target_column]

#     categorical_columns = categorical_variables
#     numerical_columns = numerical_variables

#     categorical_preprocessor = OneHotEncoder(handle_unknown="ignore") # processors for categorical columns
#     numerical_preprocessor = StandardScaler() # processors for numerical columns

#     preprocessor = ColumnTransformer([
#         ('one-hot-encoder', categorical_preprocessor, categorical_columns), # trasformer for categorical columns
#         ('standard_scaler', numerical_preprocessor, numerical_columns)]) # transformer for numerical columns

#     X = pd.DataFrame.sparse.from_spmatrix(preprocessor.fit_transform(X))   

#     X = pd.DataFrame(zscore(X.values))
#     X, t = shuffle(X, t, random_state=0)

#     X_train, X_test, y_train, y_test = train_test_split(X, t, test_size=splitting_ratio, random_state=0)
#     return X_train, X_test, y_train, y_test

def preparation(data, target_column, splitting_ratio):

    t = data[:, target_column]
    X = np.delete(data, target_column, 1)

    # X = pd.DataFrame(zscore(X.values))
    X, t = shuffle(X, t, random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(X, t, test_size=splitting_ratio, random_state=0)
    return X_train, X_test, y_train, y_test

def logistic_regression_classifier(X_tr, X_te, y_tr, y_te):
    log_classifier = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced') # regularization is applied as default
    log_classifier.fit(X_tr, y_tr)
    y_pred = log_classifier.predict(X_te)
    # print(y_te[0:5], y_pred[0:5])
    metrics = {
        'accuracy': accuracy_score(y_te, y_pred),
        'precision_greater': precision_score(y_te, y_pred, average='micro'),
        'precision_smaller': precision_score(y_te, y_pred, average='micro'),
        'recall_greater': recall_score(y_te, y_pred, average='micro'),
        'recall_smaller': recall_score(y_te, y_pred, average='micro'),
        'f1_greater': f1_score(y_te, y_pred, average='micro'),
        'f1_smaller': f1_score(y_te, y_pred, average='micro'),
    }

    print("\nMetrics:\n")
    print(metrics)
    print(classification_report(y_te, y_pred))
    print("\nAccuracy: ", accuracy_score(y_te, y_pred), "\n\n")
    print("\nConfusion Matrix:\n")
    # conf_matrix(y_te, y_pred)

    return metrics

def SVM_classifier(X_tr, X_te, y_tr, y_te, kernel):
    SVM_classifier = svm.SVC(C=1.0, random_state=1, kernel=kernel, class_weight="balanced")
    SVM_classifier.fit(X_tr, y_tr)
    y_pred = SVM_classifier.predict(X_te)
    
    metrics = {
        'accuracy': accuracy_score(y_te, y_pred),
        'precision_greater': precision_score(y_te, y_pred, average='macro'),
        'precision_smaller': precision_score(y_te, y_pred, average='macro'),
        'recall_greater': recall_score(y_te, y_pred, average='macro'),
        'recall_smaller': recall_score(y_te, y_pred, average='macro'),
        'f1_greater': f1_score(y_te, y_pred, average='macro'),
        'f1_smaller': f1_score(y_te, y_pred, average='macro'),
    }

    print("\nMetrics:\n")
    print(classification_report(y_te, y_pred))
    print("\nAccuracy: ", accuracy_score(y_te, y_pred), "\n\n")
    # print("\nConfusion Matrix:\n")
    # conf_matrix(y_te, y_pred)

    return metrics



originaldata = pd.read_csv("./data/wine.csv", index_col=False)

df = pd.DataFrame(originaldata)
# 统计每一列的唯一值数量
unique_counts = df.nunique()
print(unique_counts, df.head())

X_train, X_test, y_train, y_test = preparation(originaldata.values, target_column=0, splitting_ratio=0.2)
metrics_lr_original = logistic_regression_classifier(X_train, X_test, y_train, y_test)

metrics_svm_original = SVM_classifier(X_train, X_test, y_train, y_test, kernel="linear")


# categorical_columns = ["workclass","education","maritial-status","occupation","relationship","race","sex","native-country"]
# numerical_columns = ["age","education-num","hours-per-week","capital-gain","capital-loss"]

# X_train, X_test, y_train, y_test = preparation(originaldata, categorical_columns, numerical_columns, "income", 0.2)
     
