import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
try:
    from sklearn.model_selection import train_test_split
except:
    from sklearn.cross_validation import train_test_split

def make_train_test(df_final):
    '''
    import final dataframe from DataCleaning that you want to TTS
    will return two sets of train_test_split
    1) using home team W/L as variable trying to predict
    2) using spread as variable trying to predict
    '''
    df = df_final.copy()
    y1 = df.pop('home_team_win')
    # variable 1 to predict
    y2 = df.pop('spread')
    # variable 2 to predict
    df.drop(['home_team', 'away_team', 'date'], axis = 1, inplace = True)
    # we don't want any categorical variables in the model
    X = df.values
    columns = df.columns

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y1, test_size = 0.3, random_state = 2)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y1, test_size = 0.3, random_state = 2)

    return X_train1, X_test1, y_train1, y_test1, X_train2, X_test2, y_train2, y_test2, columns

def logistic(X_train, X_test, y_train, y_test):
    '''
    input the 4 TTS outputs
    will return accuracy, precision, and recall
    only use this for predicting home team W/L (variable 1 listed below)
    '''
    accuracies = []
    precisions = []
    recalls = []
    model = LogisticRegression()
    LR_model = model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    y_true = y_test
    accuracies.append(accuracy_score(y_true, y_predict))
    precisions.append(precision_score(y_true, y_predict))
    recalls.append(recall_score(y_true, y_predict))
    accuracy = np.average(accuracies)
    precision = np.average(precisions)
    recall = np.average(recalls)

    return accuracy, precision, recall

if __name__ == '__main__':

    # this gets the most recent final csv files
    # if you want up-to-date information, re-run DataCleaning.py
    df_final2 = pd.read_csv('data/final2.csv')
    df_final2.drop('Unnamed: 0', axis = 1, inplace = True)
    df_final5 = pd.read_csv('data/final5.csv')
    df_final5.drop('Unnamed: 0', axis = 1, inplace = True)
    df_final10 = pd.read_csv('data/final10.csv')
    df_final10.drop('Unnamed: 0', axis = 1, inplace = True)
    df_final15 = pd.read_csv('data/final15.csv')
    df_final15.drop('Unnamed: 0', axis = 1, inplace = True)
    df_final20 = pd.read_csv('data/final20.csv')
    df_final20.drop('Unnamed: 0', axis = 1, inplace = True)

    # the goal is to predict 2 variables:
    # 1) home team W/L
    # 2) spread

    # get train_test_split for 2 games
    Xtr12, Xte12, ytr12, yte12, Xtr22, Xte22, ytr22, yte22, columns2 \
    = make_train_test(df_final2)
    # get train_test_split for 5 games
    Xtr15, Xte15, ytr15, yte15, Xtr25, Xte25, ytr25, yte25, columns5 \
    = make_train_test(df_final5)
    # get train_test_split for 10 games
    Xtr110, Xte110, ytr110, yte110, Xtr210, Xte210, ytr210, yte210, columns10 \
    = make_train_test(df_final10)
    # get train_test_split for 15 games
    Xtr115, Xte115, ytr115, yte115, Xtr215, Xte215, ytr215, yte215, columns15 \
    = make_train_test(df_final15)
    # get train_test_split for 20 games
    Xtr120, Xte120, ytr120, yte120, Xtr220, Xte220, ytr220, yte220, columns20 \
    = make_train_test(df_final20)

    LogReg2_accuracy, LogReg2_precision, LogReg2_recall = logistic(Xtr12, Xte12, ytr12, yte12)
    LogReg5_accuracy, LogReg5_precision, LogReg5_recall = logistic(Xtr15, Xte15, ytr15, yte15)
    LogReg10_accuracy, LogReg10_precision, LogReg10_recall = logistic(Xtr110, Xte110, ytr110, yte110)
    LogReg15_accuracy, LogReg15_precision, LogReg15_recall = logistic(Xtr115, Xte115, ytr115, yte115)
    LogReg20_accuracy, LogReg20_precision, LogReg20_recall = logistic(Xtr120, Xte120, ytr120, yte120)
