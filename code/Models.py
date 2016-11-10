import pandas as pd
import numpy as np
from math import sqrt
# import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y1, test_size = 0.3, random_state = 2)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2, test_size = 0.3, random_state = 2)

    return X_train1, X_test1, y_train1, y_test1, X_train2, X_test2, y_train2, y_test2

def get_X_y(df_final):
    '''
    import final dataframe from DataCleaning that you want to TTS
    will return X, y matrices that will be split by TTS and column names
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

    return X, y1, y2, columns

def rmse(y_true, y_pred):
    '''
    pass in y_true (whichever y_test you want) and y_pred (the predicted values)
    will return root mean square error, which will be used to compare the fit of all the models
    '''
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return rmse

def logistic(X_train, X_test, y_train, y_test):
    '''
    pass in the 4 TTS outputs
    will return accuracy, y predicted values, and rmse
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
    accuracy = np.average(accuracies)
    rmse_score = rmse(y_test, y_predict)

    return accuracy, y_predict, rmse_score

def lasso(X_train, X_test, y_train, y_test):
    '''
    pass in the 4 TTS outputs
    will return y predicted values, rmse, r2
    only use this for predicting spread, variable 2 listed below (in relation to home team)
    '''
    model = Lasso(alpha = 5.0, fit_intercept = False, normalize = True)
    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)
    model_rmse = rmse(y_test, model_pred)
    model_r2 = model.score(X_test, y_test)

    return model_pred, model_rmse, model_r2

def ridge(X_train, X_test, y_train, y_test):
    '''
    pass in the 4 TTS outputs
    will return y predicted values, rmse, r2
    only use this for predicting spread, variable 2 listed below (in relation to home team)
    '''
    model = Ridge(alpha = 0.5, fit_intercept = False, normalize = True)
    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)
    model_rmse = rmse(y_test, model_pred)
    model_r2 = model.score(X_test, y_test)

    return model_pred, model_rmse, model_r2

def elastic(X_train, X_test, y_train, y_test):
    '''
    pass in the 4 TTS outputs
    will return y predicted values, rmse, r2
    only use this for predicting spread, variable 2 listed below (in relation to home team)
    '''
    model = ElasticNet(alpha = 0.5, fit_intercept = False, normalize = True)
    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)
    model_rmse = rmse(y_test, model_pred)
    model_r2 = model.score(X_test, y_test)

    return model_pred, model_rmse, model_r2

def linear(X_train, X_test, y_train, y_test):
    '''
    pass in the 4 TTS outputs
    will return y predicted values, rmse, r2
    only use this for predicting spread, variable 2 listed below (in relation to home team)
    '''
    model = LinearRegression(fit_intercept = False, normalize = True)
    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)
    model_rmse = rmse(y_test, model_pred)
    model_r2 = model.score(X_test, y_test)

    return model_pred, model_rmse, model_r2

def sgd(X_train, X_test, y_train, y_test):
    '''
    pass in the 4 TTS outputs
    will return y predicted values, rmse, r2
    only use this for predicting spread, variable 2 listed below (in relation to home team)
    '''
    model = SGDRegressor(n_iter = 100)
    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)
    model_rmse = rmse(y_test, model_pred)
    model_r2 = model.score(X_test, y_test)

    return model_pred, model_rmse, model_r2

def random_forest_classifier(X_train, X_test, y_train, y_test):
    '''
    pass in the 4 TTS outputs
    will return accuracy, y predicted values, rmse, and features
    only use this for predicting home team W/L (variable 1 listed below)
    '''
    rf = RandomForestClassifier(random_state = 2)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = rf.score(X_test, y_test)
    features = rf.feature_importances_
    rmse_score = rmse(y_test, y_pred)

    return accuracy, y_pred, rmse_score, features

def random_forest_regressor(X_train, X_test, y_train, y_test):
    '''
    pass in the 4 TTS outputs
    will return r2, y predicted values, rmse, and features
    only use this for predicting spread, variable 2 listed below (in relation to home team)
    '''
    rf = RandomForestRegressor(random_state = 2)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    r2 = rf.score(X_test, y_test)
    features = rf.feature_importances_
    rmse_score = rmse(y_test, y_pred)

    return r2, y_pred, rmse_score, features

# def xgboost(X_train, X_test, y_train, y_test):
#     '''
#     pass in the 4 TTS
#     will return predictions, rmse, and accuracy
#     '''
#     model = xgboost.XGBClassifier()
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     predictions = [round(value) for value in y_pred]
#     accuracy = accuracy_score(y_test, predictions)
#     rmse_score = rmse(y_test, predictions)
#
#     return predictions, rmse_score, accuracy

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
    Xtr12, Xte12, ytr12, yte12, Xtr22, Xte22, ytr22, yte22 = make_train_test(df_final2)
    # get train_test_split for 5 games
    Xtr15, Xte15, ytr15, yte15, Xtr25, Xte25, ytr25, yte25 = make_train_test(df_final5)
    # get train_test_split for 10 games
    Xtr110, Xte110, ytr110, yte110, Xtr210, Xte210, ytr210, yte210 = make_train_test(df_final10)
    # get train_test_split for 15 games
    Xtr115, Xte115, ytr115, yte115, Xtr215, Xte215, ytr215, yte215 = make_train_test(df_final15)
    # get train_test_split for 20 games
    Xtr120, Xte120, ytr120, yte120, Xtr220, Xte220, ytr220, yte220 = make_train_test(df_final20)

    # get X, y matrices for 2 games
    X_2, y1_2, y2_2, columns2 = get_X_y(df_final2)
    # get X, y matrices for 5 games
    X_5, y1_5, y2_5, columns5 = get_X_y(df_final5)
    # get X, y matrices for 10 games
    X_10, y1_10, y2_10, column10 = get_X_y(df_final10)
    # get X, y matrices for 15 games
    X_15, y1_15, y2_15, column15 = get_X_y(df_final15)
    # get X, y matrices for 20 games
    X_20, y1_20, y2_20, column20 = get_X_y(df_final20)

    # Logistic Regression results
    LogReg2_accuracy, LogReg2_predict, LogReg2_rmse = logistic(Xtr12, Xte12, ytr12, yte12)
    LogReg5_accuracy, LogReg5_predict, LogReg5_rmse = logistic(Xtr15, Xte15, ytr15, yte15)
    LogReg10_accuracy, LogReg10_predict, LogReg10_rmse = logistic(Xtr110, Xte110, ytr110, yte110)
    LogReg15_accuracy, LogReg15_predict, LogReg15_rmse = logistic(Xtr115, Xte115, ytr115, yte115)
    LogReg20_accuracy, LogReg20_predict, LogReg20_rmse = logistic(Xtr120, Xte120, ytr120, yte120)

    # Linear Regression results
    # Lasso
    Lasso2_pred, Lasso2_rmse, Lasso2_r2 = lasso(Xtr22, Xte22, ytr22, yte22)
    Lasso5_pred, Lasso5_rmse, Lasso5_r2 = lasso(Xtr25, Xte25, ytr25, yte25)
    Lasso10_pred, Lasso10_rmse, Lasso10_r2 = lasso(Xtr210, Xte210, ytr210, yte210)
    Lasso15_pred, Lasso15_rmse, Lasso15_r2 = lasso(Xtr215, Xte215, ytr215, yte215)
    Lasso20_pred, Lasso20_rmse, Lasso20_r2 = lasso(Xtr220, Xte220, ytr220, yte220)
    # Ridge
    Ridge2_pred, Ridge2_rmse, Ridge2_r2 = ridge(Xtr22, Xte22, ytr22, yte22)
    Ridge5_pred, Ridge5_rmse, Ridge5_r2 = ridge(Xtr25, Xte25, ytr25, yte25)
    Ridge10_pred, Ridge10_rmse, Ridge10_r2 = ridge(Xtr210, Xte210, ytr210, yte210)
    Ridge15_pred, Ridge15_rmse, Ridge15_r2 = ridge(Xtr215, Xte215, ytr215, yte215)
    Ridge20_pred, Ridge20_rmse, Ridge20_r2 = ridge(Xtr220, Xte220, ytr220, yte220)
    # Elastic Net
    EN2_pred, EN2_rmse, EN2_r2 = elastic(Xtr22, Xte22, ytr22, yte22)
    EN5_pred, EN5_rmse, EN5_r2 = elastic(Xtr25, Xte25, ytr25, yte25)
    EN10_pred, EN10_rmse, EN10_r2 = elastic(Xtr210, Xte210, ytr210, yte210)
    EN15_pred, EN15_rmse, EN15_r2 = elastic(Xtr215, Xte215, ytr215, yte215)
    EN20_pred, EN20_rmse, EN20_r2 = elastic(Xtr220, Xte220, ytr220, yte220)
    # Linear Regression
    LR2_pred, LR2_rmse, LR2_r2 = linear(Xtr22, Xte22, ytr22, yte22)
    LR5_pred, LR5_rmse, LR5_r2 = linear(Xtr25, Xte25, ytr25, yte25)
    LR10_pred, LR10_rmse, LR10_r2 = linear(Xtr210, Xte210, ytr210, yte210)
    LR15_pred, LR15_rmse, LR15_r2 = linear(Xtr215, Xte215, ytr215, yte215)
    LR20_pred, LR20_rmse, LR20_r2 = linear(Xtr220, Xte220, ytr220, yte220)
    # SGD Regressor
    SGD2_pred, SGD2_rmse, SGD2_r2 = sgd(Xtr22, Xte22, ytr22, yte22)
    SGD5_pred, SGD5_rmse, SGD5_r2 = sgd(Xtr25, Xte25, ytr25, yte25)
    SGD10_pred, SGD10_rmse, SGD10_r2 = sgd(Xtr210, Xte210, ytr210, yte210)
    SGD15_pred, SGD15_rmse, SGD15_r2 = sgd(Xtr215, Xte215, ytr215, yte215)
    SGD20_pred, SGD20_rmse, SGD20_r2 = sgd(Xtr220, Xte220, ytr220, yte220)

    # Random Forest Classifier results
    RFC2_accuracy, RFC2_predict, RFC2_rmse, RFC2_features = random_forest_classifier(Xtr12, Xte12, ytr12, yte12)
    RFC5_accuracy, RFC5_predict, RFC5_rmse, RFC5_features = random_forest_classifier(Xtr15, Xte15, ytr15, yte15)
    RFC10_accuracy, RFC10_predict, RFC10_rmse, RFC10_features = random_forest_classifier(Xtr110, Xte110, ytr110, yte110)
    RFC15_accuracy, RFC15_predict, RFC15_rmse, RFC15_features = random_forest_classifier(Xtr115, Xte115, ytr115, yte115)
    RFC20_accuracy, RFC20_predict, RFC20_rmse, RFC20_features = random_forest_classifier(Xtr120, Xte120, ytr120, yte120)

    # Random Forest Regressor results
    RFR2_r2, RFR2_predict, RFR2_rmse, RFR2_features = random_forest_regressor(Xtr22, Xte22, ytr22, yte22)
    RFR5_r2, RFR5_predict, RFR5_rmse, RFR5_features = random_forest_regressor(Xtr25, Xte25, ytr25, yte25)
    RFR10_r2, RFR10_predict, RFR10_rmse, RFR10_features = random_forest_regressor(Xtr210, Xte210, ytr210, yte210)
    RFR15_r2, RFR15_predict, RFR15_rmse, RFR15_features = random_forest_regressor(Xtr215, Xte215, ytr215, yte215)
    RFR20_r2, RFR20_predict, RFR20_rmse, RFR20_features = random_forest_regressor(Xtr220, Xte220, ytr220, yte220)

    # # XG Boost
    # XGB2_preds, XGB2_rmse, XGB2_accuracy = xgboost(Xtr12, Xte12, ytr12, yte12)
    # XGB5_preds, XGB5_rmse, XGB5_accuracy = xgboost(Xtr15, Xte15, ytr15, yte15)
    # XGB10_preds, XGB10_rmse, XGB10_accuracy = xgboost(Xtr110, Xte110, ytr110, yte110)
    # XGB15_preds, XGB15_rmse, XGB15_accuracy = xgboost(Xtr115, Xte115, ytr115, yte115)
    # XGB20_preds, XGB20_rmse, XGB20_accuracy = xgboost(Xtr120, Xte120, ytr120, yte120)
