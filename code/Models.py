import pandas as pd
import numpy as np
from math import sqrt
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
    columns = df.columns

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y1, test_size = 0.3, random_state = 2)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2, test_size = 0.3, random_state = 2)

    return X_train1, X_test1, y_train1, y_test1, X_train2, X_test2, y_train2, y_test2, columns

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

def linear(X_train, X_test, y_train, y_test):
    '''
    pass in the 4 TTS outputs
    will return rmse and y predicted values for all linear regression models
    only use this for predicting spread, variable 2 listed below (in relation to home team)
    '''
    lasso = Lasso(alpha = 5.0, fit_intercept = False, normalize = True)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    lasso_rmse = rmse(y_test, lasso_pred)
    lasso_r2 = lasso.score(X_test, y_test)

    ridge = Ridge(alpha = 0.5, fit_intercept = False, normalize = True)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    ridge_rmse = rmse(y_test, ridge_pred)
    ridge_r2 = ridge.score(X_test, y_test)

    elastic = ElasticNet(alpha = 0.5, l1_ratio = 0.8, fit_intercept = False, normalize = True)
    elastic.fit(X_train, y_train)
    elastic_pred = elastic.predict(X_test)
    elastic_rmse = rmse(y_test, elastic_pred)
    elastic_r2 = elastic.score(X_test, y_test)

    lr = LinearRegression(fit_intercept = False, normalize = True)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_rmse = rmse(y_test, lr_pred)
    lr_r2 = lr.score(X_test, y_test)

    sgd = SGDRegressor(n_iter = 100)
    sgd.fit(X_train, y_train)
    sgd_pred = sgd.predict(X_test)
    sgd_rmse = rmse(y_test, sgd_pred)
    sgd_r2 = sgd.score(X_test, y_test)

    return lasso_rmse, lasso_pred, ridge_rmse, ridge_pred, elastic_rmse, \
    elastic_pred, lr_rmse, lr_pred, sgd_rmse, sgd_pred, \
    lasso_r2, ridge_r2, elastic_r2, lr_r2, sgd_r2

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

    # Logistic Regression results
    LogReg2_accuracy, LogReg2_predict, LogReg2_rmse = logistic(Xtr12, Xte12, ytr12, yte12)
    LogReg5_accuracy, LogReg5_predict, LogReg5_rmse = logistic(Xtr15, Xte15, ytr15, yte15)
    LogReg10_accuracy, LogReg10_predict, LogReg10_rmse = logistic(Xtr110, Xte110, ytr110, yte110)
    LogReg15_accuracy, LogReg15_predict, LogReg15_rmse = logistic(Xtr115, Xte115, ytr115, yte115)
    LogReg20_accuracy, LogReg20_predict, LogReg20_rmse = logistic(Xtr120, Xte120, ytr120, yte120)

    # Linear Regression results
    lasso_rmse2, lasso_pred2, ridge_rmse2, ridge_pred2, elastic_rmse2, \
    elastic_pred2, lr_rmse2, lr_pred2, sgd_rmse2, sgd_pred2, \
    lasso_r22, ridge_r22, elastic_r22, lr_r22, sgd_r22 = linear(Xtr22, Xte22, ytr22, yte22)
    lasso_rmse5, lasso_pred5, ridge_rmse5, ridge_pred5, elastic_rmse5, \
    elastic_pred5, lr_rmse5, lr_pred5, sgd_rmse5, sgd_pred5, \
    lasso_r25, ridge_r25, elastic_r25, lr_r25, sgd_r25 = linear(Xtr25, Xte25, ytr25, yte25)
    lasso_rmse10, lasso_pred10, ridge_rmse10, ridge_pred10, elastic_rmse10, \
    elastic_pred10, lr_rmse10, lr_pred10, sgd_rmse10, sgd_pred10, \
    lasso_r210, ridge_r210, elastic_r210, lr_r210, sgd_r210 = linear(Xtr210, Xte210, ytr210, yte210)
    lasso_rmse15, lasso_pred15, ridge_rmse15, ridge_pred15, elastic_rmse15, \
    elastic_pred15, lr_rmse15, lr_pred15, sgd_rmse15, sgd_pred15, \
    lasso_r215, ridge_r215, elastic_r215, lr_r215, sgd_r215 = linear(Xtr215, Xte215, ytr215, yte215)
    lasso_rmse20, lasso_pred20, ridge_rmse20, ridge_pred20, elastic_rmse20, \
    elastic_pred20, lr_rmse20, lr_pred20, sgd_rmse20, sgd_pred20, \
    lasso_r220, ridge_r220, elastic_r220, lr_r220, sgd_r220 = linear(Xtr220, Xte220, ytr220, yte220)

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
