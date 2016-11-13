import pandas as pd
import numpy as np
import Models
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV as GSCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import KFold

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

    return X, y1, columns

def optimize(model, params, X, y):
    '''
    input model to check, parameters for model, and X, y matrices
    will return best parameters
    '''
    opt_mod = GSCV(model, params, scoring = make_scorer(accuracy_score), n_jobs = -1)
    opt_mod.fit(X,y)
    best_params = opt_mod.best_params_

    return best_params

def kfold_mlp(df_final):
    '''
    pass in df_final dataframe
    function performs KFold cross validation, fits model
    returns mean accuracy
    '''
    df = df_final.copy()
    y1 = df.pop('home_team_win')
    # variable 1 to predict
    y2 = df.pop('spread')
    # variable 2 to predict
    df.drop(['home_team', 'away_team', 'date'], axis = 1, inplace = True)
    # we don't want any categorical variables in the model
    X = df.values

    kf = KFold(len(X), n_folds = 5, random_state = 2, shuffle = True)
    index = 0
    accuracy = np.empty(5)
    model = MLPClassifier(solver = 'sgd', alpha = 0, random_state = 2, activation = 'logistic', learning_rate = 'constant', tol = 0.0001)
    for train, test in kf:
        scaler = StandardScaler()
        scaler.fit(X[train])
        X[train] = scaler.transform(X[train])
        X[test] = scaler.transform(X[test])
        model.fit(X[train], y1[train])
        pred = model.predict(X[test])
        accuracy[index] = accuracy_score(y1[test], pred)
        index +=1

    return np.mean(accuracy)

def kfold_xgbc(df_final):
    '''
    pass in df_final dataframe
    function performs KFold cross validation, fits model
    returns mean accuracy
    '''
    df = df_final.copy()
    y1 = df.pop('home_team_win')
    # variable 1 to predict
    y2 = df.pop('spread')
    # variable 2 to predict
    df.drop(['home_team', 'away_team', 'date'], axis = 1, inplace = True)
    # we don't want any categorical variables in the model
    X = df.values

    kf = KFold(len(X), n_folds = 5, random_state = 2, shuffle = True)
    index = 0
    accuracy = np.empty(5)
    model = xgb.XGBClassifier()
    for train, test in kf:
        model.fit(X[train], y1[train])
        pred = model.predict(X[test])
        accuracy[index] = accuracy_score(y1[test], pred)
        index +=1

    return np.mean(accuracy)

if __name__ == '__main__':

    # this gets the most recent final csv files
    # if you want up-to-date information, re-run DataCleaning.py
    df_final5 = pd.read_csv('data/final5LS.csv')
    df_final5.drop('Unnamed: 0', axis = 1, inplace = True)

    # get ratio data
    df_final5_r = Models.proportion_data(df_final5)

    # get X, y matrices for 10 games
    X5, y5, columns5 = get_X_y(df_final5)
    X5r, y5r, columns5r = get_X_y(df_final5_r)

    # models to optimize
    # model 1: mlp classifier, 5 games
    mlp = MLPClassifier()
    mlp_params = {'activation': ['identity', 'logistic', 'tanh', 'relu'], \
    'solver': ['lbfgs', 'sgd', 'adam'], \
    'alpha': [0, 0, 00001, 0.0001, 0.001, 0.01, 0.1], \
    'learning_rate': ['constant', 'invscaling', 'adaptive'], \
    # 'max_iter': [100, 200, 500, 1000, 2000, 5000, 10000], \
    'random_state': [2], \
    'tol': [1e-4, 1e-6, 1e-8, 1e-2]}

    # model 2: xgb classifier, 5 games ratio
    # xgbc = xgb.XGBClassifier()
    # xgbc_params = {'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5],  \
    # 'n_estimators': [200, 500, 1000, 1500, 2000], \
    # 'max_depth': [3, 4, 5, 6, 7, 8], \
    # 'min_child_weight': [3, 4, 5, 6, 7, 8], \
    # 'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5], \
    # 'subsample': [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], \
    # 'colsample_bytree': [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], \
    # 'reg_alpha': [0, 0.001, 0.005, 0.01], \
    # 'scale_pos_weight': [0.2, 0.5, 1, 1.5], \
    # 'seed': [2]}

    # optimized models
    # model 1
    params1 = optimize(mlp, mlp_params, X5, y5)
    # put in these optimal parameters into the MLP classifier function above

    # model 2
    # params2 = optimize(xgbc, xgbc_params, X5r, y5r)

    # getting accuracy score
    accuracy1 = kfold_mlp(df_final5)

    # HAD BETTER LUCK OPTIMIZING BY HAND
