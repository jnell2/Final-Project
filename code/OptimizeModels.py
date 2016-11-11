import pandas as pd
import numpy as np
import Models
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV as GSCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

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

def logistic(X_train, X_test, y_train, y_test):
    '''
    pass in the 4 TTS outputs
    will return accuracy, y predicted values, and rmse
    only use this for predicting home team W/L (variable 1 listed below)
    '''
    accuracies = []
    precisions = []
    recalls = []
    model = LogisticRegression(max_iter = 200, n_jobs = -1, random_state = 2, solver = 'sag')
    LR_model = model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    y_true = y_test
    accuracies.append(accuracy_score(y_true, y_predict))
    accuracy = np.average(accuracies)

    return accuracy, y_predict

def random_forest_classifier(X_train, X_test, y_train, y_test):
    '''
    pass in the 4 TTS outputs
    will return accuracy, y predicted values, and features
    only use this for predicting home team W/L (variable 1 listed below)
    '''
    rf = RandomForestClassifier(max_features = None, n_estimators = 100, n_jobs = -1, random_state = 2)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = rf.score(X_test, y_test)
    features = rf.feature_importances_

    return accuracy, y_pred, features

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

    # making new dataframes for 10, 15 games with proportion data
    # only doing this for 5, 10, and 15 games because those seem to perform the best
    df_final5_prop = Models.proportion_data(df_final5)
    df_final10_prop = Models.proportion_data(df_final10)
    df_final15_prop = Models.proportion_data(df_final15)

    # get X, y matrices for 10 games
    X10, y10, columns10 = get_X_y(df_final10)
    # get X, y matrices for 15 games
    X15, y15, columns15 = get_X_y(df_final15)
    # get X, y matrices for 10 games, proportions
    X10p, y10p, columns10p = get_X_y(df_final10_prop)
    # get X, y matrices for 15 games, proportions
    X15p, y15p, columns15p = get_X_y(df_final15_prop)

    # get train_test_split for 10 games
    Xtr110, Xte110, ytr110, yte110, Xtr210, Xte210, ytr210, yte210 = Models.make_train_test(df_final10)
    # get train_test_split for 15 games
    Xtr115, Xte115, ytr115, yte115, Xtr215, Xte215, ytr215, yte215 = Models.make_train_test(df_final15)

    # get train_test_split for 10 games, proportion_data
    X1_train10, X1_test10, y1_train10, y1_test10, X2_train10, X2_test10, y2_train10, y2_test10 = Models.make_train_test(df_final10_prop)
    # get train_test_split for 15 games, proportion_data
    X1_train15, X1_test15, y1_train15, y1_test15, X2_train15, X2_test15, y2_train15, y2_test15 = Models.make_train_test(df_final15_prop)

    # models to optimize
    logreg = LogisticRegression()
    logreg_params = {'max_iter': [150, 200, 250, 300], 'random_state': [2], \
    'solver': ['liblinear', 'lbfgs', 'sag', 'newton-cg'], 'n_jobs': [-1]}

    rfc = RandomForestClassifier()
    rfc_params = {'n_estimators': [10, 20, 40, 60, 80, 100, 150, 200], \
    'max_features': ['sqrt', 'log2', None], 'n_jobs': [-1], 'random_state': [2]}

    mlp = MLPClassifier()
    mlp_params = {'solver': ['lbfgs', 'sgd', 'adam'], 'alpha': [0, 0.0001, 0.001, 0.01, 0.1], \
    'learning_rate': ['constant', 'invscaling', 'adaptive'], 'max_iter': [2000, 5000, 50000], \
    'random_state': [2], 'tol': [1e-4, 1e-6, 1e-8, 1e-2]}

    # optimal parameters found
    # params1 = {max_iter = 200, n_jobs = -1, random_state = 2, solver = 'sag'}
    # params2 = {max_features = None, n_estimators = 100, n_jobs = -1, random_state = 2}

    # optimized models
    # model 1: Logistic Regression with 15 games data
    # opt_params1= optimize(logreg, logreg_params, X15, y15)
    # model 2: Random Forest Classifier with 15 games data
    # opt_params2 = optimize(rfc, rfc_params, X15, y15)
    # model 3: XGboost Classifier with 10 games data
    # this is already optimized
    # model 4: MLP Classifier with 10 game proportional data
    # params4, model4 = optimize(mlp, mlp_params, X10p, y10p)
    # model 5: MLP Classifier with 15 game proportional data
    # params5, model5 = optimize(mlp, mlp_params, X15p, y15p)
    #models 4 and 5 take far too long to run. we'll say these models are optimized

    # getting accuracy scores for 5 models:
    # model 1
    LogReg15_accuracy, LogReg15_predict = logistic(Xtr115, Xte115, ytr115, yte115)
    # model 2
    RFC15_accuracy, RFC15_pred, RFC15_features = random_forest_classifier(Xtr115, Xte115, ytr115, yte115)
    # model 3
    XGB10_preds, XGB10_accuracy = Models.xgboost(Xtr110, Xte110, ytr110, yte110)
    # model 4
    MLP10pred_preds, MLP10pred_accuracy = Models.mlp(X1_train10, X1_test10, y1_train10, y1_test10)
    # model 5
    MLP15pred_preds, MLP15pred_accuracy = Models.mlp(X1_train15, X1_test15, y1_train15, y1_test15)

    # GRID SEARCH DOES NOTHING
