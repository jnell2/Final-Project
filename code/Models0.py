import pandas as pd
import numpy as np
from math import sqrt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cross_validation import KFold

def proportion_data(df_final):
    '''
    pass in ifnal dataframe from DataCleaning that you want to transform
    will return a new dataframe with proportional data (home team stat / away team stat)
    currently, this function will only return dataset for W/L as variable trying to predict
    '''
    df= df_final.copy()
    df['spread'] = (df['home_spread'] + 1e-4)/(df['away_spread'] + df['home_spread'] + 1e-4)
    df['wins'] = (df['home_wins'] + 1e-4)/(df['away_wins'] + df['home_wins'] + 1e-4)
    df['goals'] = (df['home_goals'] + 1e-4)/(df['away_goals'] + df['home_goals'] + 1e-4)
    df['shots'] = (df['home_shots'] + 1e-4)/(df['away_shots'] + df['home_shots'] + 1e-4)
    df['PP%'] = (df['home_PP%'] + 1e-4)/(df['away_PP%'] + df['home_PP%'] + 1e-4)
    df['FOW%'] = (df['home_FOW%'] + 1e-4)/(df['away_FOW%'] + df['home_FOW%'] + 1e-4)
    df['PIM'] = (df['home_PIM'] + 1e-4)/(df['away_PIM'] + df['home_PIM'] + 1e-4)
    df['hits'] = (df['home_hits'] + 1e-4)/(df['away_hits'] + df['home_hits'] + 1e-4)
    df['blocked'] = (df['home_blocked'] + 1e-4)/(df['away_blocked'] + df['home_blocked'] + 1e-4)
    df['giveaways'] = (df['home_giveaways'] + 1e-4)/(df['away_giveaways'] + df['home_giveaways'] + 1e-4)
    df['takeaways'] = (df['home_takeaways'] + 1e-4)/(df['away_takeaways'] + df['home_takeaways'] + 1e-4)
    df['save%'] = (df['home_save%'] + 1e-4)/(df['away_save%'] + df['home_save%'] + 1e-4)
    df['shot%'] = (df['home_shot%'] + 1e-4)/(df['away_shot%'] + df['home_shot%'] + 1e-4)
    df['PDO'] = (df['home_PDO'] + 1e-4)/(df['away_PDO'] + df['home_PDO'] + 1e-4)
    df['corsi'] = (df['home_corsi'] + 1e-4)/(df['away_corsi'] + df['home_corsi'] + 1e-4)
    df.drop(['home_spread', 'home_wins', 'home_goals', 'home_shots', 'home_PP%', \
    'home_FOW%', 'home_PIM', 'home_hits', 'home_blocked', 'home_giveaways', \
    'home_takeaways', 'home_save%', 'home_shot%', 'home_PDO', 'home_corsi', \
    'away_spread', 'away_wins', 'away_goals', 'away_shots', 'away_PP%', \
    'away_FOW%', 'away_PIM', 'away_hits', 'away_blocked', 'away_giveaways', \
    'away_takeaways', 'away_save%', 'away_shot%', 'away_PDO', 'away_corsi'], \
    axis = 1, inplace = True)

    return df

def drop_variables(df_final):
    '''
    drop variables to test accuracies
    '''
    df_final.drop(['home_giveaways', 'away_giveaways'], axis = 1, inplace = True)

    return df_final

def drop_variables_ratio(df_final):
    '''
    drop variables to test accuracies
    '''
    df_final.drop(['corsi'], axis = 1, inplace = True)

    return df_final

def kfold_logistic(df_final):
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
    model = LogisticRegression()
    for train, test in kf:
        model.fit(X[train], y1[train])
        pred = model.predict(X[test])
        accuracy[index] = accuracy_score(y1[test], pred)
        index +=1

    return np.mean(accuracy)

def kfold_lasso(df_final):
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
    model = Lasso(alpha = 0.0001, fit_intercept = False, normalize = True)
    for train, test in kf:
        model.fit(X[train], y2[train])
        pred = model.predict(X[test])
        pred = map(lambda x: 1 if x > 0 else 0, pred)
        accuracy[index] = accuracy_score(y1[test], pred)
        index +=1

    return np.mean(accuracy)

def kfold_ridge(df_final):
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
    model = Ridge(alpha = 0.0001, fit_intercept = False, normalize = True)
    for train, test in kf:
        model.fit(X[train], y2[train])
        pred = model.predict(X[test])
        pred = map(lambda x: 1 if x > 0 else 0, pred)
        accuracy[index] = accuracy_score(y1[test], pred)
        index +=1

    return np.mean(accuracy)

def kfold_elastic(df_final):
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
    model = ElasticNet(alpha = 0.0001, fit_intercept = False, normalize = True)
    for train, test in kf:
        model.fit(X[train], y2[train])
        pred = model.predict(X[test])
        pred = map(lambda x: 1 if x > 0 else 0, pred)
        accuracy[index] = accuracy_score(y1[test], pred)
        index +=1

    return np.mean(accuracy)

def kfold_linear(df_final):
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
    model = LinearRegression(n_jobs = -1, fit_intercept = False, normalize = True)
    for train, test in kf:
        model.fit(X[train], y2[train])
        pred = model.predict(X[test])
        pred = map(lambda x: 1 if x > 0 else 0, pred)
        accuracy[index] = accuracy_score(y1[test], pred)
        index +=1

    return np.mean(accuracy)

def kfold_sgd(df_final):
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
    model = SGDRegressor(n_iter = 100)
    for train, test in kf:
        model.fit(X[train], y2[train])
        pred = model.predict(X[test])
        pred = map(lambda x: 1 if x > 0 else 0, pred)
        accuracy[index] = accuracy_score(y1[test], pred)
        index +=1

    return np.mean(accuracy)

def kfold_rfc(df_final):
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
    model = RandomForestClassifier(random_state = 2)
    for train, test in kf:
        model.fit(X[train], y1[train])
        pred = model.predict(X[test])
        accuracy[index] = accuracy_score(y1[test], pred)
        index +=1

    return np.mean(accuracy)

def kfold_rfr(df_final):
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
    model = RandomForestRegressor(random_state = 2)
    for train, test in kf:
        model.fit(X[train], y2[train])
        pred = model.predict(X[test])
        pred = map(lambda x: 1 if x > 0 else 0, pred)
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

def kfold_xgbr(df_final):
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
    model = xgb.XGBRegressor()
    for train, test in kf:
        model.fit(X[train], y2[train])
        pred = model.predict(X[test])
        pred = map(lambda x: 1 if x > 0 else 0, pred)
        accuracy[index] = accuracy_score(y1[test], pred)
        index +=1

    return np.mean(accuracy)

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
    model = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (5,2), random_state = 2)
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

def kfold_gbc(df_final):
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
    model = GradientBoostingClassifier()
    for train, test in kf:
        model.fit(X[train], y1[train])
        pred = model.predict(X[test])
        accuracy[index] = accuracy_score(y1[test], pred)
        index +=1

    return np.mean(accuracy)

if __name__ == '__main__':

    # this gets the most recent final csv files
    # if you want up-to-date information, re-run DataCleaning.py
    df_final2 = pd.read_csv('data/final2LS.csv')
    df_final2.drop(['Unnamed: 0'], axis = 1, inplace = True)
    df_final5 = pd.read_csv('data/final5LS.csv')
    df_final5.drop(['Unnamed: 0'], axis = 1, inplace = True)
    df_final10 = pd.read_csv('data/final10LS.csv')
    df_final10.drop(['Unnamed: 0'], axis = 1, inplace = True)
    df_final15 = pd.read_csv('data/final15LS.csv')
    df_final15.drop(['Unnamed: 0'], axis = 1, inplace = True)

    # making new dataframes for 10, 15 games with proportion data
    # only doing this for 5, 10, and 15 games because those seem to perform the best
    df_final5_r = proportion_data(df_final5)
    df_final10_r = proportion_data(df_final10)
    df_final15_r = proportion_data(df_final15)

    # drop variables
    df_final2 = drop_variables(df_final2)
    df_final5 = drop_variables(df_final5)
    df_final10 = drop_variables(df_final10)
    df_final15 = drop_variables(df_final15)
    df_final5_r = drop_variables_ratio(df_final5_r)
    df_final10_r = drop_variables_ratio(df_final10_r)
    df_final15_r = drop_variables_ratio(df_final15_r)

    # the goal is to predict 2 variables:
    # 1) home team W/L
    # 2) spread

    # Logistic Regression results
    logistic2_accuracy = kfold_logistic(df_final2)
    logistic5_accuracy = kfold_logistic(df_final5)
    logistic10_accuracy = kfold_logistic(df_final10)
    logistic15_accuracy = kfold_logistic(df_final15)
    logistic5_r_accuracy = kfold_logistic(df_final5_r)
    logistic10_r_accuracy = kfold_logistic(df_final10_r)
    logistic15_r_accuracy = kfold_logistic(df_final15_r)

    # Linear Regression results
    # Lasso
    lasso2_accuracy = kfold_lasso(df_final2)
    lasso5_accuracy = kfold_lasso(df_final5)
    lasso10_accuracy = kfold_lasso(df_final10)
    lasso15_accuracy = kfold_lasso(df_final15)
    lasso5_r_accuracy = kfold_lasso(df_final5_r)
    lasso10_r_accuracy = kfold_lasso(df_final10_r)
    lasso15_r_accuracy = kfold_lasso(df_final15_r)
    # Ridge
    ridge2_accuracy = kfold_ridge(df_final2)
    ridge5_accuracy = kfold_ridge(df_final5)
    ridge10_accuracy = kfold_ridge(df_final10)
    ridge15_accuracy = kfold_ridge(df_final15)
    ridge5_r_accuracy = kfold_ridge(df_final5_r)
    ridge10_r_accuracy = kfold_ridge(df_final10_r)
    ridge15_r_accuracy = kfold_ridge(df_final15_r)
    # Elastic Net
    elastic2_accuracy = kfold_elastic(df_final2)
    elastic5_accuracy = kfold_elastic(df_final5)
    elastic10_accuracy = kfold_elastic(df_final10)
    elastic15_accuracy = kfold_elastic(df_final15)
    elastic5_r_accuracy = kfold_elastic(df_final5_r)
    elastic10_r_accuracy = kfold_elastic(df_final10_r)
    elastic15_r_accuracy = kfold_elastic(df_final15_r)
    # Linear Regression
    linear2_accuracy = kfold_linear(df_final2)
    linear5_accuracy = kfold_linear(df_final5)
    linear10_accuracy = kfold_linear(df_final10)
    linear15_accuracy = kfold_linear(df_final15)
    linear5_r_accuracy = kfold_linear(df_final5_r)
    linear10_r_accuracy = kfold_linear(df_final10_r)
    linear15_r_accuracy = kfold_linear(df_final15_r)
    # SGD Regressor
    sgd2_accuracy = kfold_sgd(df_final2)
    sgd5_accuracy = kfold_sgd(df_final5)
    sgd10_accuracy = kfold_sgd(df_final10)
    sgd15_accuracy = kfold_sgd(df_final15)
    sgd5_r_accuracy = kfold_sgd(df_final5_r)
    sgd10_r_accuracy = kfold_sgd(df_final10_r)
    sgd15_r_accuracy = kfold_sgd(df_final15_r)

    # Random Forest Classifier results
    rfc2_accuracy = kfold_rfc(df_final2)
    rfc5_accuracy = kfold_rfc(df_final5)
    rfc10_accuracy = kfold_rfc(df_final10)
    rfc15_accuracy = kfold_rfc(df_final15)
    rfc5_r_accuracy = kfold_rfc(df_final5_r)
    rfc10_r_accuracy = kfold_rfc(df_final10_r)
    rfc15_r_accuracy = kfold_rfc(df_final15_r)
    # Random Forest Regressor results
    rfr2_accuracy = kfold_rfr(df_final2)
    rfr5_accuracy = kfold_rfr(df_final5)
    rfr10_accuracy = kfold_rfr(df_final10)
    rfr15_accuracy = kfold_rfr(df_final15)
    rfr5_r_accuracy = kfold_rfr(df_final5_r)
    rfr10_r_accuracy = kfold_rfr(df_final10_r)
    rfr15_r_accuracy = kfold_rfr(df_final15_r)

    # XG Boost Classifier results
    xgbc2_accuracy = kfold_xgbc(df_final2)
    xgbc5_accuracy = kfold_xgbc(df_final5)
    xgbc10_accuracy = kfold_xgbc(df_final10)
    xgbc15_accuracy = kfold_xgbc(df_final15)
    xgbc5_r_accuracy = kfold_xgbc(df_final5_r)
    xgbc10_r_accuracy = kfold_xgbc(df_final10_r)
    xgbc15_r_accuracy = kfold_xgbc(df_final15_r)
    # XG Boost Regressor results
    xgbr2_accuracy = kfold_xgbr(df_final2)
    xgbr5_accuracy = kfold_xgbr(df_final5)
    xgbr10_accuracy = kfold_xgbr(df_final10)
    xgbr15_accuracy = kfold_xgbr(df_final15)
    xgbr5_r_accuracy = kfold_xgbr(df_final5_r)
    xgbr10_r_accuracy = kfold_xgbr(df_final10_r)
    xgbr15_r_accuracy = kfold_xgbr(df_final15_r)

    # MLP results
    mlp2_accuracy = kfold_mlp(df_final2)
    mlp5_accuracy = kfold_mlp(df_final5)
    mlp10_accuracy = kfold_mlp(df_final10)
    mlp15_accuracy = kfold_mlp(df_final15)
    mlp5_r_accuracy = kfold_mlp(df_final5_r)
    mlp10_r_accuracy = kfold_mlp(df_final10_r)
    mlp15_r_accuracy = kfold_mlp(df_final15_r)

    # Gradient Boosting Classifier results
    gbc2_accuracy = kfold_gbc(df_final2)
    gbc5_accuracy = kfold_gbc(df_final5)
    gbc10_accuracy = kfold_gbc(df_final10)
    gbc15_accuracy = kfold_gbc(df_final15)
    gbc5_r_accuracy = kfold_gbc(df_final5_r)
    gbc10_r_accuracy = kfold_gbc(df_final10_r)
    gbc15_r_accuracy = kfold_gbc(df_final15_r)

    print 'logistic 2'
    print logistic2_accuracy
    print 'logistic 5'
    print logistic5_accuracy
    print 'logistic 10'
    print logistic10_accuracy
    print 'logistic 15'
    print logistic15_accuracy
    print 'logistic 5r'
    print logistic5_r_accuracy
    print 'logistic 10r'
    print logistic10_r_accuracy
    print 'logistic 15r'
    print logistic15_r_accuracy
    print 'lasso 2'
    print lasso2_accuracy
    print 'lasso 5'
    print lasso5_accuracy
    print 'lasso 10'
    print lasso10_accuracy
    print 'lasso 15'
    print lasso15_accuracy
    print 'lasso 5r'
    print lasso5_r_accuracy
    print 'lasso 10r'
    print lasso10_r_accuracy
    print 'lasso 15r'
    print lasso15_r_accuracy
    print 'ridge 2'
    print ridge2_accuracy
    print 'ridge 5'
    print ridge5_accuracy
    print 'ridge 10'
    print ridge10_accuracy
    print 'ridge 15'
    print ridge15_accuracy
    print 'ridge 5r'
    print ridge5_r_accuracy
    print 'ridge 10r'
    print ridge10_r_accuracy
    print 'ridge 15r'
    print ridge15_r_accuracy
    print 'elastic 2'
    print elastic2_accuracy
    print 'elastic 5'
    print elastic5_accuracy
    print 'elastic 10'
    print elastic10_accuracy
    print 'elastic 15'
    print elastic15_accuracy
    print 'elastic 5r'
    print elastic5_r_accuracy
    print 'elastic 10r'
    print elastic10_r_accuracy
    print 'elastic 15r'
    print elastic15_r_accuracy
    print 'linear 2'
    print linear2_accuracy
    print 'linear 5'
    print linear5_accuracy
    print 'linear 10'
    print linear10_accuracy
    print 'linear 15'
    print linear15_accuracy
    print 'linear 5r'
    print linear5_r_accuracy
    print 'linear 10r'
    print linear10_r_accuracy
    print 'linear 15r'
    print linear15_r_accuracy
    print 'sgd 2'
    print sgd2_accuracy
    print 'sgd 5'
    print sgd5_accuracy
    print 'sgd 10'
    print sgd10_accuracy
    print 'sgd 15'
    print sgd15_accuracy
    print 'sgd 5r'
    print sgd5_r_accuracy
    print 'sgd 10r'
    print sgd10_r_accuracy
    print 'sgd 15r'
    print sgd15_r_accuracy
    print 'rfc  2'
    print rfc2_accuracy
    print 'rfc  5'
    print rfc5_accuracy
    print 'rfc  10'
    print rfc10_accuracy
    print 'rfc  15'
    print rfc15_accuracy
    print 'rfc 5r'
    print rfc5_r_accuracy
    print 'rfc 10r'
    print rfc10_r_accuracy
    print 'rfc 15r'
    print rfc15_r_accuracy
    print 'rfr  2'
    print rfr2_accuracy
    print 'rfr  5'
    print rfr5_accuracy
    print 'rfr  10'
    print rfr10_accuracy
    print 'rfr  15'
    print rfr15_accuracy
    print 'rfr 5r'
    print rfr5_r_accuracy
    print 'rfr 10r'
    print rfr10_r_accuracy
    print 'rfr 15r'
    print rfr15_r_accuracy
    print 'xgbc 2'
    print xgbc2_accuracy
    print 'xgbc 5'
    print xgbc5_accuracy
    print 'xgbc 10'
    print xgbc10_accuracy
    print 'xgbc 15'
    print xgbc15_accuracy
    print 'xgbc 5r'
    print xgbc5_r_accuracy
    print 'xgbc 10r'
    print xgbc10_r_accuracy
    print 'xgbc 15r'
    print xgbc15_r_accuracy
    print 'xgbr 2'
    print xgbr2_accuracy
    print 'xgbr 5'
    print xgbr5_accuracy
    print 'xgbr 10'
    print xgbr10_accuracy
    print 'xgbr 15'
    print xgbr15_accuracy
    print 'xgbr 5r'
    print xgbr5_r_accuracy
    print 'xgbr 10r'
    print xgbr10_r_accuracy
    print 'xgbr 15r'
    print xgbr15_r_accuracy
    print 'mlp 2'
    print mlp2_accuracy
    print 'mlp 5'
    print mlp5_accuracy
    print 'mlp 10'
    print mlp10_accuracy
    print 'mlp 15'
    print mlp15_accuracy
    print 'mlp 5r'
    print mlp5_r_accuracy
    print 'mlp 10r'
    print mlp10_r_accuracy
    print 'mlp 15r'
    print mlp15_r_accuracy
    print 'gbc 2'
    print gbc2_accuracy
    print 'gbc 5'
    print gbc5_accuracy
    print 'gbc 10'
    print gbc10_accuracy
    print 'gbc 15'
    print gbc15_accuracy
    print 'gbc 5r'
    print gbc5_r_accuracy
    print 'gbc 10r'
    print gbc10_r_accuracy
    print 'gbc 15r'
    print gbc15_r_accuracy
