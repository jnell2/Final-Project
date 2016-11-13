import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cross_validation import KFold
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB

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
    # df_final['Hwins2'] = df_final['home_wins'] + df_final['home_wins']
    # df_final['Awins2'] = df_final['away_wins'] + df_final['away_wins']
    # df_final['Hgoals2'] = df_final['home_goals'] + df_final['home_goals']
    # df_final['Agoals2'] = df_final['away_goals'] + df_final['away_goals']
    # df_final['HFOW%2'] = df_final['home_FOW%'] + df_final['home_FOW%']
    # df_final['AFOW%2'] = df_final['away_FOW%'] + df_final['away_FOW%']
    # df_final['Hcorsi2'] = df_final['home_corsi'] + df_final['home_corsi']
    # df_final['Acorsi2'] = df_final['away_corsi'] + df_final['away_corsi']
    # df_final['HPIM'] = df_final['home_PIM'] + df_final['home_PIM']
    # df_final['APIM'] = df_final['away_PIM'] + df_final['away_PIM']

    df_final.drop(['home_giveaways', 'away_giveaways'], axis = 1, inplace = True)

    return df_final

def drop_variables_ratio(df_final):
    '''
    drop variables to test accuracies
    '''
    df_final.drop(['giveaways'], axis = 1, inplace = True)

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
    model = MLPClassifier(solver = 'lbfgs', alpha = 0.001100009, hidden_layer_sizes = (5,2), \
    activation = 'relu', learning_rate = 'adaptive', tol = 1e-4, random_state = 2)
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

def kfold_mlpr(df_final):
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
    model = MLPRegressor(solver = 'lbfgs', alpha = 2.0091e-5, hidden_layer_sizes = (5,2), \
    activation = 'relu', learning_rate = 'adaptive', tol = 1e-4, random_state = 2)
    for train, test in kf:
        scaler = StandardScaler()
        scaler.fit(X[train])
        X[train] = scaler.transform(X[train])
        X[test] = scaler.transform(X[test])
        model.fit(X[train], y2[train])
        pred = model.predict(X[test])
        pred = map(lambda x: 1 if x > 0 else 0, pred)
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

def kfold_gbr(df_final):
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
    model = GradientBoostingRegressor()
    for train, test in kf:
        model.fit(X[train], y2[train])
        pred = model.predict(X[test])
        pred = map(lambda x: 1 if x > 0 else 0, pred)
        accuracy[index] = accuracy_score(y1[test], pred)
        index +=1

    return np.mean(accuracy)

def kfold_svm(df_final):
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
    model = SVC(kernel = 'sigmoid')
    for train, test in kf:
        model.fit(X[train], y1[train])
        pred = model.predict(X[test])
        # pred = map(lambda x: 1 if x > 0 else 0, pred)
        accuracy[index] = accuracy_score(y1[test], pred)
        index +=1

    return np.mean(accuracy)

def kfold_svmr(df_final):
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
    model = SVR(kernel = 'sigmoid')
    for train, test in kf:
        model.fit(X[train], y2[train])
        pred = model.predict(X[test])
        pred = map(lambda x: 1 if x > 0 else 0, pred)
        accuracy[index] = accuracy_score(y1[test], pred)
        index +=1

    return np.mean(accuracy)

def kfold_nb(df_final):
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
    model = GaussianNB()
    for train, test in kf:
        model.fit(X[train], y2[train])
        pred = model.predict(X[test])
        pred = map(lambda x: 1 if x > 0 else 0, pred)
        accuracy[index] = accuracy_score(y1[test], pred)
        index +=1

    return np.mean(accuracy)

if __name__ == '__main__':

    # this gets the most recent final csv files
    # if you want up-to-date information, re-run DataCleaning.py
    # REMEMBER WE ARE TRAINING OUR MODEL ON LAST SEASONS DATA
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

    # # Logistic Regression results
    # logistic2_accuracy = kfold_logistic(df_final2)
    # logistic5_accuracy = kfold_logistic(df_final5)
    # logistic10_accuracy = kfold_logistic(df_final10)
    # logistic15_accuracy = kfold_logistic(df_final15)
    # logistic5_r_accuracy = kfold_logistic(df_final5_r)
    # logistic10_r_accuracy = kfold_logistic(df_final10_r)
    # logistic15_r_accuracy = kfold_logistic(df_final15_r)
    #
    # # Linear Regression results
    # # Lasso
    # lasso2_accuracy = kfold_lasso(df_final2)
    # lasso5_accuracy = kfold_lasso(df_final5)
    # lasso10_accuracy = kfold_lasso(df_final10)
    # lasso15_accuracy = kfold_lasso(df_final15)
    # lasso5_r_accuracy = kfold_lasso(df_final5_r)
    # lasso10_r_accuracy = kfold_lasso(df_final10_r)
    # lasso15_r_accuracy = kfold_lasso(df_final15_r)
    # # Ridge
    # ridge2_accuracy = kfold_ridge(df_final2)
    # ridge5_accuracy = kfold_ridge(df_final5)
    # ridge10_accuracy = kfold_ridge(df_final10)
    # ridge15_accuracy = kfold_ridge(df_final15)
    # ridge5_r_accuracy = kfold_ridge(df_final5_r)
    # ridge10_r_accuracy = kfold_ridge(df_final10_r)
    # ridge15_r_accuracy = kfold_ridge(df_final15_r)
    # # Elastic Net
    # elastic2_accuracy = kfold_elastic(df_final2)
    # elastic5_accuracy = kfold_elastic(df_final5)
    # elastic10_accuracy = kfold_elastic(df_final10)
    # elastic15_accuracy = kfold_elastic(df_final15)
    # elastic5_r_accuracy = kfold_elastic(df_final5_r)
    # elastic10_r_accuracy = kfold_elastic(df_final10_r)
    # elastic15_r_accuracy = kfold_elastic(df_final15_r)
    # # Linear Regression
    # linear2_accuracy = kfold_linear(df_final2)
    # linear5_accuracy = kfold_linear(df_final5)
    # linear10_accuracy = kfold_linear(df_final10)
    # linear15_accuracy = kfold_linear(df_final15)
    # linear5_r_accuracy = kfold_linear(df_final5_r)
    # linear10_r_accuracy = kfold_linear(df_final10_r)
    # linear15_r_accuracy = kfold_linear(df_final15_r)
    # # SGD Regressor
    # sgd2_accuracy = kfold_sgd(df_final2)
    # sgd5_accuracy = kfold_sgd(df_final5)
    # sgd10_accuracy = kfold_sgd(df_final10)
    # sgd15_accuracy = kfold_sgd(df_final15)
    # sgd5_r_accuracy = kfold_sgd(df_final5_r)
    # sgd10_r_accuracy = kfold_sgd(df_final10_r)
    # sgd15_r_accuracy = kfold_sgd(df_final15_r)
    #
    # # Random Forest Classifier results
    # rfc2_accuracy = kfold_rfc(df_final2)
    # rfc5_accuracy = kfold_rfc(df_final5)
    # rfc10_accuracy = kfold_rfc(df_final10)
    # rfc15_accuracy = kfold_rfc(df_final15)
    # rfc5_r_accuracy = kfold_rfc(df_final5_r)
    # rfc10_r_accuracy = kfold_rfc(df_final10_r)
    # rfc15_r_accuracy = kfold_rfc(df_final15_r)
    # # Random Forest Regressor results
    # rfr2_accuracy = kfold_rfr(df_final2)
    # rfr5_accuracy = kfold_rfr(df_final5)
    # rfr10_accuracy = kfold_rfr(df_final10)
    # rfr15_accuracy = kfold_rfr(df_final15)
    # rfr5_r_accuracy = kfold_rfr(df_final5_r)
    # rfr10_r_accuracy = kfold_rfr(df_final10_r)
    # rfr15_r_accuracy = kfold_rfr(df_final15_r)

    # # XG Boost Classifier results
    # xgbc2_accuracy = kfold_xgbc(df_final2)
    # xgbc5_accuracy = kfold_xgbc(df_final5)
    # xgbc10_accuracy = kfold_xgbc(df_final10)
    # xgbc15_accuracy = kfold_xgbc(df_final15)
    # xgbc5_r_accuracy = kfold_xgbc(df_final5_r)
    # xgbc10_r_accuracy = kfold_xgbc(df_final10_r)
    # xgbc15_r_accuracy = kfold_xgbc(df_final15_r)
    # # XG Boost Regressor results
    # xgbr2_accuracy = kfold_xgbr(df_final2)
    # xgbr5_accuracy = kfold_xgbr(df_final5)
    # xgbr10_accuracy = kfold_xgbr(df_final10)
    # xgbr15_accuracy = kfold_xgbr(df_final15)
    # xgbr5_r_accuracy = kfold_xgbr(df_final5_r)
    # xgbr10_r_accuracy = kfold_xgbr(df_final10_r)
    # xgbr15_r_accuracy = kfold_xgbr(df_final15_r)

    # MLP Classifier results
    mlp2_accuracy = kfold_mlp(df_final2)
    mlp5_accuracy = kfold_mlp(df_final5)
    mlp10_accuracy = kfold_mlp(df_final10)
    mlp15_accuracy = kfold_mlp(df_final15)
    mlp5_r_accuracy = kfold_mlp(df_final5_r)
    mlp10_r_accuracy = kfold_mlp(df_final10_r)
    mlp15_r_accuracy = kfold_mlp(df_final15_r)
    # MLP Regressor results
    mlpr2_accuracy = kfold_mlpr(df_final2)
    mlpr5_accuracy = kfold_mlpr(df_final5)
    mlpr10_accuracy = kfold_mlpr(df_final10)
    mlpr15_accuracy = kfold_mlpr(df_final15)
    mlpr5_r_accuracy = kfold_mlpr(df_final5_r)
    mlpr10_r_accuracy = kfold_mlpr(df_final10_r)
    mlpr15_r_accuracy = kfold_mlpr(df_final15_r)

    # Gradient Boosting Classifier results
    gbc2_accuracy = kfold_gbc(df_final2)
    gbc5_accuracy = kfold_gbc(df_final5)
    gbc10_accuracy = kfold_gbc(df_final10)
    gbc15_accuracy = kfold_gbc(df_final15)
    gbc5_r_accuracy = kfold_gbc(df_final5_r)
    gbc10_r_accuracy = kfold_gbc(df_final10_r)
    gbc15_r_accuracy = kfold_gbc(df_final15_r)
    # Gradient Boosting Regressor results
    gbr2_accuracy = kfold_gbr(df_final2)
    gbr5_accuracy = kfold_gbr(df_final5)
    gbr10_accuracy = kfold_gbr(df_final10)
    gbr15_accuracy = kfold_gbr(df_final15)
    gbr5_r_accuracy = kfold_gbr(df_final5_r)
    gbr10_r_accuracy = kfold_gbr(df_final10_r)
    gbr15_r_accuracy = kfold_gbr(df_final15_r)

    # SVM Classifier results
    svm2_accuracy = kfold_svm(df_final2)
    svm5_accuracy = kfold_svm(df_final5)
    svm10_accuracy = kfold_svm(df_final10)
    svm15_accuracy = kfold_svm(df_final15)
    svm5_r_accuracy = kfold_svm(df_final5_r)
    svm10_r_accuracy = kfold_svm(df_final10_r)
    svm15_r_accuracy = kfold_svm(df_final15_r)
    # SVM Regressor results
    svmr2_accuracy = kfold_svmr(df_final2)
    svmr5_accuracy = kfold_svmr(df_final5)
    svmr10_accuracy = kfold_svmr(df_final10)
    svmr15_accuracy = kfold_svmr(df_final15)
    svmr5_r_accuracy = kfold_svmr(df_final5_r)
    svmr10_r_accuracy = kfold_svmr(df_final10_r)
    svmr15_r_accuracy = kfold_svmr(df_final15_r)

    # Naive Bayes results
    nb2_accuracy = kfold_nb(df_final2)
    nb5_accuracy = kfold_nb(df_final5)
    nb10_accuracy = kfold_nb(df_final10)
    nb15_accuracy = kfold_nb(df_final15)
