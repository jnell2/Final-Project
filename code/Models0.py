import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestClassifier
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

    return

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

if __name__ == '__main__':

    # this gets the most recent final csv files
    # if you want up-to-date information, re-run DataCleaning.py
    # REMEMBER WE ARE TRAINING OUR MODEL ON LAST SEASONS DATA
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

    # drop variables
    df_final5 = drop_variables(df_final5)
    df_final15 = drop_variables(df_final15)
    df_final5_r = drop_variables_ratio(df_final5_r)
    df_final10_r = drop_variables_ratio(df_final10_r)

    # the goal is to predict 2 variables:
    # 1) home team W/L
    # 2) spread

    accuracy_svm5 = kfold_svm(df_final5)
