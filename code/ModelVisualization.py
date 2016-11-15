import ModelExploration as me
import DataCleaning as dc
import pandas as pd
import numpy as np
import cPickle as pk
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor

# This file will pickle chosen models, unpickle them, make predictions, and find
# the cumulative average accuracy as time goes on.
# From here, I will choose a final model that will then be transferred to the
# FinalModel.py file where the test data will be passed in.
# In this file, I will also make a graphic comparing the cumulative
# accuracies over time for a select few models (3-5)

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
    df['takeaways'] = (df['home_takeaways'] + 1e-4)/(df['away_takeaways'] + df['home_takeaways'] + 1e-4)
    df['save%'] = (df['home_save%'] + 1e-4)/(df['away_save%'] + df['home_save%'] + 1e-4)
    df['shot%'] = (df['home_shot%'] + 1e-4)/(df['away_shot%'] + df['home_shot%'] + 1e-4)
    df['PDO'] = (df['home_PDO'] + 1e-4)/(df['away_PDO'] + df['home_PDO'] + 1e-4)
    df['corsi'] = (df['home_corsi'] + 1e-4)/(df['away_corsi'] + df['home_corsi'] + 1e-4)
    df.drop(['home_spread', 'home_wins', 'home_goals', 'home_shots', 'home_PP%', \
    'home_FOW%', 'home_PIM', 'home_hits', 'home_blocked', \
    'home_takeaways', 'home_save%', 'home_shot%', 'home_PDO', 'home_corsi', \
    'away_spread', 'away_wins', 'away_goals', 'away_shots', 'away_PP%', \
    'away_FOW%', 'away_PIM', 'away_hits', 'away_blocked', \
    'away_takeaways', 'away_save%', 'away_shot%', 'away_PDO', 'away_corsi'], \
    axis = 1, inplace = True)

    return df

# Best Model contenders

def mlpr(df_final):
    '''
    pass in df_final dataframe
    function fits model
    '''
    df = df_final.copy()
    y1 = df.pop('home_team_win')
    # variable 1 to predict
    y2 = df.pop('spread')
    # variable 2 to predict
    df.drop(['home_team', 'away_team', 'date'], axis = 1, inplace = True)
    # we don't want any categorical variables in the model
    X = df.values

    model = MLPRegressor(solver = 'lbfgs', alpha = 2.0091e-5, hidden_layer_sizes = (5,2), \
    activation = 'relu', learning_rate = 'adaptive', tol = 1e-4, random_state = 2)

    scaler = StandardScaler()
    scaler.fit(X)
    X= scaler.transform(X)
    model.fit(X, y2)
    predictions = model.predict(X)
    predictions = map(lambda x: 1 if x > 0 else 0, predictions)

    return model, predictions

def mlp(df_final):
    '''
    pass in df_final dataframe
    function fits model
    '''
    df = df_final.copy()
    y1 = df.pop('home_team_win')
    # variable 1 to predict
    y2 = df.pop('spread')
    # variable 2 to predict
    df.drop(['home_team', 'away_team', 'date'], axis = 1, inplace = True)
    # we don't want any categorical variables in the model
    X = df.values

    model = MLPClassifier(solver = 'lbfgs', alpha = 0.001100009, hidden_layer_sizes = (5,2), \
    activation = 'relu', learning_rate = 'adaptive', tol = 1e-4, random_state = 2)

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    model.fit(X, y1)
    predictions = model.predict(X)

    return model, predictions

def xgbc(df_final):
    '''
    pass in df_final dataframe
    function fits model
    '''
    df = df_final.copy()
    y1 = df.pop('home_team_win')
    # variable 1 to predict
    y2 = df.pop('spread')
    # variable 2 to predict
    df.drop(['home_team', 'away_team', 'date'], axis = 1, inplace = True)
    # we don't want any categorical variables in the model
    X = df.values

    model = xgb.XGBClassifier()

    model.fit(X, y1)
    predictions = model.predict(X)

    return model, predictions

def elastic(df_final):
    '''
    pass in df_final dataframe
    function fits model
    '''
    df = df_final.copy()
    y1 = df.pop('home_team_win')
    # variable 1 to predict
    y2 = df.pop('spread')
    # variable 2 to predict
    df.drop(['home_team', 'away_team', 'date'], axis = 1, inplace = True)
    # we don't want any categorical variables in the model
    X = df.values

    model = ElasticNet(alpha = 0.8, fit_intercept = False)

    model.fit(X, y2)
    predictions = model.predict(X)
    predictions = map(lambda x: 1 if x > 0 else 0, predictions)

    return model, predictions

def lasso(df_final):
    '''
    pass in df_final dataframe
    function fits model
    '''
    df = df_final.copy()
    y1 = df.pop('home_team_win')
    # variable 1 to predict
    y2 = df.pop('spread')
    # variable 2 to predict
    df.drop(['home_team', 'away_team', 'date'], axis = 1, inplace = True)
    # we don't want any categorical variables in the model
    X = df.values

    model = Lasso(alpha = 0.0119, random_state = 2, fit_intercept = False)

    model.fit(X, y2)
    predictions = model.predict(X)
    predictions = map(lambda x: 1 if x > 0 else 0, predictions)

    return model, predictions

def ridge(df_final):
    '''
    pass in df_final dataframe
    function fits model
    '''
    df = df_final.copy()
    y1 = df.pop('home_team_win')
    # variable 1 to predict
    y2 = df.pop('spread')
    # variable 2 to predict
    df.drop(['home_team', 'away_team', 'date'], axis = 1, inplace = True)
    # we don't want any categorical variables in the model
    X = df.values

    model = Ridge(alpha = 15, fit_intercept = False, random_state = 2)

    model.fit(X, y2)
    predictions = model.predict(X)
    predictions = map(lambda x: 1 if x > 0 else 0, predictions)

    return model, predictions

def cumulative_accuracy(df_final, predictions):
    '''
    takes in last season dataframe and pickled model you want to visualize
    return a dataframe with cumulative average over time
    '''
    df_final = df_final[['home_team', 'away_team', 'date', 'home_team_win']]
    preds = pd.DataFrame(predictions)
    preds.columns = [['prediction']]
    final = pd.merge(df_final, preds, how = 'left', left_index = True, right_index = True)
    final.sort_values('date', ascending = True, inplace = True)
    final = final.reset_index(drop=True)
    final['match'] = np.where((final['home_team_win']) == final['prediction'], 1, 0)
    final['cumulative_average'] = pd.expanding_mean(final['match'], 1)

    return final

if __name__ == '__main__':

    # read in data to train model: LAST SEASON
    df_final5_LS = pd.read_csv('data/final5LS.csv')
    df_final5_LS.drop(['Unnamed: 0', 'home_giveaways', 'away_giveaways'], axis = 1, inplace = True)
    df_final10_LS = pd.read_csv('data/final10LS.csv')
    df_final10_LS.drop(['Unnamed: 0', 'home_giveaways', 'away_giveaways'], axis = 1, inplace = True)

    df_final5_LSr = proportion_data(df_final5_LS)

    # read in data to train model: THIS SEASON
    df_final5 = pd.read_csv('data/final5.csv')
    df_final5.drop(['Unnamed: 0', 'home_giveaways', 'away_giveaways'], axis = 1, inplace = True)
    df_final10 = pd.read_csv('data/final10.csv')
    df_final10.drop(['Unnamed: 0', 'home_giveaways', 'away_giveaways'], axis = 1, inplace = True)

    df_final5_r = proportion_data(df_final5)

    # gets models, predictions: LAST SEASON
    mlprLS, mlprLS_preds = mlpr(df_final5_LS)
    mlpLS, mlpLS_preds = mlp(df_final5_LS)
    xgbcLS, xgbcLS_preds = xgbc(df_final5_LSr)
    enLS, enLS_preds = elastic(df_final5_LS)
    lasso5LS, lasso5LS_preds = lasso(df_final5_LS)
    lasso10LS, lasso10LS_preds = lasso(df_final10_LS)
    ridgeLS, ridgeLS_preds = ridge(df_final10_LS)

    # gets models, predictions: THIS SEASON
    mlpr, mlpr_preds = mlpr(df_final5)
    mlp, mlp_preds = mlp(df_final5)
    xgbc, xgbc_preds = xgbc(df_final5_r)
    en, en_preds = elastic(df_final5)
    lasso5, lasso5_preds = lasso(df_final5)
    lasso10, lasso10_preds = lasso(df_final10)
    ridge, ridge_preds = ridge(df_final10)

    # LAST SEASON DATA
    # THESE MODELS ARE FOR VISUALIZATIONS ONLY
    # unpickle model, get predictions, and return df with cumulative average accuracy
    df_mlprLS = cumulative_accuracy(df_final5_LS, mlprLS_preds)
        # accuracy: 62.3%
    df_mlpLS = cumulative_accuracy(df_final5_LS, mlpLS_preds)
        # accuracy: 65.8%
    df_xgbcLS = cumulative_accuracy(df_final5_LSr, xgbcLS_preds)
        # accuracy: 80.9%
    df_enLS = cumulative_accuracy(df_final5_LS, enLS_preds)
        # accuracy: 53.4%
    df_lasso5LS = cumulative_accuracy(df_final5_LS, lasso5LS_preds)
        # accuracy: 55.7%
    df_lasso10LS = cumulative_accuracy(df_final10_LS, lasso10LS_preds)
        # accuracy: 57.3%
    df_ridgeLS = cumulative_accuracy(df_final10_LS, ridgeLS_preds)
        # accuracy: 57.6%

    # THIS SEASON DATA
    # THESE MODELS ARE FOR VISUALIZATIONS ONLY
    # unpickle model, get predictions, and return df with cumulative average accuracy
    df_mlpr = cumulative_accuracy(df_final5, mlpr_preds)
        # accuracy: 63.8%
    df_mlp = cumulative_accuracy(df_final5, mlp_preds)
        # accuracy: 70.8%
    df_xgbc = cumulative_accuracy(df_final5_r, xgbc_preds)
        # accuracy: 99.0%
    df_en = cumulative_accuracy(df_final5, en_preds)
        # accuracy: 60.8%
    df_lasso5 = cumulative_accuracy(df_final5, lasso5_preds)
        # accuracy: 64.8%
    df_lasso10 = cumulative_accuracy(df_final10, lasso10_preds)
        # accuracy: 61.3%
    df_ridge = cumulative_accuracy(df_final10, ridge_preds)
        # accuracy: 62.3%
