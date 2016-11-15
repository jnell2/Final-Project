import ModelExploration as me
import ModelVisualization as mv
import DataCleaning as dc
import pandas as pd
import numpy as np
import cPickle as pk
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

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

def get_data():
    '''
    gets up to date df_games and df_all for add_rows function
    '''
    df_dts = pd.read_csv('data/dts.csv')
    df_dts = df_dts[['team', 'opponent', 'date', 'home_ind', 'W', 'L', \
    'GF', 'GA', 'SF', 'SA', 'PP%', 'PK%', 'FOW%']]
    df_dp = pd.read_csv('data/dp.csv')
    df_dp = df_dp[['team', 'opponent', 'date', 'PIM', 'penalties']]
    df_ds = pd.read_csv('data/ds.csv')
    df_ds = df_ds[['team', 'opponent', 'date', 'hits', 'blocked_shots', \
    'giveaways', 'takeaways', 'save%']]

    df_games = dc.make_GbG_df(df_dts, df_dp, df_ds)
    df_all, df_games = dc.GbG_cumulative_df(df_games)

    return df_all, df_games

def add_rows(df_all, df_games, home_team, away_team, date):
    '''
    df_games and df_all will be updated when code is run with most recent data
    both will have all games up through current day
    need to add future games that we would like to predict

    input home team (3 letter abbr), away team (3 letter abbr), and date (YYYY-MM-DD)

    will add this game to df_games (list of games with H/A/date/H win/spread)
    will then pull relevant stats from df_all to populate stats fields
    '''
    df = pd.DataFrame([home_team, away_team, date, -100, -100]).T
    df.columns = [['home_team', 'away_team', 'date', 'home_team_win', 'spread']]
    df['date'] = pd.to_datetime(df['date'])
    df['home_team_win'] = df['home_team_win'].astype(int)
    df['spread'] = df['spread'].astype(int)

    df_games = df_games.append(df, ignore_index = True)

    df_final5 = dc.cumulative_stats(df_all, df_games, 5)
    df_final5.drop(['home_giveaways', 'away_giveaways'], axis = 1, inplace = True)

    return df_games, df_final5

def pickle_model(model, filename):
    '''
    pass in model that you would like to pickle and filename you would like to use
    will return pickled model
    '''
    pk.dump(model, open(filename, 'w'), 2)

def unpickle_and_predict(df_final, filename):
    '''
    pass in the dataframe that you want to predict on
    function unpickles the model
    returns predictions
    '''
    model = pk.load(open(filename))

    df = df_final.copy()
    y1 = df.pop('home_team_win')
    # variable 1 to predict
    y2 = df.pop('spread')
    # variable 2 to predict
    df.drop(['home_team', 'away_team', 'date'], axis = 1, inplace = True)
    # we don't want any categorical variables in the model
    X = df.values

    predictions = model.predict(X)
    predictions = map(lambda x: 1 if x > 0 else 0, predictions)
    return predictions

if __name__ == '__main__':

    # read in data to train model
    df_final5_LS = pd.read_csv('data/final5LS.csv')
    df_final5_LS.drop(['Unnamed: 0', 'home_giveaways', 'away_giveaways'], axis = 1, inplace = True)

    # gets data that we want to predict on, if don't want to add new rows (past games)
    df_final5 = pd.read_csv('data/final5.csv')
    df_final5.drop(['Unnamed: 0', 'home_giveaways', 'away_giveaways'], axis = 1, inplace = True)

    # if you want to know new games, this appends rows to past games to make 1 big df
    df_all, df_games = get_data()
    df_games, df_final5_new = add_rows(df_all, df_games, 'NYI', 'TBL', '2016-11-14')
    # every time you want to add a new row, copy this exact line and
    # only change team names and date

    # gets model
    mlpr, predictionsLS = mlpr(df_final5_LS)

    # pickles model
    pickle_model(mlpr, filename = 'mlp_regressor_model.pk')

    # unpickle model and get predictions
    predictions = unpickle_and_predict(df_final5_new, filename = 'mlp_regressor_model.pk')

    # append predictions to df_final5_new and drop all columns that we don't care about
    df_final = df_final5_new[['home_team', 'away_team', 'date', 'home_team_win']]
    preds = pd.DataFrame(predictions)
    preds.columns = [['prediction']]
    final = pd.merge(df_final, preds, how = 'left', left_index = True, right_index = True)
    final.sort_values('date', ascending = True, inplace = True)
    final = final.reset_index(drop=True)
    final['match'] = np.where(final['home_team_win'] == final['prediction'], 1, 0)
    final['cumulative_average'] = pd.expanding_mean(final['match'], 1)
    # most recent games will be at the bottom
    # 59.8% accuracy

    # last season
    df_finalLS = df_final5_LS[['home_team', 'away_team', 'date', 'home_team_win']]
    predsLS = pd.DataFrame(predictionsLS)
    predsLS.columns = [['prediction']]
    finalLS = pd.merge(df_finalLS, predsLS, how = 'left', left_index = True, right_index = True)
    finalLS.sort_values('date', ascending = True, inplace = True)
    finalLS = finalLS.reset_index(drop=True)
    finalLS['match'] = np.where(finalLS['home_team_win'] == finalLS['prediction'], 1, 0)
    finalLS['cumulative_average'] = pd.expanding_mean(finalLS['match'], 1)
    # these results don't match what was found in ModelVisualization.py
    # 62.3% accuracy
