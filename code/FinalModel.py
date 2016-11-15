import ModelExploration as me
import ModelVisualization as mv
import DataCleaning as dc
import pandas as pd
import numpy as np
import cPickle as pk
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
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

    return model

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

    df_final5_LSr = proportion_data(df_final5_LS)

    # gets data that we want to predict on, if don't want to add new rows (past games)
    df_final5 = pd.read_csv('data/final5.csv')
    df_final5.drop(['Unnamed: 0', 'home_giveaways', 'away_giveaways'], axis = 1, inplace = True)


    # if you want to know new games, this appends rows to past games to make 1 big df
    df_all, df_games = get_data()
    df_games, df_final5_new = add_rows(df_all, df_games, 'NYI', 'TBL', '2016-11-14')
    # every time you want to add a new row, copy this exact line and
    # only change team names and date

    df_final5_new_r = proportion_data(df_final5_new)

    # gets model
    xgbc = mv.xgbc(df_final5_LSr)

    # pickles model
    pickle_model(xgbc, filename = 'xgb_classifier_model.pk')

    # unpickle model and get predictions
    predictions = unpickle_and_predict(df_final5_new_r, filename = 'xgb_classifier_model.pk')

    # append predictions to df_final5_new and drop all columns that we don't care about
    df_final = df_final5_new[['home_team', 'away_team', 'date', 'home_team_win']]
    preds = pd.DataFrame(predictions)
    preds.columns = [['prediction']]
    final = pd.merge(df_final, preds, how = 'left', left_index = True, right_index = True)
    final.sort_values('date', ascending = False, inplace = True)
    final = final.reset_index(drop=True)
    # most recent games will be at the top

    # if you want to know the accuracy of the current season predictions:
    # keep in mind, all of the "actual" values for future games is -100
    # so those will bring down the accuracy
    new_accuracy = accuracy_score(final['home_team_win'], final['prediction'])
