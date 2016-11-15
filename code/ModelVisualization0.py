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

def rfr(df_final):
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

    model = RandomForestRegressor(random_state = 2)

    model.fit(X, y2)

    return model

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

def cumulative_accuracy(df_final, filename):
    '''
    takes in last season dataframe and pickled model you want to visualize
    return a dataframe with cumulative average over time
    '''
    predictions = unpickle_and_predict(df_final, filename)
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
    df_final15_LS = pd.read_csv('data/final15LS.csv')
    df_final15_LS.drop(['Unnamed: 0', 'home_giveaways', 'away_giveaways'], axis = 1, inplace = True)

    df_final5_LSr = proportion_data(df_final5_LS)

    # read in data to train model: THIS SEASON
    df_final5 = pd.read_csv('data/final5.csv')
    df_final5.drop(['Unnamed: 0', 'home_giveaways', 'away_giveaways'], axis = 1, inplace = True)
    df_final10 = pd.read_csv('data/final10.csv')
    df_final10.drop(['Unnamed: 0', 'home_giveaways', 'away_giveaways'], axis = 1, inplace = True)
    df_final15 = pd.read_csv('data/final15.csv')
    df_final15.drop(['Unnamed: 0', 'home_giveaways', 'away_giveaways'], axis = 1, inplace = True)

    df_final5_r = proportion_data(df_final5)

    # gets models: LAST SEASON
    mlprLS = mlpr(df_final5_LS)
    mlpLS = mlp(df_final5_LS)
    xgbcLS = xgbc(df_final5_LSr)
    enLS = elastic(df_final5_LS)
    lasso5LS = lasso(df_final5_LS)
    lasso10LS = lasso(df_final10_LS)
    ridgeLS = ridge(df_final10_LS)
    rfrLS = rfr(df_final15_LS)

    # gets models: THIS SEASON
    mlpr = mlpr(df_final5)
    mlp = mlp(df_final5)
    xgbc = xgbc(df_final5_r)
    en = elastic(df_final5)
    lasso5 = lasso(df_final5)
    lasso10 = lasso(df_final10)
    ridge = ridge(df_final10)
    rfr = rfr(df_final15)

    # pickles models: LAST SEASON
    pickle_model(mlprLS, filename = 'mlpr_regressor_modelLS.pk')
    pickle_model(mlpLS, filename = 'mlp_classifier_modelLS.pk')
    pickle_model(xgbcLS, filename = 'xgb_classifier_modelLS.pk')
    pickle_model(enLS, filename = 'elastic_net_modelLS.pk')
    pickle_model(lasso5LS, filename = 'lasso5_modelLS.pk')
    pickle_model(lasso10LS, filename = 'lasso10_modelLS.pk')
    pickle_model(ridgeLS, filename = 'ridge_regression_modelLS.pk')
    pickle_model(rfrLS, filename = 'random_forest_regressorLS.pk')

    # pickles models: THIS SEASON
    pickle_model(mlpr, filename = 'mlpr_regressor_model.pk')
    pickle_model(mlp, filename = 'mlp_classifier_model.pk')
    pickle_model(xgbc, filename = 'xgb_classifier_model.pk')
    pickle_model(en, filename = 'elastic_net_model.pk')
    pickle_model(lasso5, filename = 'lasso5_model.pk')
    pickle_model(lasso10, filename = 'lasso10_model.pk')
    pickle_model(ridge, filename = 'ridge_regression_model.pk')
    pickle_model(rfr, filename = 'random_forest_regressor.pk')

    # LAST SEASON DATA
    # THESE MODELS ARE FOR VISUALIZATIONS ONLY
    # unpickle model, get predictions, and return df with cumulative average accuracy
    df_mlprLS = cumulative_accuracy_scaler(df_final5_LS, filename = 'mlpr_regressor_modelLS.pk')
        # accuracy: 62.3%
    df_mlpLS = cumulative_accuracy_scaler(df_final5_LS, filename = 'mlp_classifier_modelLS.pk')
        # accuracy: 65.8%
    df_xgbcLS = cumulative_accuracy(df_final5_LSr, filename = 'xgb_classifier_modelLS.pk')
        # accuracy: 80.9%
    df_enLS = cumulative_accuracy(df_final5_LS, filename = 'elastic_net_modelLS.pk')
        # accuracy: 53.4%
    df_lasso5LS = cumulative_accuracy(df_final5_LS, filename = 'lasso5_modelLS.pk')
        # accuracy: 55.7%
    df_lasso10LS = cumulative_accuracy(df_final10_LS, filename = 'lasso10_modelLS.pk')
        # accuracy: 57.3%
    df_ridgeLS = cumulative_accuracy(df_final10_LS, filename = 'ridge_regression_modelLS.pk')
        # accuracy: 57.6%
    df_rfrLS = cumulative_accuracy(df_final15_LS, filename = 'random_forest_regressorLS.pk')
        # accuracy: 91.5%

    # THIS SEASON DATA
    # THESE MODELS ARE FOR VISUALIZATIONS ONLY
    # unpickle model, get predictions, and return df with cumulative average accuracy
    df_mlpr = cumulative_accuracy_scaler(df_final5, filename = 'mlpr_regressor_model.pk')
        # accuracy: 63.8%
    df_mlp = cumulative_accuracy_scaler(df_final5, filename = 'mlp_classifier_model.pk')
        # accuracy: 70.8%
    df_xgbc = cumulative_accuracy(df_final5_r, filename = 'xgb_classifier_model.pk')
        # accuracy: 99.0%
    df_en = cumulative_accuracy(df_final5, filename = 'elastic_net_model.pk')
        # accuracy: 60.1%
    df_lasso5 = cumulative_accuracy(df_final5, filename = 'lasso5_model.pk')
        # accuracy: 64.8%
    df_lasso10 = cumulative_accuracy(df_final10, filename = 'lasso10_model.pk')
        # accuracy: 61.3%
    df_ridge = cumulative_accuracy(df_final10, filename = 'ridge_regression_model.pk')
        # accuracy: 62.3%
    df_rfr = cumulative_accuracy(df_final15, filename = 'random_forest_regressor.pk')
        # accuracy: 88.4%
