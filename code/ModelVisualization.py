import Models
import DataCleaning as dc
import pandas as pd
import numpy as np
import cPickle as pk
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cross_validation import KFold

# This file will pickle chosen models, unpickle them, make predictions, and find
# the cumulative average accuracy as time goes on.
# From here, I will choose a final model that will then be transferred to the
# FinalModel.py file where the test data will be passed in.
# In this file, I will also make a graphic comparing the cumulative
# accuracies over time for a select few models (3-5)

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

    return model

def mlp(df_final):
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

    model = MLPClassifier(solver = 'lbfgs', alpha = 0.001100009, hidden_layer_sizes = (5,2), \
    activation = 'relu', learning_rate = 'adaptive', tol = 1e-4, random_state = 2)

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    model.fit(X, y1)

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

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    predictions = model.predict(X)
    predictions = map(lambda x: 1 if x > 0 else 0, predictions)
    return predictions

def cumulative_accuracy(df_final_LS, filename):
    '''
    takes in last season dataframe and pickled model you want to visualize
    return a dataframe with cumulative average over time
    '''
    predictions = unpickle_and_predict(df_final_LS, filename)
    df_final = df_final5_LS[['home_team', 'away_team', 'date', 'home_team_win']]
    preds = pd.DataFrame(predictions)
    preds.columns = [['prediction']]
    final = pd.merge(df_final, preds, how = 'left', left_index = True, right_index = True)
    final.sort_values('date', ascending = True, inplace = True)
    final = final.reset_index(drop=True)
    final['match'] = np.where((final['home_team_win']) == final['prediction'], 1, 0)
    final['cumulative_average'] = pd.expanding_mean(final['match'], 1)

    return final

if __name__ == '__main__':

    # read in data to train model
    df_final5_LS = pd.read_csv('data/final5LS.csv')
    df_final5_LS.drop(['Unnamed: 0', 'home_giveaways', 'away_giveaways'], axis = 1, inplace = True)

    # gets models and accuracy of models
    mlpr = mlpr(df_final5_LS)
    mlp = mlp(df_final5_LS)

    # pickles models
    pickle_model(mlpr, filename = 'mlpr_regressor_model.pk')
    pickle_model(mlp, filename = 'mlp_classifier_model.pk')

    # unpickle model, get predictions, and return df with cumulative average accuracy
    df_mlpr = cumulative_accuracy(df_final5_LS, filename = 'mlpr_regressor_model.pk')
    df_mlp = cumulative_accuracy(df_final5_LS, filename = 'mlp_classifier_model.pk')
