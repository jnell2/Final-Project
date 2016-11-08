import pandas as pd
import numpy as np
from datetime import datetime
import WebScraping as ws

abbr_dict = {'Canadiens': 'MTL',
            'Penguins': 'PIT',
            'Oilers': 'EDM',
            'Rangers': 'NYR',
            'Wings': 'DET',
            'Blackhawks': 'CHI',
            'Wild': 'MIN',
            'Capitals': 'WSH',
            'Lightning': 'TBL',
            'Sharks': 'SJS',
            'Senators': 'OTT',
            'Blues': 'STL',
            'Flyers': 'PHI',
            'Bruins': 'BOS',
            'Ducks': 'ANA',
            'Sabres': 'BUF',
            'Devils': 'NJD',
            'Flames': 'CGY',
            'Canucks': 'VAN',
            'Panthers': 'FLA',
            'Jackets': 'CBJ',
            'Leafs': 'TOR',
            'Islanders': 'NYI',
            'Jets': 'WPG',
            'Kings': 'LAK',
            'Avalanche': 'COL',
            'Stars': 'DAL',
            'Predators': 'NSH',
            'Hurricanes': 'CAR',
            'Coyotes': 'ARI'}

def clean_dts(df):
    '''
    pass in appropriate dataframe from WebScraping.py
    will return the clean df for the game-by-game team summary
    '''
    df['team'] = df['team'].str.split().str[-1]
    df.replace({'team': abbr_dict}, inplace = True)
    df['date'] = df['game'].str.split().str[0]
    df['home_ind'] = df['game'].str.split().str[1]
    df['home_ind'] = df['home_ind'].map({'vs': 1, '@': 0})
    df['W'] = df['W'].astype(int)
    df['L'] = df['L'].astype(int)
    df['OTL'] = df['OTL'].astype(int)
    df['L'] = df['L'] + df['OTL']
    df['GF'] = df['GF'].astype(int)
    df['GA'] = df['GA'].astype(int)
    df['SF'] = df['SF'].astype(int)
    df['SA'] = df['SA'].astype(int)
    df['PP%'] = df['PP%'].astype(float)
    df['PK%'] = df['PK%'].astype(float)
    df['FOW%'] = df['FOW%'].astype(float)
    df['date'] = pd.to_datetime(df['date'])
    df.drop(['#', 'game', 'GP', 'T', 'OTL', 'points', 'PPG', 'PP', 'timesSH', 'PPGA', 'FOW', 'FOL'], axis=1, inplace=True)
    df = df.reset_index(drop=True)
    return df

def clean_dp(df):
    '''
    pass in appropriate dataframe from WebScraping.py
    will return the clean df for the game-by-game penalty report
    '''
    df['team'] = df['team'].str.split().str[-1]
    df.replace({'team': abbr_dict}, inplace = True)
    df['date'] = df['game'].str.split().str[0]
    df['PIM'] = df['PIM'].astype(int)
    df['penalties'] = df['penalties'].astype(int)
    df['date'] = pd.to_datetime(df['date'])
    df.drop(['#', 'game', 'GP', 'W', 'L', 'T', 'OTL', 'points', 'minor', 'major', 'misconduct', 'game_misconduct', 'match'], axis=1, inplace=True)
    return df

def clean_ds(df):
    '''
    pass in appropriate dataframe from WebScraping.py
    will return the clean df for the game-by-game shots report
    '''
    df['team'] = df['team'].str.split().str[-1]
    df.replace({'team': abbr_dict}, inplace = True)
    df['date'] = df['game'].str.split().str[0]
    df['hits'] = df['hits'].astype(int)
    df['blocked_shots'] = df['blocked_shots'].astype(int)
    df['giveaways'] = df['giveaways'].astype(int)
    df['takeaways'] = df['takeaways'].astype(int)
    df['save%'] = df['save%'].astype(float)
    df['date'] = pd.to_datetime(df['date'])
    df.drop(['#', 'game', 'GP', 'W', 'L', 'T', 'OTL', 'points', 'FOW', 'FOL', 'FO', 'FOW%', 'SF', 'GF'], axis=1, inplace=True)
    return df

def get_clean_data(df_dts, df_dp, df_ds):
    '''
    pass in the 8 dataframes, in the order listed above
    will return the cleaned up dataframes
    '''
    df_dts = clean_dts(df_dts)
    df_dp = clean_dp(df_dp)
    df_ds = clean_ds(df_ds)
    return df_dts, df_dp, df_ds

def make_GbG_df(df_dts, df_dp, df_ds):
    '''
    pass in the three appropriate dataframes containing game-by-game data
    will return a dataframe with all relevant information for a GbG basis
    '''
    df = pd.merge(df_dts, df_dp, how = 'left', on = ['team', 'opponent', 'date'])
    df = pd.merge(df, df_ds, how = 'left', on = ['team', 'opponent', 'date'])
    df['home_spread'] = df['GF'] - df['GA']
    df['away_spread'] = df['GA'] - df['GF']
    df['shot%'] = df['GF'] / df['SF']
    df['PDO'] = df['save%'] + df['shot%']
    df['corsi'] = df['SF'] - df['SA']
    df['date'] = pd.to_datetime(df['date'])
    return df

def GbG_cumulative_df(df):
    '''
    pass in dataframe from make_GbG_df
    will return dataframe in desired format
    '''
    dfc = df.copy()
    df = df.loc[df['home_ind']==1]
    df = df[['team', 'opponent', 'date', 'home_spread', 'away_spread', \
    'W', 'L', 'GF', 'GA', 'SF', 'SA', 'PP%', 'FOW%', 'PIM', 'hits', \
    'blocked_shots', 'giveaways', 'takeaways', 'save%', 'shot%', 'PDO', 'corsi']]
    df.columns = ['home_team', 'away_team', 'date', 'home_spread', \
    'away_spread', 'home_team_win', 'away_team_win', 'home_goals', 'away_goals', \
    'home_shots', 'away_shots', 'home_PP%', 'home_FOW%', 'home_PIM', 'home_hits', \
    'home_blocked', 'home_giveaways', 'home_takeaways', 'home_save%', 'home_shot%', \
    'home_PDO', 'home_corsi']

    dfc = dfc[['team', 'opponent', 'date', 'PP%', 'FOW%', 'PIM', 'hits', \
    'blocked_shots', 'giveaways', 'takeaways', 'save%', 'shot%', 'PDO', 'corsi']]
    dfc.columns = ['home_team', 'away_team', 'date', 'away_PP%', 'away_FOW%', \
    'away_PIM', 'away_hits', 'away_blocked', 'away_giveaways', 'away_takeaways',  \
    'away_save%', 'away_shot%', 'away_PDO', 'away_corsi']

    df2 = pd.merge(df, dfc, how = 'left', on = ['home_team', 'away_team', 'date'])

    df_home = df2[['home_team', 'date', 'home_spread', 'home_team_win', 'home_goals', \
    'home_shots', 'home_PP%', 'home_FOW%', 'home_PIM', 'home_hits', 'home_blocked', \
    'home_giveaways', 'home_takeaways', 'home_save%', 'home_shot%', 'home_PDO', \
    'home_corsi']]
    df_home['home_ind'] = 1
    df_home.columns = ['team', 'date', 'spread', 'win', 'goals', 'shots', 'PP%', \
    'FOW%', 'PIM', 'hits', 'blocked', 'giveaways', 'takeaways', 'save%', 'shot%', \
    'PDO', 'corsi', 'home_ind']
    df_away = df2[['away_team', 'date', 'away_spread', 'away_team_win', 'away_goals', \
    'away_shots', 'away_PP%', 'away_FOW%', 'away_PIM', 'away_hits', 'away_blocked',  \
    'away_giveaways', 'away_takeaways', 'away_save%', 'away_shot%', 'away_PDO', \
    'away_corsi']]
    df_away['home_ind'] = 0
    df_away.columns = ['team', 'date', 'spread', 'win', 'goals', 'shots', 'PP%', 'FOW%',\
    'PIM', 'hits', 'blocked', 'giveaways', 'takeaways', 'save%', 'shot%', 'PDO', \
    'corsi', 'home_ind']

    all_data = np.vstack((df_home, df_away))
    df_all = pd.DataFrame(all_data)
    df_all.columns = ['team', 'date', 'spread', 'win', 'goals', 'shots', 'PP%', \
    'FOW%', 'PIM', 'hits', 'blocked', 'giveaways', 'takeaways', 'save%', 'shot%', \
    'PDO', 'corsi', 'home_ind']
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all['spread'] = df_all['spread'].astype(int)
    df_all['win'] = df_all['win'].astype(int)
    df_all['goals'] = df_all['goals'].astype(int)
    df_all['shots'] = df_all['shots'].astype(int)
    df_all['PP%'] = df_all['PP%'].astype(float)
    df_all['FOW%'] = df_all['FOW%'].astype(float)
    df_all['PIM'] = df_all['PIM'].astype(int)
    df_all['hits'] = df_all['hits'].astype(int)
    df_all['blocked'] = df_all['blocked'].astype(int)
    df_all['giveaways'] = df_all['giveaways'].astype(int)
    df_all['takeaways'] = df_all['takeaways'].astype(int)
    df_all['save%'] = df_all['save%'].astype(float)
    df_all['shot%'] = df_all['shot%'].astype(float)
    df_all['home_ind'] = df_all['home_ind'].astype(int)
    df_all['PDO'] = df_all['PDO'].astype(float)
    df_all['corsi'] = df_all['corsi'].astype(int)
    # df_all will give team by team stats for each game

    df_all.sort(['date', 'team'], ascending = True, inplace = True)

    df_games = df2[['home_team', 'away_team', 'date', 'home_team_win', 'home_spread']]
    df_games.columns = ['home_team', 'away_team', 'date', 'home_team_win', 'spread']
    df_games.sort(['date', 'home_team'], ascending = True, inplace = True)
    df_games.reset_index(drop = True, inplace = True)
    return df_all, df_games

def cumulative_stats(df_all, df_games, n = 10):
    '''
    pass in df_all and df_games from GbG_cumulative_df above
    will return final dataframe
    '''
    df_home = pd.DataFrame()

    for row in df_games.iterrows():
        # ONLY looking at games from 10/16 on
        if row[1].date >= pd.to_datetime('2016-10-16'):
            df = df_all[(df_all['team'] == row[1].home_team) & (df_all['date'] < row[1].date)]
            df.sort('date', ascending = False, inplace = True)
            sample = df.head(n)
            sample = sample.mean()
            sample = sample.to_frame()
            sample = sample.T
            sample.drop(['home_ind'], axis = 1, inplace = True)
            sample.columns = ['home_spread', 'home_wins', 'home_goals', 'home_shots', \
            'home_PP%', 'home_FOW%', 'home_PIM', 'home_hits', 'home_blocked', \
            'home_giveaways', 'home_takeaways', 'home_save%', 'home_shot%', \
            'home_PDO', 'home_corsi']
            df_home = df_home.append(sample)
            df_home.reset_index(drop=True, inplace = True)
        else:
            sample = np.zeros(15)
            sample = pd.DataFrame(sample)
            sample = sample.T
            sample.columns = \
            ['home_spread', 'home_wins', 'home_goals', 'home_shots', \
            'home_PP%', 'home_FOW%', 'home_PIM', 'home_hits', 'home_blocked', \
            'home_giveaways', 'home_takeaways', 'home_save%', 'home_shot%', \
            'home_PDO', 'home_corsi']
            # sample = pd.DataFrame([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], columns = \
            # ['home_spread', 'home_wins', 'home_goals', 'home_shots', \
            # 'home_PP%', 'home_FOW%', 'home_PIM', 'home_hits', 'home_blocked', \
            # 'home_giveaways', 'home_takeaways', 'home_save%', 'home_shot%', \
            # 'home_PDO', 'home_corsi'])
            df_home = df_home.append(sample)
            df_home.reset_index(drop = True, inplace = True)

    df_away = pd.DataFrame()

    for row in df_games.iterrows():
        # ONLY looking at games from 10/16 on
        if row[1].date >= pd.to_datetime('2016-10-16'):
            df = df_all[(df_all['team'] == row[1].away_team) & (df_all['date'] < row[1].date)]
            df.sort('date', ascending = False, inplace = True)
            sample = df.head(n)
            sample = sample.mean()
            sample = sample.to_frame()
            sample = sample.T
            sample.drop(['home_ind'], axis = 1, inplace = True)
            sample.columns = ['away_spread', 'away_wins', 'away_goals', 'away_shots', \
            'away_PP%', 'away_FOW%', 'away_PIM', 'away_hits', 'away_blocked', \
            'away_giveaways', 'away_takeaways', 'away_save%', 'away_shot%', \
            'away_PDO', 'away_corsi']
            df_away = df_away.append(sample)
            df_away.reset_index(drop=True, inplace = True)
        else:
            sample = np.zeros(15)
            sample = pd.DataFrame(sample)
            sample = sample.T
            sample.columns = \
            ['away_spread', 'away_wins', 'away_goals', 'away_shots', \
            'away_PP%', 'away_FOW%', 'away_PIM', 'away_hits', 'away_blocked', \
            'away_giveaways', 'away_takeaways', 'away_save%', 'away_shot%', \
            'away_PDO', 'away_corsi']
            df_away = df_away.append(sample)
            df_away.reset_index(drop = True, inplace = True)

        total = pd.merge(df_games, df_home, how = 'left', left_index = True, right_index = True)
        total = pd.merge(total, df_away, how = 'left', left_index = True, right_index = True)
        final = total[total['date'] >= pd.to_datetime('2016-10-16')]
        final.reset_index(drop = True, inplace = True)

    return final

if __name__ == '__main__':

    '''
    UNCOMMENT THIS SECTION TO RE-RUN AND GET UP-TO-DATE DATA

    df_dts, df_dp, df_ds = ws.get_data()

    dts, dp, ds = get_clean_data(df_dts, df_dp, df_ds)

    dts.to_csv('/home/jnell2/Documents/DataScienceImmersive/Final-Project/data/dts.csv')
    dp.to_csv('/home/jnell2/Documents/DataScienceImmersive/Final-Project/data/dp.csv')
    ds.to_csv('/home/jnell2/Documents/DataScienceImmersive/Final-Project/data/ds.csv')
    '''

    # cleaning up dataframes
    df_dts = pd.read_csv('data/dts.csv')
    df_dts = df_dts[['team', 'opponent', 'date', 'home_ind', 'W', 'L', \
    'GF', 'GA', 'SF', 'SA', 'PP%', 'PK%', 'FOW%']]
    df_dp = pd.read_csv('data/dp.csv')
    df_dp = df_dp[['team', 'opponent', 'date', 'PIM', 'penalties']]
    df_ds = pd.read_csv('data/ds.csv')
    df_ds = df_ds[['team', 'opponent', 'date', 'hits', 'blocked_shots', \
    'giveaways', 'takeaways', 'save%']]

    df_games = make_GbG_df(df_dts, df_dp, df_ds)
    df_all, df_games = GbG_cumulative_df(df_games)
    # df_all is the standard template, will add all cumulative stats to this

    df_final2 = cumulative_stats(df_all, df_games, 2)
    df_final5 = cumulative_stats(df_all, df_games, 5)
    df_final10 = cumulative_stats(df_all, df_games, 10)
