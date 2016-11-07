import pandas as pd
import numpy as np
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
    df.drop(['#', 'game', 'GP', 'T', 'OTL', 'points', 'PPG', 'PP', 'timesSH', 'PPGA', 'FOW', 'FOL'], axis=1, inplace=True)
    df = df.reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    idx = df.groupby('team')['date'].nsmallest(10).index.get_level_values(1)
    df['last10'] = np.where(df.index.isin(idx),1,0)
    idx2 = df.groupby('team')['date'].nsmallest(5).index.get_level_values(1)
    df['last5'] = np.where(df.index.isin(idx2),1,0)
    idx3 = df.groupby('team')['date'].nsmallest(2).index.get_level_values(1)
    df['last2'] = np.where(df.index.isin(idx3),1,0)
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

def clean_sts(df, df2):
    '''
    pass in appropriate dataframes from WebScraping.py
    will return the clean dfs for the season-by-season team summaries
    '''
    df['team'] = df['team'].str.split().str[-1]
    df.replace({'team': abbr_dict}, inplace = True)
    df['L'] = df['L'] + df['OTL']
    df['rank'] = df['rank'].astype(int)
    df['GP'] = df['GP'].astype(int)
    df['W'] = df['W'].astype(int)
    df['L'] = df['L'].astype(int)
    df['point%'] = df['point%'].astype(float)
    df['GF'] = df['GF'].astype(int)
    df['GA'] = df['GA'].astype(int)
    df['GF/GP'] = df['GF/GP'].astype(float)
    df['GA/GP'] = df['GA/GP'].astype(float)
    df['PP%'] = df['PP%'].astype(float)
    df['PK%'] = df['PK%'].astype(float)
    df['SF/GP'] = df['SF/GP'].astype(float)
    df['SA/GP'] = df['SA/GP'].astype(float)
    df['FOW%'] = df['FOW%'].astype(float)
    df['SA'] = df['SA/GP'] * df['GP'] # calculated approximate SA
    df.drop(['season', 'T', 'OTL', 'points', 'ROW'], axis=1, inplace=True)
    df2['team'] = df2['team'].str.split().str[-1]
    df2.replace({'team': abbr_dict}, inplace = True)
    df2 = df2[['rank', 'team']]
    df2.columns = ['LSrank', 'team']
    return df, df2

def clean_ss(df):
    '''
    pass in appropriate dataframe from WebScraping.py
    will return the clean df for the season-by-season shot report
    '''
    df['team'] = df['team'].str.split().str[-1]
    df.replace({'team': abbr_dict}, inplace = True)
    df['hits'] = df['hits'].astype(int)
    df['blocked_shots'] = df['blocked_shots'].astype(int)
    df['missed_shots'] = df['missed_shots'].astype(int)
    df['giveaways'] = df['giveaways'].astype(int)
    df['takeaways'] = df['takeaways'].astype(int)
    df['GF'] = df['GF'].astype(int)
    df['SF'] = df['SF'].astype(int)
    df['save%'] = df['save%'].astype(float)
    df['shot%'] = df['GF']/df['SF'] # calculated shot%
    df['PDO'] = df['save%'] + df['shot%'] # calculated PDO
    df.drop(['#', 'season', 'GP', 'W', 'L', 'T', 'OTL', 'points', 'FOW', 'FOL', 'FO', 'FOW%', 'GF'], axis=1, inplace=True)
    return df

def clean_sgf(df):
    '''
    pass in appropriate dataframe from WebScraping.py
    will return the clean df for the season-by-season goals for report
    '''
    df['team'] = df['team'].str.split().str[-1]
    df.replace({'team': abbr_dict}, inplace = True)
    df['GF55'] = df['GF55'].astype(int)
    df['GF54'] = df['GF54'].astype(int)
    df['GF45'] = df['GF45'].astype(int)
    df = df[['team', 'GF55', 'GF54', 'GF45']]
    return df

def clean_sga(df):
    '''
    pass in appropriate dataframe from WebScraping.py
    will return the clean df for the season-by-season goals against report
    '''
    df['team'] = df['team'].str.split().str[-1]
    df.replace({'team': abbr_dict}, inplace = True)
    df['GA55'] = df['GA55'].astype(int)
    df['GA54'] = df['GA54'].astype(int)
    df['GA45'] = df['GA45'].astype(int)
    df = df[['team', 'GA55', 'GA54', 'GA45']]
    return df

def get_clean_data(df_dts, df_dp, df_ds, df_sts, df_sts_past, df_ss, df_sgf, df_sga):
    '''
    pass in the 8 dataframes, in the order listed above
    will return the cleaned up dataframes
    '''
    df_dts = clean_dts(df_dts)
    df_dp = clean_dp(df_dp)
    df_ds = clean_ds(df_ds)
    df_sts, df_sts_past = clean_sts(df_sts, df_sts_past)
    df_ss = clean_ss(df_ss)
    df_sgf = clean_sgf(df_sgf)
    df_sga = clean_sga(df_sga)
    return df_dts, df_dp, df_ds, df_sts, df_sts_past, df_ss, df_sgf, df_sga

def make_GbG_df(df_dts, df_dp, df_ds):
    '''
    pass in the three appropriate dataframes containing game-by-game data
    will return a three dataframes with all relevant information for a GbG basis
    returned dataframes will contain games for past 10, 5, and 2 games, respectively
    '''
    df = pd.merge(df_dts, df_dp, how = 'left', on = ['team', 'opponent', 'date'])
    df = pd.merge(df, df_ds, how = 'left', on = ['team', 'opponent', 'date'])
    df['spread'] = df['GF']-df['GA']
    return df

def make_season_df(df_sts, df_sts_past, df_ss, df_sgf, df_sga):
    '''
    pass in the 5 appropriate dataframes containing season data
    will return a single dataframe containing all season data
    '''
    df = pd.merge(df_sts, df_sts_past, how = 'left', on = ['team'])
    df = pd.merge(df, df_ss, how = 'left', on = ['team'])
    df = pd.merge(df, df_sgf, how = 'left', on = ['team'])
    df = pd.merge(df, df_sga, how = 'left', on = ['team'])
    df['corsi'] = df['SF']-df['SA'] # corsi-type feature
    return df

def GbG_cumulative_df(df):
    '''
    pass in dataframe from make_GbG_df
    will return dataframe in desired format
    '''
    dfc = df.copy()
    df = df.loc[df['home_ind']==1]
    df['shot%'] = df['GF']/df['SF']
    df = df[['team', 'opponent', 'date', 'W', 'L', 'spread', 'GF', 'GA', 'SF', 'SA',
    'PP%', 'FOW%', 'PIM', 'hits', 'blocked_shots', 'giveaways', \
    'takeaways', 'save%', 'shot%', 'last10', 'last5', 'last2']]
    df.columns = ['home_team', 'away_team', 'date', 'home_team_win', 'away_team_win',\
     'spread','home_goals', 'away_goals', 'home_shots', 'away_shots', 'home_PP%', \
    'home_FOW%', 'home_PIM', 'home_hits', 'home_blocked', \
    'home_ga', 'home_ta', 'home_save%', 'home_shot%', 'last10', 'last5', 'last2']

    dfc['shot%'] = dfc['GF']/dfc['SF']
    dfc = dfc[['team', 'opponent', 'date', 'PP%', 'FOW%', 'PIM', \
    'hits', 'blocked_shots', 'giveaways', 'takeaways', 'save%', 'shot%']]
    dfc.columns = ['home_team', 'away_team', 'date', 'away_PP%', 'away_FOW%', \
    'away_PIM', 'away_hits', 'away_blocked', 'away_ga', 'away_ta', 'away_save%', \
    'away_shot%']

    df2 = pd.merge(df, dfc, how = 'left', on = ['home_team', 'away_team', 'date'])

    df_home = df2[['home_team', 'date', 'home_team_win', 'home_goals', 'home_shots',\
    'home_PP%', 'home_FOW%', 'home_PIM', 'home_hits', 'home_blocked', 'home_ga',\
    'home_ta', 'home_save%', 'home_shot%', 'last10', 'last5', 'last2']]
    df_home.columns = ['team', 'date', 'win', 'goals', 'shots', 'PP%', 'FOW%',\
    'PIM', 'hits', 'blocked', 'giveaways', 'takeaways', 'save%', 'shot%', \
    'last10', 'last5', 'last2']
    df_away = df2[['away_team', 'date', 'away_team_win', 'away_goals', 'away_shots', \
    'away_PP%', 'away_FOW%', 'away_PIM', 'away_hits', 'away_blocked',  'away_ga',\
    'away_ta', 'away_save%', 'away_shot%', 'last10', 'last5', 'last2']]
    df_away.columns = ['team', 'date', 'win', 'goals', 'shots', 'PP%', 'FOW%',\
    'PIM', 'hits', 'blocked', 'giveaways', 'takeaways', 'save%', 'shot%', \
    'last10', 'last5', 'last2']

    all_data = np.vstack((df_home, df_away))
    df_all = pd.DataFrame(all_data)
    df_all.columns = ['team', 'date', 'win', 'goals', 'shots', 'PP%', 'FOW%',\
    'PIM', 'hits', 'blocked', 'giveaways', 'takeaways', 'save%', 'shot%', \
    'last10', 'last5', 'last2']
    df_all['date'] = pd.to_datetime(df_all['date'])
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
    df_all['last10'] = df_all['last10'].astype(int)
    df_all['last5'] = df_all['last5'].astype(int)
    df_all['last2'] = df_all['last2'].astype(int)
    df_all['PDO'] = df_all['save%'] + df_all['shot%']
    # df_all will give team by team stats for each game

    # this will give game by game template
    df_games = df2[['home_team', 'away_team', 'date', 'home_team_win', 'spread']]

    return df_all, df_games

def make_cumulative_stats(df_all, df_games):
    '''
    pass in df_all, df_games that are returned from GbG_cumulative_df above
    will return 3 dataframes with cumulative stats on GbG template
    cumulative stats for past 10 games, 5 games, 2 games
    '''
    # adding cumulative stats: wins
    df_all = df_all.assign(elig=(df_all['win'] ==1) & df_all['last10']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['win'].sum(), on='team', rsuffix='_total_10')
    df_all.loc[~df_all['elig'],'win_total_10']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['win'] ==1) & df_all['last5']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['win'].sum(), on='team', rsuffix='_total_5')
    df_all.loc[~df_all['elig'],'win_total_5']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['win'] ==1) & df_all['last10']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['win'].sum(), on='team', rsuffix='_total_2')
    df_all.loc[~df_all['elig'],'win_total_2']=0
    df_all.drop('elig', axis = 1, inplace = True)
    # adding cumulative stats: goals
    df_all = df_all.assign(elig=(df_all['goals'] > 0) & df_all['last5']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['goals'].sum(), on='team', rsuffix='_total_10')
    df_all.loc[~df_all['elig'],'goals_total_10']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['goals'] > 0) & df_all['last5']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['goals'].sum(), on='team', rsuffix='_total_5')
    df_all.loc[~df_all['elig'],'goals_total_5']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['goals'] > 0) & df_all['last2']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['goals'].sum(), on='team', rsuffix='_total_2')
    df_all.loc[~df_all['elig'],'goals_total_2']=0
    df_all.drop('elig', axis = 1, inplace = True)
    # adding cumulative stats: shots
    df_all = df_all.assign(elig=(df_all['shots'] > 0) & df_all['last10']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['shots'].sum(), on='team', rsuffix='_total_10')
    df_all.loc[~df_all['elig'],'shots_total_10']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['shots'] > 0) & df_all['last5']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['shots'].sum(), on='team', rsuffix='_total_5')
    df_all.loc[~df_all['elig'],'shots_total_5']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['shots'] > 0) & df_all['last2']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['shots'].sum(), on='team', rsuffix='_total_2')
    df_all.loc[~df_all['elig'],'shots_total_2']=0
    df_all.drop('elig', axis = 1, inplace = True)
    # adding cumulative stats: PP% avg
    df_all = df_all.assign(elig=(df_all['PP%'] > -1) & df_all['last10']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['PP%'].mean(), on='team', rsuffix='_avg_10')
    df_all.loc[~df_all['elig'],'PP%_avg_10']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['PP%'] > -1) & df_all['last5']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['PP%'].mean(), on='team', rsuffix='_avg_5')
    df_all.loc[~df_all['elig'],'PP%_avg_5']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['PP%'] > -1) & df_all['last2']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['PP%'].mean(), on='team', rsuffix='_avg_2')
    df_all.loc[~df_all['elig'],'PP%_avg_2']=0
    df_all.drop('elig', axis = 1, inplace = True)
    # adding cumulative stats: FOW% avg
    df_all = df_all.assign(elig=(df_all['FOW%'] > -1) & df_all['last10']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['FOW%'].mean(), on='team', rsuffix='_avg_10')
    df_all.loc[~df_all['elig'],'FOW%_avg_10']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['FOW%'] > -1) & df_all['last5']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['FOW%'].mean(), on='team', rsuffix='_avg_5')
    df_all.loc[~df_all['elig'],'FOW%_avg_5']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['FOW%'] > -1) & df_all['last2']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['FOW%'].mean(), on='team', rsuffix='_avg_2')
    df_all.loc[~df_all['elig'],'FOW%_avg_2']=0
    df_all.drop('elig', axis = 1, inplace = True)
    # adding cumulative stats: PIM
    df_all = df_all.assign(elig=(df_all['PIM'] > 0) & df_all['last10']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['PIM'].sum(), on='team', rsuffix='_total_10')
    df_all.loc[~df_all['elig'],'PIM_total_10']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['PIM'] > 0) & df_all['last5']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['PIM'].sum(), on='team', rsuffix='_total_5')
    df_all.loc[~df_all['elig'],'PIM_total_5']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['PIM'] > 0) & df_all['last2']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['PIM'].sum(), on='team', rsuffix='_total_2')
    df_all.loc[~df_all['elig'],'PIM_total_2']=0
    df_all.drop('elig', axis = 1, inplace = True)
    # adding cumulative stats: hits
    df_all = df_all.assign(elig=(df_all['hits'] > 0) & df_all['last10']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['hits'].sum(), on='team', rsuffix='_total_10')
    df_all.loc[~df_all['elig'],'hits_total_10']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['hits'] > 0) & df_all['last5']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['hits'].sum(), on='team', rsuffix='_total_5')
    df_all.loc[~df_all['elig'],'hits_total_5']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['hits'] > 0) & df_all['last2']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['hits'].sum(), on='team', rsuffix='_total_2')
    df_all.loc[~df_all['elig'],'hits_total_2']=0
    df_all.drop('elig', axis = 1, inplace = True)
    # adding cumulative stats: blocked shots
    df_all = df_all.assign(elig=(df_all['blocked'] > 0) & df_all['last10']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['blocked'].sum(), on='team', rsuffix='_total_10')
    df_all.loc[~df_all['elig'],'blocked_total_10']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['blocked'] > 0) & df_all['last5']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['blocked'].sum(), on='team', rsuffix='_total_5')
    df_all.loc[~df_all['elig'],'blocked_total_5']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['blocked'] > 0) & df_all['last2']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['blocked'].sum(), on='team', rsuffix='_total_2')
    df_all.loc[~df_all['elig'],'blocked_total_2']=0
    df_all.drop('elig', axis = 1, inplace = True)
    # adding cumulative stats: givaways
    df_all = df_all.assign(elig=(df_all['giveaways'] > 0) & df_all['last10']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['giveaways'].sum(), on='team', rsuffix='_total_10')
    df_all.loc[~df_all['elig'],'giveaways_total_10']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['giveaways'] > 0) & df_all['last5']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['giveaways'].sum(), on='team', rsuffix='_total_5')
    df_all.loc[~df_all['elig'],'giveaways_total_5']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['giveaways'] > 0) & df_all['last2']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['giveaways'].sum(), on='team', rsuffix='_total_2')
    df_all.loc[~df_all['elig'],'giveaways_total_2']=0
    df_all.drop('elig', axis = 1, inplace = True)
    # adding cumulative stats: takeaways
    df_all = df_all.assign(elig=(df_all['takeaways'] > 0) & df_all['last10']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['takeaways'].sum(), on='team', rsuffix='_total_10')
    df_all.loc[~df_all['elig'],'takeaways_total_10']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['takeaways'] > 0) & df_all['last5']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['takeaways'].sum(), on='team', rsuffix='_total_5')
    df_all.loc[~df_all['elig'],'takeaways_total_5']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['takeaways'] > 0) & df_all['last2']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['takeaways'].sum(), on='team', rsuffix='_total_2')
    df_all.loc[~df_all['elig'],'takeaways_total_2']=0
    df_all.drop('elig', axis = 1, inplace = True)
    # adding cumulative stats: save% avg
    df_all = df_all.assign(elig=(df_all['save%'] > -1) & df_all['last10']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['save%'].mean(), on='team', rsuffix='_avg_10')
    df_all.loc[~df_all['elig'],'save%_avg_10']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['save%'] > -1) & df_all['last5']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['save%'].mean(), on='team', rsuffix='_avg_5')
    df_all.loc[~df_all['elig'],'save%_avg_5']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['save%'] > -1) & df_all['last2']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['save%'].mean(), on='team', rsuffix='_avg_2')
    df_all.loc[~df_all['elig'],'save%_avg_2']=0
    df_all.drop('elig', axis = 1, inplace = True)
    # adding cumulative stats: shot% avg
    df_all = df_all.assign(elig=(df_all['shot%'] > -1) & df_all['last10']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['shot%'].mean(), on='team', rsuffix='_avg_10')
    df_all.loc[~df_all['elig'],'shot%_avg_10']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['shot%'] > -1) & df_all['last5']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['shot%'].mean(), on='team', rsuffix='_avg_5')
    df_all.loc[~df_all['elig'],'shot%_avg_5']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['shot%'] > -1) & df_all['last2']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['shot%'].mean(), on='team', rsuffix='_avg_2')
    df_all.loc[~df_all['elig'],'shot%_avg_2']=0
    df_all.drop('elig', axis = 1, inplace = True)
    # adding cumulative stats: PDO avg
    df_all = df_all.assign(elig=(df_all['PDO'] > 0) & df_all['last10']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['PDO'].mean(), on='team', rsuffix='_avg_10')
    df_all.loc[~df_all['elig'],'PDO_avg_10']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['PDO'] > 0) & df_all['last5']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['PDO'].mean(), on='team', rsuffix='_avg_5')
    df_all.loc[~df_all['elig'],'PDO_avg_5']=0
    df_all.drop('elig', axis = 1, inplace = True)
    df_all = df_all.assign(elig=(df_all['PDO'] > 0) & df_all['last2']==1)
    df_all = df_all.join(df_all[df_all['elig']].groupby('team')['PDO'].mean(), on='team', rsuffix='_avg_2')
    df_all.loc[~df_all['elig'],'PDO_avg_2']=0
    df_all.drop('elig', axis = 1, inplace = True)

    # need to get 3 tables containing only cumulative stats for total games specified
    df10 = df_all.loc[df_all['last10']==1]
    df5 = df_all.loc[df_all['last5']==1]
    df2 = df_all.loc[df_all['last2']==1]

    return df10, df5, df2

    # df = pd.merge(df_games, df_all, how = 'left', left_on = ['home_team', ])


# def games_final(df_games, df_stats):
#     '''
#     pass in dataframes for df_games and df_stats (returned from GbG_cumulative_df function)
#     will return final game by game dataframes
#     '''
#     df_final = pd.merge(df_games, df_stats, how = 'left', on = ['team']
#     pass

if __name__ == '__main__':

    df_dts, df_dp, df_ds, df_sts, df_sts_past, df_ss, df_sgf, df_sga = ws.get_data()

    df_dts, df_dp, df_ds, df_sts, df_sts_past, df_ss, df_sgf, df_sga = \
    get_clean_data(df_dts, df_dp, df_ds, df_sts, df_sts_past, df_ss, df_sgf, df_sga)

    df_games = make_GbG_df(df_dts, df_dp, df_ds)
    df_season = make_season_df(df_sts, df_sts_past, df_ss, df_sgf, df_sga)

    df_all, df_games = GbG_cumulative_df(df_games)
    # df_games is the standard template, will add all cumulative stats to this

    df10, df5, df2 = make_cumulative_stats(df_all, df_games)
