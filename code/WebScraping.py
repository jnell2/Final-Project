from bs4 import BeautifulSoup
from requests import get
import pandas as pd
import numpy as np
import datetime
import dateutil.relativedelta
from selenium import webdriver
from selenium.webdriver.common.by import By

# make sure that you have Google Chrome and selenium installed!

def get_soup(url, page):
    '''
    pass in a url
    will return html output of BeautifulSoup
    uses selenium
    '''
    d = webdriver.Chrome()
    # this will launch a new Chrome browser (maybe multiple)
    # don't exit out until process is finished running
    d.get(url)
    d.find_element(By.XPATH, "//select[@class='pager-select']/option[text()='{}']".format(page)).click()
    result = d.page_source
    soup = BeautifulSoup(result, 'html.parser')
    d.close()
    return soup

def Date_TS(soup):
    '''
    pass in html output from BeautifulSoup
    will return a pandas dataframe
    this gets us the df for the game-by-game team summary
    '''
    full_table = soup.find("table", {"class": "stat-table"})
    cols = ['#', 'team', 'game', 'opponent', 'GP', 'W', 'L', 'T', 'OTL', 'points', \
    'GF', 'GA', 'SF', 'SA', 'PPG', 'PP', 'PP%', 'timesSH', 'PPGA', 'PK%', \
    'FOW', 'FOL', 'FOW%']
    mat = np.ndarray((0,23))
    lst = soup.find_all('tr', attrs = {'class': 'standard-row'})
    for i in lst:
        row = [elem.text for elem in i.find_all('td')]
        mat = np.vstack((mat, row))
    df = pd.DataFrame(mat,columns = cols)
    return df

def Date_Penalties(soup):
    '''
    pass in html output from BeautifulSoup
    will return a pandas dataframe
    this gets us the df for the game-by-game penalties report
    '''
    full_table = soup.find("table", {"class": "stat-table"})
    cols = ['#', 'team', 'game', 'opponent', 'GP', 'W', 'L', 'T', 'OTL', 'points', \
    'PIM', 'penalties', 'minor', 'major', 'misconduct', 'game_misconduct', 'match']
    mat = np.ndarray((0,17))
    lst = soup.find_all('tr', attrs = {'class': 'standard-row'})
    for i in lst:
        row = [elem.text for elem in i.find_all('td')]
        mat = np.vstack((mat, row))
    df = pd.DataFrame(mat,columns = cols)
    return df

def Date_Shots(soup):
    '''
    pass in html output from BeautifulSoup
    will return a pandas dataframe
    this gets us the df for the game-by-game shots report
    '''
    full_table = soup.find("table", {"class": "stat-table"})
    cols = ['#', 'team', 'game', 'opponent', 'GP', 'W', 'L', 'T', 'OTL', 'points', \
    'hits', 'blocked_shots', 'giveaways', 'takeaways', 'FOW', 'FOL', 'FO', \
    'FOW%', 'SF', 'GF', 'save%']
    mat = np.ndarray((0,21))
    lst = soup.find_all('tr', attrs = {'class': 'standard-row'})
    for i in lst:
        row = [elem.text for elem in i.find_all('td')]
        mat = np.vstack((mat, row))
    df = pd.DataFrame(mat,columns = cols)
    return df

def Season_TS(soup):
    '''
    pass in html output from BeautifulSoup
    will return a pandas dataframe
    this gets us the df for the season-by-season team summary
    '''
    full_table = soup.find("table", {"class": "stat-table"})
    cols = ['rank', 'team', 'season', 'GP', 'W', 'L', 'T', 'OTL', 'points', \
    'ROW', 'point%', 'GF', 'GA', 'GF/GP', 'GA/GP', 'PP%','PK%', 'SF/GP', \
    'SA/GP', 'FOW%']
    mat = np.ndarray((0,20))
    lst = soup.find_all('tr', attrs = {'class': 'standard-row'})
    for i in lst:
        row = [elem.text for elem in i.find_all('td')]
        mat = np.vstack((mat, row))
    df = pd.DataFrame(mat,columns = cols)
    return df
    # can get standings from this table, as it is sorted by points, wins

def Season_Shots(soup):
    '''
    pass in html output from BeautifulSoup
    will return a pandas dataframe
    this gets us the df for the season-by-season shots report
    '''
    full_table = soup.find("table", {"class": "stat-table"})
    cols = ['#', 'team', 'season', 'GP', 'W', 'L', 'T', 'OTL', 'points', \
    'hits', 'blocked_shots', 'missed_shots', 'giveaways', 'takeaways', 'FOW', \
    'FOL', 'FO', 'FOW%', 'SF', 'GF', 'save%']
    mat = np.ndarray((0,21))
    lst = soup.find_all('tr', attrs = {'class': 'standard-row'})
    for i in lst:
        row = [elem.text for elem in i.find_all('td')]
        mat = np.vstack((mat, row))
    df = pd.DataFrame(mat,columns = cols)
    return df

def Season_GF_Strength(soup):
    '''
    pass in html output from BeautifulSoup
    will return a pandas dataframe
    this gets us the df for the season-by-season goal for by strength report
    '''
    full_table = soup.find("table", {"class": "stat-table"})
    cols = ['#', 'team', 'season', 'GP', 'W', 'L', 'T', 'OTL', 'points', \
    'GF', 'GA', 'GF55', 'GF54', 'GF53', 'GF44', 'GF43', 'GF33', 'GF34', 'GF35', \
    'GF36', 'GF45', 'GF46', 'GF56', 'GF65', 'GF64', 'GF63', 'GF/GP']
    mat = np.ndarray((0,27))
    lst = soup.find_all('tr', attrs = {'class': 'standard-row'})
    for i in lst:
        row = [elem.text for elem in i.find_all('td')]
        mat = np.vstack((mat, row))
    df = pd.DataFrame(mat,columns = cols)
    return df

def Season_GA_Strength(soup):
    '''
    pass in html output from BeautifulSoup
    will return a pandas dataframe
    this gets us the df for the season-by-season goals against by strength report
    '''
    full_table = soup.find("table", {"class": "stat-table"})
    cols = ['#', 'team', 'season', 'GP', 'W', 'L', 'T', 'OTL', 'points', \
    'GF', 'GA', 'GA55', 'GA54', 'GA53', 'GA44', 'GA43', 'GA33', 'GA34', 'GA35', \
    'GA36', 'GA45', 'GA46', 'GA56', 'GA65', 'GA64', 'GA63', 'GA/GP']
    mat = np.ndarray((0,27))
    lst = soup.find_all('tr', attrs = {'class': 'standard-row'})
    for i in lst:
        row = [elem.text for elem in i.find_all('td')]
        mat = np.vstack((mat, row))
    df = pd.DataFrame(mat,columns = cols)
    return df

def Date_TS_loop(url, num_pages):
    '''
    pass in url and number of pages that are in the table (found below the table at the URL)
    will return a dataframe with all of the pages together
    gives us the df for the game-by-game team summary (total)
    '''
    # creating a dictionary of soup outputs
    # length should be equal to number of pages in table
    soup_dct = {}
    for i in range(1,num_pages+1):
        soup_dct[i] = get_soup(url, page=i)
    # creating a dictionary of dataframes
    # length should be equal to number of pages in table
    df_dct = {}
    for p, soup in soup_dct.items():
        df_dct[p] = Date_TS(soup)
    # appending all dataframes together
    for i in range(2,num_pages+1):
        df_dct[1] = df_dct[1].append(df_dct[i])
    df_dts = df_dct[1]
    return df_dts
    del soup_dct
    del df_dct

def Date_Penalties_loop(url, num_pages):
    '''
    pass in url and number of pages that are in the table (found below the table at the URL)
    will return a dataframe with all of the pages together
    gives us the df for the game-by-game penalties report (total)
    '''
    # creating a dictionary of soup outputs
    # length should be equal to number of pages in table
    soup_dct = {}
    for i in range(1,num_pages+1):
        soup_dct[i] = get_soup(url, page=i)
    # creating a dictionary of dataframes
    # length should be equal to number of pages in table
    df_dct = {}
    for p, soup in soup_dct.items():
        df_dct[p] = Date_Penalties(soup)
    # appending all dataframes together
    for i in range(2,num_pages+1):
        df_dct[1] = df_dct[1].append(df_dct[i])
    df_dp = df_dct[1]
    return df_dp
    del soup_dct
    del df_dct

def Date_Shots_loop(url, num_pages):
    '''
    pass in url and number of pages that are in the table (found below the table at the URL)
    will return a dataframe with all of the pages together
    gives us the df for the game-by-game shots report (total)
    '''
    # creating a dictionary of soup outputs
    # length should be equal to number of pages in table
    soup_dct = {}
    for i in range(1,num_pages+1):
        soup_dct[i] = get_soup(url, page=i)
    # creating a dictionary of dataframes
    # length should be equal to number of pages in table
    df_dct = {}
    for p, soup in soup_dct.items():
        df_dct[p] = Date_Shots(soup)
    # appending all dataframes together
    for i in range(2,num_pages+1):
        df_dct[1] = df_dct[1].append(df_dct[i])
    df_ds = df_dct[1]
    return df_ds
    del soup_dct
    del df_dct

def get_data():

    date2 = datetime.date.today().strftime('%Y-%m-%d') # in format YYYY-MM-DD, this is the current date
    d = datetime.datetime.strptime(date2, "%Y-%m-%d")
    date1 = (d - dateutil.relativedelta.relativedelta(months=2)).strftime('%Y-%m-%d') #in format YYYY-MM-DD, this is 2 months before todays date

    current_year = datetime.date.today().year # in format YYYY
    prior_year = current_year - 1
    prior_2year = current_year - 2
    next_year = current_year + 1

    if datetime.date.today().month >= 9:
        season = str(current_year) + str(next_year)
        past_season = str(prior_year) + str(current_year)
    else:
        season = str(prior_year) + str(current_year)
        past_season = str(prior_2year) + str(prior_year)

    Date_TeamSummary_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=teamsummary&reportType=game&startDate={}&endDate={}&filter=gamesPlayed,gte,&sort=wins,points'.format(date1, date2)
    Date_Penalties_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=penalties&reportType=game&startDate={}&endDate={}&filter=gamesPlayed,gte,&sort=penaltyMinutes'.format(date1, date2)
    Date_Shots_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=realtime&reportType=game&startDate={}&endDate={}&filter=gamesPlayed,gte,&sort=hits'.format(date1, date2)
    Season_TeamSummary_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=teamsummary&reportType=season&seasonFrom={}&seasonTo={}&filter=gamesPlayed,gte,&sort=points,wins,gamesPlayed'.format(season, season)
    Season_TeamSummary_past_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=teamsummary&reportType=season&seasonFrom={}&seasonTo={}&filter=gamesPlayed,gte,&sort=points,wins,gamesPlayed'.format(past_season, past_season)
    Season_Shots_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=realtime&reportType=season&seasonFrom={}&seasonTo={}&filter=gamesPlayed,gte,&sort=hits'.format(season, season)
    Season_GF_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=goalsbystrength&reportType=season&seasonFrom={}&seasonTo={}&filter=gamesPlayed,gte,&sort=goalsFor'.format(season, season)
    Season_GA_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=goalsagainstbystrength&reportType=season&seasonFrom={}&seasonTo={}&filter=gamesPlayed,gte,&sort=goalsAgainst'.format(season, season)

    soup = get_soup(Date_TeamSummary_url, page = 1)
    pages = soup.find('select', attrs = {'class': 'pager-select'})
    num_pages = int(pages.text[-1])

    df_dts = Date_TS_loop(Date_TeamSummary_url, num_pages)
    df_dp = Date_Penalties_loop(Date_Penalties_url, num_pages)
    df_ds = Date_Shots_loop(Date_Shots_url, num_pages)

    soup_sts = get_soup(Season_TeamSummary_url, page=1)
    df_sts = Season_TS(soup_sts)
    soup_sts_past = get_soup(Season_TeamSummary_past_url, page=1)
    df_sts_past = Season_TS(soup_sts_past)

    soup_ss = get_soup(Season_Shots_url, page=1)
    df_ss = Season_Shots(soup_ss)

    soup_sgf = get_soup(Season_GF_url, page=1)
    df_sgf = Season_GF_Strength(soup_sgf)

    soup_sga = get_soup(Season_GA_url, page=1)
    df_sga = Season_GA_Strength(soup_sga)

    return df_dts, df_dp, df_ds, df_sts, df_sts_past, df_ss, df_sgf, df_sga

if __name__ == '__main__':

    df_dts, df_dp, df_ds, df_sts, df_sts_past, df_ss, df_sgf, df_sga = get_data()

    # df_dts.to_csv('/home/jnell2/Documents/DataScienceImmersive/Final-Project/data/DateTeamSummary.csv')
