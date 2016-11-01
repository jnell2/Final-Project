from pymongo import MongoClient
from bs4 import BeautifulSoup
from requests import get
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

# make sure that you have Google Chrome and selenium installed!

def get_soup(url):
    '''
    pass in a url
    will return html output of BeautifulSoup
    '''
    d = webdriver.Chrome()
    # this will launch a new Chrome browser (maybe multiple)
    # don't exit out until process is finished running
    d.get(url)
    result = d.page_source
    soup = BeautifulSoup(result, 'html.parser')
    return soup

def Date_TS(soup):
    '''
    pass in html output from BeautifulSoup
    will return a pandas dataframe
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
    '''
    full_table = soup.find("table", {"class": "stat-table"})
    cols = ['#', 'team', 'season', 'GP', 'W', 'L', 'T', 'OTL', 'points', \
    'ROW', 'point%', 'GF', 'GA', 'GF/GP', 'GA/GP', 'PP%','PK%', 'SF/GP', \
    'SA/GP', 'FOW%']
    mat = np.ndarray((0,20))
    lst = soup.find_all('tr', attrs = {'class': 'standard-row'})
    for i in lst:
        row = [elem.text for elem in i.find_all('td')]
        mat = np.vstack((mat, row))
    df = pd.DataFrame(mat,columns = cols)
    return df

def Season_Shots(soup):
    '''
    pass in html output from BeautifulSoup
    will return a pandas dataframe
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

def get_soup_standings(url):
    '''
    pass in a url
    will return html output of BeautifulSoup
    '''
    d = webdriver.Chrome()
    # this will launch a new Chrome browser (maybe multiple)
    # don't exit out until process is finished running
    d.get(url)
    elm = browser.find_element_by_link_text('LEAGUE')
    browswer.implicitly_wait(5)
    elm.click()
    result = d.page_source
    soup = BeautifulSoup(result, 'html.parser')
    return soup

def Season_Standings(soup):
    '''
    pass in html output from BeautifulSoup
    will return a pandas dataframe
    '''
    full_table = soup.find("section", {"class": "g5-component--standings__full-view"})
    cols = ['NHL', 'GP', 'W', 'L', 'OT', 'points', 'ROW', 'GF', 'GA', 'diff', \
    'home', 'away', 'S/O', 'L10', 'streak', 'last', 'next']
    mat = np.ndarray((0,17))
    lst = soup.find_all('tr', attrs = {'class': ""})
    for i in lst:
        row = [elem.text for elem in i.find_all('td')]
        mat = np.vstack((mat, row))
    df = pd.DataFrame(mat,columns = cols)
    return df


if __name__ == '__main__':
    # client = MongoClient()
    # db = client['']
    # table = db['']

    date1 = '2016-10-01' # in format YYYY-MM-DD
    date2 = '2016-10-31' # in format YYYY-MM-DD
    season1 = '20162017' # in format YYYYyyyy (ie: 20162017)
    season2 = '20162017' # in format YYYYyyyy (ie: 20162017)

    Date_TeamSummary_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=teamsummary&reportType=game&startDate={}&endDate={}&filter=gamesPlayed,gte,&sort=wins,points'.format(date1, date2)
    Date_Penalties_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=penalties&reportType=game&startDate={}&endDate={}&filter=gamesPlayed,gte,&sort=penaltyMinutes'.format(date1, date2)
    Date_Shots_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=realtime&reportType=game&startDate={}&endDate={}&filter=gamesPlayed,gte,&sort=hits'.format(date1, date2)
    Season_TeamSummary_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=teamsummary&reportType=season&seasonFrom={}&seasonTo={}&filter=gamesPlayed,gte,&sort=wins,points'.format(season1, season2)
    Season_Shots_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=realtime&reportType=season&seasonFrom={}&seasonTo={}&filter=gamesPlayed,gte,&sort=hits'.format(season1, season2)
    Season_GF_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=goalsbystrength&reportType=season&seasonFrom={}&seasonTo={}&filter=gamesPlayed,gte,&sort=goalsFor'.format(season1, season2)
    Season_GA_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=goalsagainstbystrength&reportType=season&seasonFrom={}&seasonTo={}&filter=gamesPlayed,gte,&sort=goalsAgainst'.format(season1, season2)
    Season_Standing_url = 'https://www.nhl.com/standings/league'

    soup_dts = get_soup(Date_TeamSummary_url)
    df_dts = Date_TS(soup_dts)

    soup_dp = get_soup(Date_Penalties_url)
    df_dp = Date_Penalties(soup_dp)

    soup_ds = get_soup(Date_Shots_url)
    df_ds = Date_Shots(soup_ds)

    soup_sts = get_soup(Season_TeamSummary_url)
    df_sts = Season_TS(soup_sts)

    soup_ss = get_soup(Season_Shots_url)
    df_ss = Season_Shots(soup_ss)

    soup_sgf = get_soup(Season_GF_url)
    df_sgf = Season_GF_Strength(soup_sgf)

    soup_sga = get_soup(Season_GA_url)
    df_sga = Season_GA_Strength(soup_sga)

    soup_stand = get_soup(Season_Standing_url)
    df_stand = Season_Standings(soup_stand)
