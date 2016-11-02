from bs4 import BeautifulSoup
from requests import get
import pandas as pd
import numpy as np
from selenium import webdriver

# make sure that you have Google Chrome and selenium installed!

def get_soup(url):
    '''
    pass in a url
    will return html output of BeautifulSoup
    uses selenium
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
