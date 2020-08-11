# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:00:06 2020

@author: Andrew
"""

import pandas as pd
import numpy as np
import datetime

import requests
from bs4 import BeautifulSoup

import re

# myurl='https://sdgs.un.org/goals'

# req = requests.get(myurl)

# soup = BeautifulSoup(
#     req.text, "xml"
# )  

# content = soup.find('div',{'class':'view-content row'})

# data = pd.DataFrame(columns = ["goal","partnerships","events","publications","targets"], index=np.arange(17))

# cards = content.find_all('div',{'data-toggle':'modal'})

# for idx,card in enumerate(cards):
#     print(idx)
#     data["goal"][idx]= card.find('p',{'class':'goal-text'}).text
    
#     info = card.find_all('span')
#     data["partnerships"][idx]=int(info[1].text)
#     data["events"][idx]=int(info[2].text)
#     data["publications"][idx]=int(info[3].text)
#     data["targets"][idx]=int(info[4].text)
    
# data.to_excel(r'C:\Users\Andrew\Documents\PythonScripts\climate work\SDG overview.xlsx')

data = pd.read_excel(r'C:\Users\Andrew\Documents\PythonScripts\climate work\SDG overview.xlsx')
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

myurl = 'https://sustainabledevelopment.un.org/partnerships/'

cols = ["goal","partner_name","url","entity","partners_of_partner","description","date_start",
        "date_end","geographic_coverage","beneficiary_countires","based"]

# data_partnerships = pd.DataFrame(columns=cols, index=np.arange(15000))
data_partnerships = pd.read_excel(r'C:\Users\Andrew\Documents\PythonScripts\climate work\SDG Partnerships.xlsx',index_col=0)

# driver = webdriver.Chrome()

goals = ["goal1","goal2","goal3","goal4","goal5","goal6","goal7","goal8","goal9","goal10",
         'goal11','goal12','goal13','goal14','goal15','goal16','goal17']

goal_dict = dict(zip(goals,data["goal"].unique()))

# import time

#for grabbing partner names and urls
# df_idx=0
# for goal in goals[5:]:
#     print(goal)
#     driver.get(myurl+goal)
#     time.sleep(1)
    
#     soup = BeautifulSoup(driver.page_source,'html.parser')
#     total = soup.find('div',{'id':'moreButton'})
    
#     scroll = data[data["goal"]==goal_dict[goal]]["partnerships"]
#     scroll = int(scroll)/20    
    
#     scroll_times = 0
#     print('scrolling...')
#     while(scroll_times <= round(scroll)):
#         try:
#             time.sleep(8)
#             driver.find_element_by_css_selector('#theMoreButton').click()
#             scroll_times+=1
#         except:
#             print("scroll end")
#             break
                
#     soup = BeautifulSoup(driver.page_source,'html.parser')
    
#     contents = soup.find_all('div',{'class':'projectRow'})
#     print('taking data...')
    
#     #get urls
#     for content in contents:
#         data_partnerships['goal'][df_idx]=goal_dict[goal]
#         data_partnerships['partner_name'][df_idx]=content.find('a').text
#         data_partnerships['url'][df_idx]='https://sustainabledevelopment.un.org'+content.find('a')['href']
#         df_idx+=1
    
# data_partnerships.to_excel(r'C:\Users\Andrew\Documents\PythonScripts\climate work\SDG Partnerships.xlsx')

#for grabbing everything else using just bs4 (could get resources but that requires selenium). Check before running~~!

urls = data_partnerships["url"] #can't do unique... duh
data_partnerships["based"]=""

IP = "54.38.218.213"
df_idx=0
for url in urls:
    print(str(df_idx) + ", "+url)
    req = requests.get(url,proxies={'http':IP})
    soup = BeautifulSoup(req.text,'html.parser')
    try:
        #get description area
        data_partnerships['description'][df_idx]= soup.find('div',{'id':'intro'}).text

        column = soup.find('div',{'class':'homeRight'})
        column_headers = column.find_all('div',{'class':'columnHeader'})
        column_data = column.find_all('div',{'class':'wrap','style':'padding:10px;'})
        
        #get column order
        basic_idx = -1
        entity_idx = -1
        partners_idx = -1
        geo_idx = -1
        based_idx = -1
        bene_idx = -1
        
        for idx,header in enumerate(column_headers):
            if header.text == "Basic information":
                basic_idx = idx
            if header.text == "Partners":
                partners_idx = idx
            if header.text == "Entity":
                entity_idx = idx
            if header.text == "Countries" or header.text =="Beneficiary countries":
                bene_idx = idx
            if header.text == "Headquarters":
                based_idx = idx
            if header.text == "Geographical coverage":
                geo_idx = idx
                
        search_cols = ["entity","partners_of_partner","geographic_coverage","beneficiary_countires","based"]
        all_idx = [entity_idx, partners_idx, geo_idx, bene_idx,based_idx]
        search_dict = dict(zip(search_cols,all_idx))
            
        #get project dates
        try:
            dates = column_data[basic_idx].find_all('div',{'class':'inforow'})
            data_partnerships['date_start'][df_idx]=dates[0].text[7:]
            if len(dates)>1:
                data_partnerships['date_end'][df_idx]=dates[1].text[12:]
        except IndexError:
            #odd column type
            data_partnerships['date_start'][df_idx]=column_data[1].text.split('-')[1].split(':')[1].strip()
            data_partnerships['date_end'][df_idx]=column_data[1].text.split('-')[2].split('\n')[0].strip()
        
        #get all other data based on index
        for col in search_cols:
            if search_dict[col]!=-1:
                data_partnerships[col][df_idx]=column_data[search_dict[col]].text
            
    except AttributeError:
        print('404 error')
        
    df_idx+=1
    
data_partnerships = data_partnerships.fillna("null")
data_partnerships.to_pickle(r'C:\Users\Andrew\Documents\PythonScripts\climate work\SDG Partnerships_raw.pkl')