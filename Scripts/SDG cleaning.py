# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 10:43:22 2020

@author: Andrew
"""

import pandas as pd
import numpy as np
import datetime

df = pd.read_pickle(r'C:\Users\Andrew\Documents\PythonScripts\climate work\SDG article\SDG-analysis\SDG Partnerships_raw.pkl')

'''stripping'''
cols = ["goal","partner_name","url","entity","partners_of_partner","description","date_start",
        "date_end","geographic_coverage","beneficiary_countires","based"]

import math 

for col in cols[1:]:
    print(col)
    df[col] = df[col].apply(lambda x: x.replace('\n'," ").strip())

df = df[:14249]

'''date stuff'''
#fix ame: scraping error
dates_data = df["date_start"]
for idx,date in enumerate(dates_data):
    try:
        pd.to_datetime(date)
    except:
        # print(date)
        if len(date) > 10:
            df["date_end"][idx] = date.split(':')[1].split(' - ')[1].strip()
            df["date_start"][idx] = date.split(':')[1].split(' - ')[0].strip()

# df.drop("impact",axis=1, inplace=True)
df["date_start"] = df["date_start"].apply(lambda x: x.replace('/ ','/').replace(' /','/').replace('(','').replace(')','').replace('  ',' '))
df["date_end"] = df["date_end"].apply(lambda x: x.replace('/ ','/').replace(' /','/').replace('(','').replace(')','').replace('  ',' '))

date_dict={'diciembre':'December','decembre':'December','abril':'April','agosto':'August','enero':'january',
            'marzo':'march','febrero':'february','febuary':'february','janvier':'january',
            'mat':'may','mayo':'may','septembre':'september','junio':'june','julio':'july','octubre':'october'}

def replace_date_spellings(x):
    for t in date_dict:
        if t in x.lower():
            x = x.lower().replace(t, date_dict[t])
    return x

df["date_start"] = df["date_start"].apply(lambda x: replace_date_spellings(x))
df["date_end"] = df["date_end"].apply(lambda x: replace_date_spellings(x))

total_ongoing = []
def replace_ongoing(x):
    if 'ongoing' in x.lower() or 'contin' in x.lower():
        print(x)
        total_ongoing.append(x)
        return '12-31-2030'
    else:
        return x

df["date_end"] = df["date_end"].apply(lambda x: replace_ongoing(x))
print(len(total_ongoing))

unclean_end = []
def clean_date_end(x):
    try:
        x = pd.to_datetime(x)
    except:
        x="ignore"
        unclean_end.append(x)
    return x

unclean = []
def clean_date_start(x):
    try:
        x = pd.to_datetime(x)
    except:
        x="ignore"
        unclean.append(x)
    return x

df["date_end"] = df["date_end"].apply(lambda x: clean_date_end(x))
df["date_start"] = df["date_start"].apply(lambda x: clean_date_start(x))

def get_year(x):
    try:
        year = x.year
        return year
    except:
        print('ignored')
        return 'ignore'

df["year_start"] = df["date_start"].apply(lambda x: get_year(x))
df["year_end"] = df["date_end"].apply(lambda x: get_year(x))

df["length_of_partnership"] = 0
for idx,end in enumerate(df["year_end"]):
    try:
        df["length_of_partnership"][idx] = end - df["year_start"][idx]
    except:
        df["length_of_partnership"][idx] = 'ignore'

'''desc stuff'''
#translate other languages to english description
from googletrans import Translator
from langdetect import detect
import time
translator = Translator()

vi_errors= []
#loop through with pauses, can't create a bulk string cause some languages are different that others. 
i = 0
for desc in df["description"][i:]:
    print(i)
    if detect(desc)=='en' or detect(desc)=='vi': #vi is some character error, already in english though. 
        print('is already english')
        if detect(desc)=='vi':
            print('vi error')
            vi_errors.append(desc)
    else:
        print('translating')
        try:
            df["description"][i] = translator.translate(desc).text
        except:
            print('waiting...')
            break #break when connection error occurs
    i+=1

import string
def remove_weird_letters(x):
    new_string = ""
    accepted_char = list(string.ascii_lowercase) + list(string.ascii_uppercase) + list(string.punctuation) + list(" ")
    
    for letter in x:
        for char in accepted_char:
            if char in letter:
                new_string = new_string + char
                break
    return new_string

df["description"] = df["description"].apply(lambda x: remove_weird_letters(x))

df.to_pickle(r'C:\Users\Andrew\Documents\PythonScripts\climate work\SDG article\SDG-analysis\SDG Partnerships_clean.pkl')
