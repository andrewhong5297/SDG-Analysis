# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:47:18 2020

@author: Andrew
"""

import pandas as pd
import numpy as np
import re

df = pd.read_pickle(r'C:\Users\Andrew\Documents\PythonScripts\climate work\SDG article\SDG-analysis\SDG Partnerships_clean.pkl')
goals = np.arange(18).astype(str)
goals = pd.Series(goals).apply(lambda x: "goal "+x)
goal_dict = dict(zip(df["goal"].unique(),goals[1:]))
# df = df.replace('null',np.nan)
# df.to_csv(r'C:\Users\Andrew\Documents\PythonScripts\climate work\SDG Partnerships_clean.csv')
'''figuring out overlaps in goals'''
pivot = df.pivot_table(index='url',columns='goal',values='description',aggfunc='count')

goals = pivot.columns
unique_tuples = []
i = 0
while i < pivot.shape[0]:
    row = pivot.iloc[i,:]
    related_goals = []
    for idx,col in enumerate(row):
        if col==1:
            related_goals.append(idx+1)
    unique_tuples.append(related_goals)
    i+=1

unique_tuples = pd.Series(unique_tuples)
related_unique_tuples = unique_tuples.value_counts()
# related_unique_tuples = related_unique_tuples.reset_index()
related_pairs = [x for x in unique_tuples if len(x)>=2]
related_pairs = pd.Series(related_pairs).value_counts()
related_pairs[:30].plot(kind="bar", figsize=(8,4)).set(title="Goals Tackled Together by Partnership")

url_unique = pivot.sum(axis=1)
url_unique_idx = [idx for idx,i in enumerate(url_unique) if i==1]
url_unique = url_unique[url_unique_idx]

'''geographic coverage'''
geo = pd.DataFrame(index=np.arange(14249),data=df["beneficiary_countires"])
geo["beneficiary_countires"] = geo["beneficiary_countires"].apply(lambda x: re.findall('[A-Z][^A-Z]*', str(x)))
df["beneficiary_countires"] = geo["beneficiary_countires"]
all_bene = []

geo["beneficiary_countires"]= geo["beneficiary_countires"].apply(lambda x: [i.replace(' ','') for i in x]) #standardize country names

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

geo_df = pd.DataFrame(mlb.fit_transform(geo["beneficiary_countires"]),
                   columns=mlb.classes_,
                   index=geo.index)

count_geo_df = geo_df.sum().sort_values(ascending=False)
count_geo_df = count_geo_df.reset_index()
count_geo_df = count_geo_df.drop([0,3,14])
count_geo_df.set_index("index",inplace=True)
count_geo_df[:20].plot.barh(figsize=(8,4),legend=False).set_title("Top Beneficiaries of Partnerships")

base = df.pivot_table(index="based",values="url",aggfunc="count") #only 1000 show base
base = base.sort_values(by="url",ascending=False)
base.iloc[1:20,:].plot.barh(figsize=(8,4),legend=False).set_title("Top Headquarters of Partnerships")

#need to figure out how to get pairs from list and list?
# df["geographic_pairs"] =df["based"] + ', ' + df["beneficiary_countires"] 

# pairs = df["geographic_pairs"].value_counts()
# pairs = pairs.reset_index()
# pairs["index"] = pairs["index"].apply(lambda x: x.split(', '))
# pairs.set_index('index',inplace=True)
# pairs = [x for x in pairs.index if (x[0]!="") & (x[1]!="")]
# pairs_count = pd.Series(pairs).value_counts()

'''partners and entity analysis'''
seperated = df["partners_of_partner"].apply(lambda x: x.split(',')).apply(lambda x: [i.replace(' ','') for i in x]) #split then standardize 
#if for some reason you want to apply to a series, then use map. 

all_part = []
for lists in seperated:
    all_part.extend(lists)
unique_part = set(all_part)

mlb = MultiLabelBinarizer()

sep_df = pd.DataFrame(mlb.fit_transform(seperated),
                   columns=mlb.classes_,
                   index=seperated.index)

count_sep_df = sep_df.sum().sort_values(ascending=False)
count_sep_df = count_sep_df.reset_index()
count_sep_df["index"] = count_sep_df["index"].apply(lambda x: x[:50])
count_sep_df.set_index('index',inplace=True)
count_sep_df[:25].plot.barh(figsize=(8,10),legend=False).set_title("Top Sub-Partners of Partnerships")

df["entity"].value_counts()[1:20].plot.barh(figsize=(8,10),legend=False).set_title("Top Entities of Partnerships")

'''time series analysis'''
import plotly.express as px
from plotly.offline import plot

time_base = df.pivot_table(index='goal',columns='year_end',values='url',aggfunc="count")
time_base = time_base.fillna(0)

transpose_time = time_base.T.iloc[-30:-1,:]
pct_transpose_time = transpose_time.div(transpose_time.sum(axis=1),axis=0)

fig = px.bar(transpose_time)
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=3
))
plot(fig)

timeline = time_base.sum(axis=0)
timeline[:-1].plot(kind="line", figsize=(10,10))

a = df[df["length_of_partnership"]!='ignore']["length_of_partnership"]
a = a[a>0]
a.hist(figsize=(8,4),bins=30).set(title="Length of Partnerships")

