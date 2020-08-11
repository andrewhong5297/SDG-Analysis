# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:49:16 2020

@author: Andrew
"""
import pandas as pd
import numpy as np
import re

df = pd.read_pickle(r'C:\Users\Andrew\Documents\PythonScripts\climate work\SDG article\SDG-analysis\SDG Partnerships_clean.pkl')
goals = np.arange(18).astype(str)
goals = pd.Series(goals).apply(lambda x: "goal "+x)
goal_dict = dict(zip(df["goal"].unique(),goals[1:]))

'''resource search'''
###set resource conditions
technology = ['tech', 'IT', 'digital', 'data']
training = ['teach','train', 'skill', 'courses', 'class']
food = ['seeds', 'agriculture', 'irrigation']
fiscal = ['subsidy', 'fund', 'trade', 'finance', 'investment']
research = ['research', 'framework', 'assess', 'monitor']
conservation = ['conservation', 'civil engineer', 'biodiversity']
legal_political = ['democratic', 'rights', 'policy', 'law']

all_resources = [technology, training, food, fiscal, research, conservation, legal_political]
all_resources_words = ['technology','training', 'food', 'fiscal', 'research', 'conservation', 'legal_political','other']
#get one-hot dataframe
resource_tracker = []
for idx,desc in enumerate(df["description"]):
    print(idx)
    current_desc = []
    for idx,resource in enumerate(all_resources):
        for keyword in resource:
            if keyword in desc:
                current_desc.append(all_resources_words[idx])
    if len(current_desc)==0:
        current_desc.append(all_resources_words[-1])
    resource_tracker.append(current_desc)

#plot bar
pd.Series(resource_tracker).value_counts()[:30].plot.barh(figsize=(8,8)).set(title="Top Resources by Partnership")

mlb = MultiLabelBinarizer()

resource_df = pd.DataFrame(mlb.fit_transform(resource_tracker),
                    columns=mlb.classes_,
                    index=geo.index)

count_resource_df = pd.concat([df,resource_df],axis=1)
pivot = count_resource_df.pivot_table(index="goal",values = all_resources_words,aggfunc='sum')
pivot.reset_index(inplace=True)
pivot["goal"] = pivot["goal"].apply(lambda x: goal_dict[x])
pivot.set_index("goal",inplace=True)
pivot = pivot[all_resources_words]
fig = px.bar(pivot,orientation='h')
# plot(fig, filename="resources_stacked.html")

pivot_pct = pivot.div(pivot.sum(axis=1),axis=0)
fig = px.bar(pivot_pct,orientation='h')
# plot(fig, filename="resources_pct.html")

