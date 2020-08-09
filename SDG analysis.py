# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 12:04:58 2020

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
count_geo_df = count_geo_df.drop([2,3,15])
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
a = a.reset_index()
a.hist(figsize=(8,4),bins=30).set(title="Length of Partnerships")

'''NLP PCA'''
import string

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer #acts like a model pretty much

corpus = df[df["description"]!=""] #not empty

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens
# initialize count vectorizer object
vect = CountVectorizer(tokenizer=tokenize)
X = vect.fit_transform(corpus["description"]) # [] needs to be wrapped around the string to make it a "document" if you select single element
vect.vocabulary_

#keywords
key_df = pd.DataFrame(data=X.toarray())
key_df.columns = vect.vocabulary_
key = key_df.sum()

from sklearn.feature_extraction.text import TfidfTransformer

# initialize tf-idf transformer object
transformer = TfidfTransformer(smooth_idf=False)
tfidf_X = transformer.fit_transform(X)

#plotting
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
clust_df = pd.DataFrame(data=tfidf_X.toarray()) #try PCA on both tfidf and bag of words
clust_df.columns = vect.vocabulary_

pca = PCA(n_components=20)
principalComponents = pca.fit_transform(clust_df) #replace this with cosine matrix instead when you run that

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
plt.show()

from gensim.summarization import summarize
# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)
PCA_components["goal"] = corpus["goal"]
PCA_components["url"] = corpus["url"]

PCA_components["Year"] = corpus["year_start"]

symbols = [0, 'circle', 100, 'circle-open', 200, 'circle-dot', 300,
            'circle-open-dot', 1, 'square', 101, 'square-open', 201,
            'square-dot', 301, 'square-open-dot', 2, 'diamond', 102,
            'diamond-open', 202, 'diamond-dot', 302,
            'diamond-open-dot', 3, 'cross', 103, 'cross-open', 203,
            'cross-dot', 303, 'cross-open-dot', 4, 'x', 104, 'x-open',
            204, 'x-dot', 304, 'x-open-dot', 5, 'triangle-up', 105,
            'triangle-up-open', 205, 'triangle-up-dot', 305,
            'triangle-up-open-dot', 6, 'triangle-down', 106,
            'triangle-down-open', 206, 'triangle-down-dot', 306,
            'triangle-down-open-dot', 7, 'triangle-left', 107,
            'triangle-left-open', 207, 'triangle-left-dot', 307,
            'triangle-left-open-dot', 8, 'triangle-right', 108,
            'triangle-right-open', 208, 'triangle-right-dot', 308,
            'triangle-right-open-dot', 9, 'triangle-ne', 109,
            'triangle-ne-open', 209, 'triangle-ne-dot', 309,
            'triangle-ne-open-dot', 10, 'triangle-se', 110,
            'triangle-se-open', 210, 'triangle-se-dot', 310,
            'triangle-se-open-dot', 11, 'triangle-sw', 111,
            'triangle-sw-open', 211, 'triangle-sw-dot', 311,
            'triangle-sw-open-dot', 12, 'triangle-nw', 112,
            'triangle-nw-open', 212, 'triangle-nw-dot', 312,
            'triangle-nw-open-dot', 13, 'pentagon', 113,
            'pentagon-open', 213, 'pentagon-dot', 313,
            'pentagon-open-dot', 14, 'hexagon', 114, 'hexagon-open',
            214, 'hexagon-dot', 314, 'hexagon-open-dot', 15,
            'hexagon2', 115, 'hexagon2-open', 215, 'hexagon2-dot',
            315, 'hexagon2-open-dot', 16, 'octagon', 116,
            'octagon-open', 216, 'octagon-dot', 316,
            'octagon-open-dot', 17, 'star', 117, 'star-open', 217,
            'star-dot', 317, 'star-open-dot', 18, 'hexagram', 118,
            'hexagram-open', 218, 'hexagram-dot', 318,
            'hexagram-open-dot', 19, 'star-triangle-up', 119]

plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='blue')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

colortype = "goal"

import plotly.express as px
from plotly.offline import plot
import random
PCA_filtered = PCA_components[PCA_components['url'].isin(url_unique.index)] #can't filter until the rest has already been run once
fig = px.scatter(PCA_filtered, x=0, y=1, color=colortype, hover_data=['goal',PCA_filtered.index],opacity=0.7)
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=3
))
i=0
chosen_symbols=[]
while i < len(corpus["goal"].unique()):
    fig['data'][i]['marker']['symbol'] = random.choice(symbols[:30]) #chosen_symbols[i]
    i+=1 #comment out if symbols haven't been chosen 
    if fig['data'][i]['marker']['symbol'] in chosen_symbols:
        print('getting new symbol')
    else:
        chosen_symbols.append(fig['data'][i]['marker']['symbol'])        
        i+=1

plot(fig, filename="pca_.html")

# PCA_components_dash = PCA_components[PCA_components["Year"]!='ignore']
# PCA_components_dash = PCA_components_dash[PCA_components_dash["Year"]!= np.nan]

# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output, State
# import dash_bootstrap_components as dbc

# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# server = app.server

# app.layout = dbc.Card(
#                 dbc.CardBody([
#                         dbc.Row([
#                              dbc.Col(
#                               dcc.Graph(id = "plot",figure=fig, style={'height': '900px'})
#                               ,width=12),
                              
#                               ]),
                        
#                         dbc.Row([
#                             dbc.Col(
#                               dcc.Slider(
#                                     id='year-selector',
#                                     min=PCA_components_dash['Year'].min(),
#                                     max=PCA_components_dash['Year'].max(),
#                                     value=PCA_components_dash['Year'].min(),
#                                     marks={str(year): str(year) for year in PCA_components_dash['Year'].unique()},
#                                     step=None
#                                 )
#                               ,width=12),
#                             ]),
#                             ]),
#                     className="mt-3",
#                     ) #card end

# @app.callback(
#     Output('plot', 'figure'),
#     [Input('year-selector', 'value')])
# def set_counties_options(year):
#     PCA_selected = PCA_components_dash[PCA_components_dash["Year"]<=year]
#     fig = px.scatter(PCA_selected, x=0, y=1, color=colortype, hover_data=['goal',PCA_selected.index],opacity=0.3)
#     fig.update_layout(legend=dict(
#         yanchor="top",
#         y=0.99,
#         xanchor="right",
#         x=3
#     ))
#     i=0
#     while i < len(PCA_selected["goal"].unique()):
#         fig['data'][i]['marker']['symbol'] = chosen_symbols[i] 
#         i+=1
#     return fig 

# if __name__ == '__main__':
#     app.run_server(debug=False)
    
import time
from sklearn.manifold import TSNE

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=300)
tsne_results = tsne.fit_transform(PCA_components.iloc[:,:2])

PCA_components["tsne-one"]=tsne_results[:,0]
PCA_components["tsne-two"]=tsne_results[:,1]

PCA_components.dropna(inplace=True)
import plotly.express as px
from plotly.offline import plot
fig = px.scatter(PCA_components, x="tsne-one", y="tsne-two", color=colortype, hover_data=['goal',PCA_components.index],
                  opacity=0.3)
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=3
))

i=0
#use same chosen_symbols as above
while i < len(corpus["goal"].unique()):
    fig['data'][i]['marker']['symbol'] = random.choice(symbols[:30])
    i+=1

plot(fig, filename="tsne.html")

#create a function that finds top keywords in a certain cluster? take in x and y bounds then sort by sum(axis=0)? 
def area_search_function(x_min,x_max,y_min,y_max,df):
    df_t = key_df[(df[0] > x_min) &
            (df[0] < x_max) &
            (df[1] > y_min) &
            (df[1] < y_max)]
    count_keywords = df_t.sum(axis=0).sort_values(ascending=False)
    return count_keywords

#maybe we can get summaries instead?

keywords_left = area_search_function(-0.2,0,-0.15,0,PCA_components)
keywords_left = keywords_left.reset_index()

keywords_right = area_search_function(0,0.3,-0.1,0.1,PCA_components)
keywords_right = keywords_right.reset_index()

'''similarity matrix'''
# #cosine similarity search function https://www.machinelearningplus.com/nlp/cosine-similarity/ #not enough memory 
# from sklearn.metrics.pairwise import cosine_similarity
# clust_df["goal"] = corpus["goal"]
# clust_df["url"] = corpus["url"]
# clust_df.set_index(["goal","url"])
# cosine_search = cosine_similarity(clust_df, clust_df)

import gensim
from gensim.matutils import softcossim 
from gensim import corpora
import gensim.downloader as api
from gensim.utils import simple_preprocess

fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
# Prepare a dictionary and a corpus.
dictionary = corpora.Dictionary([simple_preprocess(doc) for doc in df["description"][330:340]])

# Prepare the similarity matrix
similarity_matrix = fasttext_model300.similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0, nonzero_limit=100)

# Convert the sentences into bag-of-words vectors.
sentences = []
for doc in df["description"][330:340]:
    doc_x = dictionary.doc2bow(simple_preprocess(doc))
    sentences.append(doc_x)

def create_soft_cossim_matrix(sentences):
    len_array = np.arange(len(sentences))
    xx, yy = np.meshgrid(len_array, len_array)
    cossim_mat = pd.DataFrame([[round(softcossim(sentences[i],sentences[j], similarity_matrix) ,2) for i, j in zip(x,y)] for y, x in zip(xx, yy)])
    return cossim_mat

cosine = create_soft_cossim_matrix(sentences)

cosine.to_csv(r'C:\Users\Andrew\Documents\PythonScripts\climate work\SDG article\SDG-Analysis\cosine.csv')

def search_cosine(idx):
    row = cosine.iloc[idx,:]
    row = row.sort_values(ascending=False)
    return row[:20]

first_search = search_cosine(10)

'''resource search'''
####set resource conditions
# technology = ['tech', 'IT', 'digital', 'data']
# training = ['teach','train', 'skill', 'courses', 'class']
# food = ['seeds', 'agriculture', 'irrigation']
# fiscal = ['subsidy', 'fund', 'trade', 'finance', 'investment']
# research = ['research', 'framework', 'assess', 'monitor']
# conservation = ['conservation', 'civil engineer', 'biodiversity']
# legal_politcal = ['democratic', 'rights', 'policy', 'law']

# all_resources = [technology, training, food, fiscal, research, conservation, legal_politcal]
# all_resources_words = ['technology','training', 'food', 'fiscal', 'research', 'conservation', 'legal_politcal','other']
# #get one-hot dataframe
# resource_tracker = []
# for idx,desc in enumerate(df["description"]):
#     print(idx)
#     current_desc = []
#     for idx,resource in enumerate(all_resources):
#         for keyword in resource:
#             if keyword in desc:
#                 current_desc.append(all_resources_words[idx])
#     if len(current_desc)==0:
#         current_desc.append(all_resources_words[-1])
#     resource_tracker.append(current_desc)

# #plot bar
# pd.Series(resource_tracker).value_counts()[:30].plot.barh(figsize=(8,8)).set(title="Top Resources by Partnership")

# mlb = MultiLabelBinarizer()

# resource_df = pd.DataFrame(mlb.fit_transform(resource_tracker),
#                    columns=mlb.classes_,
#                    index=geo.index)

# count_resource_df = pd.concat([df,resource_df],axis=1)
# pivot = count_resource_df.pivot_table(index="goal",values = all_resources_words,aggfunc='sum')
# pivot.reset_index(inplace=True)
# pivot["goal"] = pivot["goal"].apply(lambda x: goal_dict[x])
# pivot.set_index("goal",inplace=True)
# pivot = pivot[all_resources_words]
# fig = px.bar(pivot,orientation='h')
# # plot(fig, filename="resources_stacked.html")

# pivot_pct = pivot.div(pivot.sum(axis=1),axis=0)
# fig = px.bar(pivot_pct,orientation='h')
# # plot(fig, filename="resources_pct.html")


#px.bar(pivot where goals are rows, resources are the columns)