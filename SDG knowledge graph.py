# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 22:20:33 2020

@author: Andrew
"""
import pandas as pd
import numpy as np
import datetime

import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

'''Knowledge Graph'''
# #https://www.analyticsvidhya.com/blog/2019/10/how-to-build-knowledge-graph-text-using-spacy/
# import pandas as pd
# import bs4
# import requests
# import spacy
# from spacy import displacy
# import en_core_web_sm
# nlp = en_core_web_sm.load()

# from spacy.matcher import Matcher 
# from spacy.tokens import Span 

# import networkx as nx

# import matplotlib.pyplot as plt
# from tqdm import tqdm

# # doc = nlp("the drawdown process is governed by astm standard d823")

# # for tok in doc:
# #   print(tok.text, "...", tok.dep_)
  
# def get_entities(sent):
    
#   ## chunk 1
#   ent1 = ""
#   ent2 = ""

#   prv_tok_dep = ""    # dependency tag of previous token in the sentence
#   prv_tok_text = ""   # previous token in the sentence

#   prefix = ""
#   modifier = ""

#   #############################################################
  
#   for tok in nlp(sent):
#     ## chunk 2
#     # if token is a punctuation mark then move on to the next token
#     if tok.dep_ != "punct":
#       # check: token is a compound word or not
#       if tok.dep_ == "compound":
#         prefix = tok.text
#         # if the previous word was also a 'compound' then add the current word to it
#         if prv_tok_dep == "compound":
#           prefix = prv_tok_text + " "+ tok.text
      
#       # check: token is a modifier or not
#       if tok.dep_.endswith("mod") == True:
#         modifier = tok.text
#         # if the previous word was also a 'compound' then add the current word to it
#         if prv_tok_dep == "compound":
#           modifier = prv_tok_text + " "+ tok.text
      
#       ## chunk 3
#       if tok.dep_.find("subj") == True:
#         ent1 = modifier +" "+ prefix + " "+ tok.text
#         prefix = ""
#         modifier = ""
#         prv_tok_dep = ""
#         prv_tok_text = ""      

#       ## chunk 4
#       if tok.dep_.find("obj") == True:
#         ent2 = modifier +" "+ prefix +" "+ tok.text
        
#       ## chunk 5  
#       # update variables
#       prv_tok_dep = tok.dep_
#       prv_tok_text = tok.text
#   #############################################################

#   return [ent1.strip(), ent2.strip()]

# def get_relation(sent):

#   doc = nlp(sent)

#   # Matcher class object 
#   matcher = Matcher(nlp.vocab)

#   #define the pattern 
#   pattern = [{'DEP':'ROOT'}, 
#             {'DEP':'prep','OP':"?"},
#             {'DEP':'agent','OP':"?"},  
#             {'POS':'ADJ','OP':"?"}] 

#   matcher.add("matching_1", None, pattern) 

#   matches = matcher(doc)
#   k = len(matches) - 1

#   span = doc[matches[k][1]:matches[k][2]] 

#   return(span.text)

# entity_pairs = []

# #create list with all sentences
# tokens = df["description"].apply(lambda x: sent_tokenize(x))
# all_sent = []
# for token in tokens:
#     all_sent.extend(token)

# for i in tqdm(all_sent):
#   entity_pairs.append(get_entities(i))

# #figure out patterns before trying to get_relationships

# relations = [get_relation(i) for i in tqdm(all_sent)]

# # extract subject
# source = [i[0] for i in entity_pairs]

# # extract object
# target = [i[1] for i in entity_pairs]

# kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})

# G=nx.from_pandas_edgelist(kg_df, "source", "target", 
#                           edge_attr=True, create_using=nx.MultiDiGraph())

# plt.figure(figsize=(12,12))

# pos = nx.spring_layout(G)
# nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
# plt.show()

# G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="composed by"], "source", "target", 
#                           edge_attr=True, create_using=nx.MultiDiGraph())

# plt.figure(figsize=(12,12))
# pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes
# nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
# plt.show()

#make it interactive with plotly https://plotly.com/python/network-graphs/


'''network graph or sankey graph (numerical?)'''
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

# source is partner, target is goal, will need relations to be if certain words are contained? 

# relations = ???

# source = df["goal"]

# target = df["partner_name"]

# kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})

# G=nx.from_pandas_edgelist(kg_df, "source", "target", 
#                           edge_attr=True, create_using=nx.MultiDiGraph())

# plt.figure(figsize=(12,12))

# pos = nx.spring_layout(G)
# nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
# plt.show()

###### https://plotly.com/python/network-graphs/
# import plotly.graph_objects as go

# edge_x = []
# edge_y = []
# for edge in G.edges():
#     x0, y0 = G.nodes[edge[0]]['pos']
#     x1, y1 = G.nodes[edge[1]]['pos']
#     edge_x.append(x0)
#     edge_x.append(x1)
#     edge_x.append(None)
#     edge_y.append(y0)
#     edge_y.append(y1)
#     edge_y.append(None)

# edge_trace = go.Scatter(
#     x=edge_x, y=edge_y,
#     line=dict(width=0.5, color='#888'),
#     hoverinfo='none',
#     mode='lines')

# node_x = []
# node_y = []
# for node in G.nodes():
#     x, y = G.nodes[node]['pos']
#     node_x.append(x)
#     node_y.append(y)

# node_trace = go.Scatter(
#     x=node_x, y=node_y,
#     mode='markers',
#     hoverinfo='text',
#     marker=dict(
#         showscale=True,
#         # colorscale options
#         #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
#         #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
#         #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
#         colorscale='YlGnBu',
#         reversescale=True,
#         color=[],
#         size=10,
#         colorbar=dict(
#             thickness=15,
#             title='Node Connections',
#             xanchor='left',
#             titleside='right'
#         ),
#         line_width=2))

# node_adjacencies = []
# node_text = []
# for node, adjacencies in enumerate(G.adjacency()):
#     node_adjacencies.append(len(adjacencies[1]))
#     node_text.append('# of connections: '+str(len(adjacencies[1])))

# node_trace.marker.color = node_adjacencies
# node_trace.text = node_text

# fig = go.Figure(data=[edge_trace, node_trace],
#              layout=go.Layout(
#                 title='<br>Network graph made with Python',
#                 titlefont_size=16,
#                 showlegend=False,
#                 hovermode='closest',
#                 margin=dict(b=20,l=5,r=5,t=40),
#                 annotations=[ dict(
#                     text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
#                     showarrow=False,
#                     xref="paper", yref="paper",
#                     x=0.005, y=-0.002 ) ],
#                 xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                 yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
#                 )
# fig.show()