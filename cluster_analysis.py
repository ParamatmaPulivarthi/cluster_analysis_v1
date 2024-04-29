# ************************** importing library's *******************************************
import networkx as nx
import re
import nltk
import warnings
import numpy as np
from nltk import word_tokenize
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralClustering
import dash
import dash_cytoscape as cyto
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input
import pandas as pd

df = pd.read_json(r'D:\Prx_Project\ss_clsteranalysis\V10_All_From_MongoDb_v1.json',lines=True)
df1=df[:1000]
# df1=df
# ************************Step1. pre processing data*******************************
# ----------collecting authors data from data_____________________________
aut_data = []
for author in df1.values:
    #     print(author[1])
    for aut in author[2]:
        #         print(aut)
        aut_data.append([aut, author[1]])

# ----------collecting book title data------------------------
title_data=[]
for author in df1.values:
    title_data.append([author[4],author[1]])

# -----creation authors dataframe & title dataframe
df_clu=pd.DataFrame(aut_data,columns=['author','abstract'])
df_title=pd.DataFrame(title_data,columns=['title','abstract'])

# ---removing the duplicate authors details
df_clu.drop_duplicates(inplace=True)
df_title.drop_duplicates(inplace=True)

# -----------sorting data by author and title wise
df_clu.sort_values("author",ascending=True)
df_title.sort_values("title",ascending=True)

# ----------------taking unquie author and title values for drop down
author=df_clu.author.unique()
title=df_title.title.unique()

# ************************************* Step2. cluster Analysis***********************************
# ***** data preparation for clusters Analysis ***

# ----tokenizer------
# to remove unwanted words, and stop words
def simple_tokenizer(text):
    nltk.download('stopwords')
    stopWords = stopwords.words('english')
    charfilter = re.compile('[a-zA-Z]+')
    nltk.download('punkt')
    # tokenizing the words:
    words = word_tokenize(text)
    # converting all the tokens to lower case:
    words = map(lambda word: word.lower(), words)
    # let's remove every stopwords
    words = [word for word in words if word not in stopWords]
    # stemming all the tokens
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
    ntokens = list(filter(lambda token: charfilter.match(token), tokens))
    return ntokens

#---------- cluster analysis for each Author------------
def author_cluster(author_name):
    df_aut_one=df_clu[df_clu.author==author_name]
    df_aut_one.dropna(inplace=True)
    abstr=df_aut_one.abstract.tolist()
    tf_idf_vector = TfidfVectorizer(tokenizer = simple_tokenizer, max_features = 1000, norm = 'l2')
    from sklearn.model_selection import train_test_split

    transformed_vector = tf_idf_vector.fit_transform(abstr)
    clustering = SpectralClustering(n_clusters=5, assign_labels="discretize", random_state=0)
    prediction = clustering.fit_predict(transformed_vector)
    labels = clustering.labels_
    pred_table=pd.DataFrame(abstr)
    pred_table['labels']=labels
    num=np.arange(1,len(labels)+1)

    s = pd.Series(num)
    pred_table['index']=s.astype('int')
    serieslst=pred_table.values.tolist()
    newseries = []
    for i in serieslst:
        newseries.append([i[1], i[2]])
    lables = pred_table.labels.tolist()
    labels_new = list(set(lables))
    # return [labels_new,newseries]
    G=nx.Graph()
    G.add_nodes_from(labels_new,width=6)
    G.add_edges_from(newseries,width=1)
    nx.draw(G,with_labels=True,node_color='g')
    plt.show()

#---------- cluster analysis for each Book Title------------
def title_cluster(title_name):
    # ---collecting given title aritical
    df_title_one=df_title[df_title.title==title_name]
    df_title_one.dropna(inplace=True)

    # ---------------collecting abstract details for analysis
    abstr=df_title_one.abstract.tolist()
    print("-------------abstr",abstr)
    if len(abstr)<=1:
        abstr=abstr+abstr

    # -----converting abstract text to numberic for semantic analysis
    tf_idf_vector = TfidfVectorizer(tokenizer = simple_tokenizer, max_features = 1000, norm = 'l2')

    # -----collecting stop word and removing from the data text
    nltk.download('stopwords')
    stopWords = stopwords.words('english')
    charfilter = re.compile('[a-zA-Z]+')

    nltk.download('punkt')
    transformed_vector = tf_idf_vector.fit_transform(abstr)


    # -------applying the spectralcluster analysis, default cluster as 5
    clustering = SpectralClustering(n_clusters=5, assign_labels="discretize", random_state=0)

    # generating predictions for the clusters
    prediction = clustering.fit_predict(transformed_vector)

    # creating a data frame with cluster labels and predictions
    labels = clustering.labels_
    pred_table=pd.DataFrame(abstr)
    pred_table['labels']=labels
    num=np.arange(1,len(labels)+1)

    s = pd.Series(num)
    pred_table['index']=s.astype('int')
    serieslst=pred_table.values.tolist()
    newseries = []
    for i in serieslst:
        newseries.append([i[1], i[2]])

    # generating the network graph for title
    # return newseries
    G=nx.Graph()
    G.add_edges_from(newseries)
    nx.draw(G)
    plt.show()


#---------- cluster analysis for Author and Book Title------------
def author_title_cluster(author_name,title_name):
    # ---collecting given title aritical
    df_title_one=df_title[df_title.title==title_name]
    df_title_one.dropna(inplace=True)
    df_clu_one = df_clu[df_clu.author == author_name]
    df_clu_one.dropna(inplace=True)
    df_aut_title=pd.concat([df_title_one,df_clu_one],axis=0)
    print(df_aut_title.head())


    print("-----df_title_one", df_title_one.shape)
    print("-----df_clu_one", df_clu_one.shape)
    print("-----df_aut_title", df_aut_title.shape)

    # ---------------collecting abstract details for analysis
    abstr=df_aut_title.abstract.tolist()
    print("-------------abstr",abstr)
    if len(abstr)<=1:
        abstr=abstr+abstr

    # -----converting abstract text to numberic for semantic analysis
    tf_idf_vector = TfidfVectorizer(tokenizer = simple_tokenizer, max_features = 1000, norm = 'l2')

    # -----collecting stop word and removing from the data text
    nltk.download('stopwords')
    stopWords = stopwords.words('english')
    charfilter = re.compile('[a-zA-Z]+')

    nltk.download('punkt')
    transformed_vector = tf_idf_vector.fit_transform(abstr)


    # -------applying the spectralcluster analysis, default cluster as 5
    clustering = SpectralClustering(n_clusters=5, assign_labels="discretize", random_state=0)

    # generating predictions for the clusters
    prediction = clustering.fit_predict(transformed_vector)

    # creating a data frame with cluster labels and predictions
    labels = clustering.labels_
    pred_table=pd.DataFrame(abstr)
    pred_table['labels']=labels
    num=np.arange(1,len(labels)+1)

    s = pd.Series(num)
    pred_table['index']=s.astype('int')
    serieslst=pred_table.values.tolist()
    newseries = []
    for i in serieslst:
        newseries.append([i[1], i[2]])
    G=nx.Graph()
    G.add_edges_from(newseries)
    nx.draw(G)
    plt.show()


def cluster_analysis():
    # ---collecting given title aritical
    df_title_one = df_title
    df_title_one.dropna(inplace=True)
    df_clu_one = df_clu
    df_clu_one.dropna(inplace=True)
    df_aut_title = pd.concat([df_title_one, df_clu_one], axis=0)
    # ---------------collecting abstract details for analysis
    abstr = df_aut_title.abstract.tolist()

    if len(abstr) <= 1:
        abstr = abstr + abstr

    # -----converting abstract text to numberic for semantic analysis
    tf_idf_vector = TfidfVectorizer(tokenizer=simple_tokenizer, max_features=1000, norm='l2')

    # -----collecting stop word and removing from the data text
    nltk.download('stopwords')
    stopWords = stopwords.words('english')
    charfilter = re.compile('[a-zA-Z]+')

    nltk.download('punkt')
    transformed_vector = tf_idf_vector.fit_transform(abstr)

    # -------applying the spectralcluster analysis, default cluster as 5
    clustering = SpectralClustering(n_clusters=5, assign_labels="discretize", random_state=0)

    # generating predictions for the clusters
    prediction = clustering.fit_predict(transformed_vector)

    # creating a data frame with cluster labels and predictions
    labels = clustering.labels_
    pred_table = pd.DataFrame(abstr)
    pred_table['labels'] = labels
    num = np.arange(1, len(labels) + 1)

    s = pd.Series(num)
    pred_table['index'] = s.astype('int')
    serieslst = pred_table.values.tolist()
    newseries = []
    for i in serieslst:
        newseries.append([i[1], i[2]])
    return newseries

clu_out=cluster_analysis()

# ------node and edges for graph
nodelist=[]
for i in clu_out:
    nodelist.append(i[0])
    nodelist.append(i[1])
nodelist_v1 = list(dict.fromkeys(nodelist))
node_l = [dict(dict(zip(['id'],[x]))) for x in nodelist_v1]
node_j=[dict(zip(['data'],[x])) for x in node_l]
edge_l= [dict(zip(['source','target'],[x[0],x[1]])) for x in clu_out]
edge_j=[dict(zip(['data'],[x])) for x in edge_l]
node_edge=node_j+edge_j
# *****************************************Step3: Dash Report / Application ****************************************************************

# --------------App layout--> Author, Title Selection
app = dash.Dash(__name__, )
app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.H3('My cluster Analysis', style = {"margin-bottom": "0px", 'color': 'white'}),
            ]),
        ], className = "six column", id = "title")

    ], id = "header", className = "row flex-display", style = {"margin-bottom": "25px"}),
    html.Div([
        html.Div([
            html.P('Select Author:', className = 'fix_label', style = {'color': 'white'}),
            dcc.Dropdown(id = 'w_authors',
                         multi = False,
                         clearable = True,
                         disabled = False,
                         style = {'display': True},
                         value = 'Makoto Satoh',
                         placeholder = 'Select Authors',
                         options = [{'label': c, 'value': c}
                                    for c in author], className = 'dcc_compon'),

            html.P('Select Title:', className = 'fix_label', style = {'color': 'white'}),
            dcc.Dropdown(id = 'w_title',
                         multi = False,
                         clearable = True,
                         disabled = False,
                         style = {'display': True},
                         placeholder = 'Select Titles',
                         options = [{'label': c, 'value': c}
                                    for c in title], className = 'dcc_compon')

        ], className = "create_container three columns"),
        html.Div([
            cyto.Cytoscape(
                id='org-chart',
                layout={'name': 'random'},
                style={'width': '100%', 'height': '500px'},
                elements=node_edge
            )
        ], className='six columns')
        ], className = "row flex-display"),
],
    id = "mainContainer", style = {"display": "flex", "flex-direction": "column"})


@app.callback(
    Output('w_authors', 'value'),
    Input('w_authors', 'options'))

def get_author_options(w_authors):
    return [{'label': i, 'value': i} for i in df_clu['author']]

@app.callback(
    Output('w_title', 'value'),
    Input('w_title', 'options'))
def get_title_value(w_title):
    return [{'label': i, 'value': i} for i in df_title['title']]

@app.callback(Output('org-chart', 'figure'),
              [Input('w_authors', 'value')],
              [Input('w_title', 'value')])
def update_graph(w_authors, w_title):
    if w_authors != '' and w_title != '':
        author_title_cluster(w_authors, w_title)
    elif w_authors != '':
        author_cluster(w_authors)
    elif w_title != '':
        title_cluster(w_title)
    else:
        print("Nothing to show")


# ---------calling dash report
if __name__ == '__main__':
    app.run_server(debug = False)