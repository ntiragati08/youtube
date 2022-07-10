#!pip install psycopg2
#from msilib.schema import Component
from streamlit.components.v1 import html
import streamlit.components.v1 as components
import streamlit as st
import psycopg2 as ps
import pandas as pd
import numpy as np
from dateutil import parser
#!pip install isodate
import isodate
# Data visualization libraries
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.model_selection import train_test_split
sns.set(style="darkgrid", color_codes=True)

st.title('Social Media Metrics Analysis and Predictions')
html_string = "<div class='tableauPlaceholder' id='viz1657405928821' style='position: relative'><noscript><a href='#'><img alt='Dashboard 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Yo&#47;YouTubeEDA_16568234121340&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='YouTubeEDA_16568234121340&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Yo&#47;YouTubeEDA_16568234121340&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1657405928821');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='1227px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"

#st.markdown(html_string, unsafe_allow_html=True)
components.html("""
</center><div class='tableauPlaceholder' id='viz1657405928821' style='position: relative'>
<noscript><a href='#'><img alt='Dashboard 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Yo&#47;YouTubeEDA_16568234121340&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a>
</noscript>
<object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='YouTubeEDA_16568234121340&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Yo&#47;YouTubeEDA_16568234121340&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' />
</object></div>
<script type='text/javascript'>
var divElement = document.getElementById('viz1657405928821');                    
var vizElement = divElement.getElementsByTagName('object')[0];                    
if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} 
else if ( divElement.offsetWidth > 500 ) 
{ vizElement.style.width='1000px';vizElement.style.height='827px';} 
else { vizElement.style.width='100%';vizElement.style.height='1227px';}                     
var scriptElement = document.createElement('script');                    
scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    
vizElement.parentNode.insertBefore(scriptElement, vizElement);                
</script></center>
    """, height = 1000, width = 1000, )

def connect_to_db(host_name, dbname, port, username, password):
    try:
        conn = ps.connect(host=host_name, database=dbname, user=username, password=password, port=port)

    except ps.OperationalError as e:
        raise e
    else:
        print('Connected!')
        return conn

def create_table(curr):
    create_table_command = ("""CREATE TABLE IF NOT EXISTS YTanalysis (
                    video_id TEXT PRIMARY KEY,
                    channelTitle TEXT,
                    title TEXT,
                    description TEXT,
                    tags TEXT,
                    publishedAt date,
                    viewCount NUMERIC NOT NULL,
                    likeCount NUMERIC NOT NULL,
                    favouriteCount NUMERIC,
                    commentCount NUMERIC,
                    duration TEXT,
                    definition TEXT,
                    caption BOOLEAN,
                    pushblishDayName TEXT,
                    durationSecs FLOAT NOT NULL,
                    tagsCount NUMERIC,
                    likeRatio FLOAT,
                    commentRatio FLOAT,
                    titleLength NUMERIC NOT NULL,
                    title_no_stopwords TEXT
 )""")

    curr.execute(create_table_command)

def insert_into_table(curr, video_id,channelTitle,
                    title,
                    description,
                    tags,
                    publishedAt,
                    viewCount,
                    likeCount,
                    favouriteCount,
                    commentCount,
                    duration,
                    definition,
                    caption,
                    pushblishDayName,
                    durationSecs,
                    tagsCount,
                    likeRatio,
                    commentRatio,
                    titleLength,
                    title_no_stopwords):
    insert_into_videos = ("""INSERT INTO YTanalysis (video_id,channelTitle,
                    title,
                    description,
                    tags,
                    publishedAt,
                    viewCount,
                    likeCount,
                    favouriteCount,
                    commentCount,
                    duration,
                    definition,
                    caption,
                    pushblishDayName,
                    durationSecs,
                    tagsCount,
                    likeRatio,
                    commentRatio,
                    titleLength,
                    title_no_stopwords)
    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);""")
    row_to_insert = (video_id,
                    channelTitle,
                    title,
                    description,
                    tags,
                    publishedAt,
                    viewCount,
                    likeCount,
                    favouriteCount,
                    commentCount,
                    duration,
                    definition,
                    caption,
                    pushblishDayName,
                    durationSecs,
                    tagsCount,
                    likeRatio,
                    commentRatio,
                    titleLength,
                    title_no_stopwords)
    curr.execute(insert_into_videos, row_to_insert)


def update_row(curr, video_id,channelTitle,
                    title,
                    description,
                    tags,
                    publishedAt,
                    viewCount,
                    likeCount,
                    favouriteCount,
                    commentCount,
                    duration,
                    definition,
                    caption,
               pushblishDayName,
                    durationSecs,
                    tagsCount,
                    likeRatio,
                    commentRatio,
                    titleLength,
                    title_no_stopwords):
    query = ("""UPDATE YTanalysis
            SET channelTitle = %s,
                title = %s,
                description = %s,
                tags = %s,
                publishedAt = %s,
                viewCount = %s,
                likeCount = %s,
                favouriteCount = %s,
                commentCount = %s,
                duration = %s,
                definition = %s,
                caption = %s,
                pushblishDayName - %s,
                durationSecs = %s,
                tagsCount = %s,
                likeRatio = %s,
                commentRatio = %s,
                titleLength = %s,
                title_no_stopwords = %s
            WHERE video_id = %s;""")
    vars_to_update = (video_id,channelTitle,
                    title,
                    description,
                    tags,
                    publishedAt,
                    viewCount,
                    likeCount,
                    favouriteCount,
                    commentCount,
                    duration,
                    definition,
                    caption,
                    pushblishDayName,
                    durationSecs,
                    tagsCount,
                    likeRatio,
                    commentRatio,
                    titleLength,
                    title_no_stopwords)
    curr.execute(query, vars_to_update)


def check_if_video_exists(curr, video_id): 
    query = ("""SELECT video_id FROM YTanalysis WHERE video_id = %s""")

    curr.execute(query, (video_id,))
    return curr.fetchone() is not None



def truncate_table(curr):
    truncate_table = ("""TRUNCATE TABLE YTanalysis""")

    curr.execute(truncate_table)


def append_from_df_to_db(curr,df):
    for i, row in df.iterrows():
        insert_into_table(curr, row['video_id'], row['channelTitle'], row['title'],
                    row['description'],
                    row['tags'],
                    row['publishedAt'],
                    row['viewCount'],
                    row['likeCount'],
                    row['favouriteCount'],
                    row['commentCount'],
                    row['duration'],
                    row['definition'],
                    row['caption'],
                    row['pushblishDayName'],
                    row['durationSecs'],
                    row['tagsCount'],
                    row['likeRatio'],
                    row['commentRatio'],
                    row['titleLength'],
                    row['title_no_stopwords'])


def update_db(curr,df):
    tmp_df = pd.DataFrame(columns=['video_id','channelTitle',
                    'title',
                    'description',
                    'tags',
                    'publishedAt',
                    'viewCount',
                    'likeCount',
                    'favouriteCount',
                    'commentCount',
                    'duration',
                    'definition',
                    'caption',
                    'pushblishDayName',
                    'durationSecs',
                    'tagsCount',
                    'likeRatio',
                    'commentRatio',
                    'titleLength',
                    'title_no_stopwords'])
    for i, row in df.iterrows():
        if check_if_video_exists(curr, row['video_id']): # If video already exists then we will update
            update_row(curr,row['video_id'], row['channelTitle'], row['title'],
                    row['description'],
                    row['tags'],
                    row['publishedAt'],
                    row['viewCount'],
                    row['likeCount'],
                    row['favouriteCount'],
                    row['commentCount'],
                    row['duration'],
                    row['definition'],
          row['caption'],
          row['pushblishDayName'],
                    row['durationSecs'],
                    row['tagsCount'],
                    row['likeRatio'],
                    row['commentRatio'],
                    row['titleLength'],
                    row['title_no_stopwords'])
        else: # The video doesn't exists so we will add it to a temp df and append it using append_from_df_to_db
            tmp_df = tmp_df.append(row)

    return tmp_df

host_name = 'my-db-instance.c331yxmg3hso.us-east-2.rds.amazonaws.com'
dbname = 'datascience'
port = '5432'
username = 'datascience'
password = 'NavSunny'
conn = None

conn = connect_to_db(host_name, dbname, port, username, password)
curr = conn.cursor()

query = "SELECT * FROM YTanalysis;"
df = pd.read_sql(query, conn)
#df.shape

query = "SELECT * FROM YTanalysis WHERE commentcount < 17000 AND likecount < 120000;"
df = pd.read_sql(query, conn)
#df.shape

X_count = df[["likecount", "commentcount"]]
Y_count = df[["viewcount"]]

data = df[["likecount", "commentcount", "viewcount"]]

from sklearn.preprocessing import StandardScaler
sclar = StandardScaler()

X_count = sclar.fit(data)

ML_array = (sclar.transform(data))
X_count = ML_array[:, [0,1]]
Y_count = ML_array[:, [2]]


# Splitting our data using train_test_split() in a 80:20 ratio
X_train,X_test,y_train,y_test = train_test_split(X_count,Y_count, test_size=.2, random_state=42)

#X_train.shape

#y_train.shape


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

y_predLinear = model.predict(X_test)

#print(type(X_test))

#y_predLinear[0]


# importing r2_score module
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# predicting the accuracy score
score=r2_score(y_test,y_predLinear)

print("r2 score is ",score)
st.title("Linear Regression")
st.write("Linear Regression r2 score is ", score)
st.write("Linear Regression mean_sqrd_error is==",mean_squared_error(y_test,y_predLinear))
st.write("Linear Regression root_mean_squared error of is==",np.sqrt(mean_squared_error(y_test,y_predLinear)))

from sklearn.metrics import mean_absolute_error
st.write("Linear Regression mean_absolute_error of is==",mean_absolute_error(y_test,y_predLinear))


from sklearn.linear_model import Ridge

ridgey = Ridge(alpha=1.0)
ridgey.fit(X_train, y_train)

y_predRidge = ridgey.predict(X_test)

#y_predRidge[0]

# importing r2_score module
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# predicting the accuracy score
score=r2_score(y_test,y_predRidge)
st.title("Ridge Regression")
st.write("Ridge Rigression r2 score is ",score)
st.write("Ridge Rigression mean_sqrd_error is==",mean_squared_error(y_test,y_predRidge))
st.write("Ridge Rigression root_mean_squared error of is==",np.sqrt(mean_squared_error(y_test,y_predRidge)))

from sklearn.metrics import mean_absolute_error
st.write("Ridge Rigression mean_absolute_error of is==", mean_absolute_error(y_test,y_predRidge))

from sklearn.linear_model import Lasso

Lassoy = Lasso(alpha=0.1)
Lassoy.fit(X_train, y_train)

y_predLasso = Lassoy.predict(X_test)

#y_predLasso[0]

# importing r2_score module
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# predicting the accuracy score
score=r2_score(y_test,y_predLasso)
st.title("Lasso Regression")
st.write("Lasso Regression r2 score is ",score)
st.write("Lasso Regression mean_sqrd_error is==",mean_squared_error(y_test,y_predLasso))
st.write("Lasso Regression root_mean_squared error of is==",np.sqrt(mean_squared_error(y_test,y_predLasso)))

from sklearn.metrics import mean_absolute_error
st.write("Lasso Regression mean_absolute_error of is==", mean_absolute_error(y_test,y_predLasso))



from sklearn.ensemble import RandomForestRegressor

RandomForestRegressory = RandomForestRegressor(max_depth=3)
RandomForestRegressory.fit(X_train, y_train)

y_predRandomForestRegressor = RandomForestRegressory.predict(X_test)

#y_predRandomForestRegressor[0]

# importing r2_score module
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# predicting the accuracy score
score=r2_score(y_test,y_predRandomForestRegressor)
st.title("RandomForest Regression")
st.write("RandomForest r2 score is ",score)
st.write("RandomForest mean_sqrd_error is==",mean_squared_error(y_test,y_predRandomForestRegressor))
st.write("RandomForest root_mean_squared error of is==",np.sqrt(mean_squared_error(y_test,y_predRandomForestRegressor)))

from sklearn.metrics import mean_absolute_error
st.write("RandomForest mean_absolute_error of is==",mean_absolute_error(y_test,y_predRandomForestRegressor))

from sklearn.tree import DecisionTreeRegressor

DecisionTreeRegressory = DecisionTreeRegressor()
DecisionTreeRegressory.fit(X_train, y_train)

y_predDecisionTreeRegressor = DecisionTreeRegressory.predict(X_test)

#y_predDecisionTreeRegressor[0]

# importing r2_score module
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# predicting the accuracy score
score=r2_score(y_test,y_predDecisionTreeRegressor)
st.title("DecisionTree Regression")
st.write("DecisionTree r2 score is ",score)
st.write("DecisionTree mean_sqrd_error is==",mean_squared_error(y_test,y_predDecisionTreeRegressor))
st.write("DecisionTree root_mean_squared error of is==",np.sqrt(mean_squared_error(y_test,y_predDecisionTreeRegressor)))

from sklearn.metrics import mean_absolute_error
st.write("DecisionTree mean_absolute_error of is==",mean_absolute_error(y_test,y_predDecisionTreeRegressor))
