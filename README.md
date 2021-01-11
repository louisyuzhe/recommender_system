# Type of Recommendation Systems
### 1. Simple Recommender: 
This system used overall TMDB Vote Count and Vote Averages to build Top Movies Charts, for specific genre in general. The IMDB Weighted Rating System was used to calculate ratings on which the sorting was finally performed.

### 2. Content Based Recommender: 
1. Movie overview and taglines based, 
2. Cast, crew, genre and keywords based. A simple filter is added to give greater preference to movies with more votes and higher ratings.

### 3. Collaborative Filtering: 
Used Surprise module to build a collaborative filter based on single value decomposition. The RMSE obtained was less than 1 and the engine gave estimated ratings for a given user and movie.

### 4. Hybrid Engine: 
Combined content and collaborative filterting to build an engine that gave movie suggestions to a particular user based on the estimated ratings that it had internally calculated for that user.


```python
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
#from nltk.stem.wordnet import WordNetLemmatibzer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD

import warnings; warnings.simplefilter('ignore')
```


```python
md = pd.read_csv('dataset/movie/movies_metadata.csv')
md.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>adult</th>
      <th>belongs_to_collection</th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>imdb_id</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>...</th>
      <th>release_date</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>video</th>
      <th>vote_average</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>{'id': 10194, 'name': 'Toy Story Collection', ...</td>
      <td>30000000</td>
      <td>[{'id': 16, 'name': 'Animation'}, {'id': 35, '...</td>
      <td>http://toystory.disney.com/toy-story</td>
      <td>862</td>
      <td>tt0114709</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>Led by Woody, Andy's toys live happily in his ...</td>
      <td>...</td>
      <td>10/30/1995</td>
      <td>373554033.0</td>
      <td>81.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Toy Story</td>
      <td>False</td>
      <td>7.7</td>
      <td>5415.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>NaN</td>
      <td>65000000</td>
      <td>[{'id': 12, 'name': 'Adventure'}, {'id': 14, '...</td>
      <td>NaN</td>
      <td>8844</td>
      <td>tt0113497</td>
      <td>en</td>
      <td>Jumanji</td>
      <td>When siblings Judy and Peter discover an encha...</td>
      <td>...</td>
      <td>12/15/1995</td>
      <td>262797249.0</td>
      <td>104.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}, {'iso...</td>
      <td>Released</td>
      <td>Roll the dice and unleash the excitement!</td>
      <td>Jumanji</td>
      <td>False</td>
      <td>6.9</td>
      <td>2413.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>{'id': 119050, 'name': 'Grumpy Old Men Collect...</td>
      <td>0</td>
      <td>[{'id': 10749, 'name': 'Romance'}, {'id': 35, ...</td>
      <td>NaN</td>
      <td>15602</td>
      <td>tt0113228</td>
      <td>en</td>
      <td>Grumpier Old Men</td>
      <td>A family wedding reignites the ancient feud be...</td>
      <td>...</td>
      <td>12/22/1995</td>
      <td>0.0</td>
      <td>101.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>Still Yelling. Still Fighting. Still Ready for...</td>
      <td>Grumpier Old Men</td>
      <td>False</td>
      <td>6.5</td>
      <td>92.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>NaN</td>
      <td>16000000</td>
      <td>[{'id': 35, 'name': 'Comedy'}, {'id': 18, 'nam...</td>
      <td>NaN</td>
      <td>31357</td>
      <td>tt0114885</td>
      <td>en</td>
      <td>Waiting to Exhale</td>
      <td>Cheated on, mistreated and stepped on, the wom...</td>
      <td>...</td>
      <td>12/22/1995</td>
      <td>81452156.0</td>
      <td>127.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>Friends are the people who let you be yourself...</td>
      <td>Waiting to Exhale</td>
      <td>False</td>
      <td>6.1</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>{'id': 96871, 'name': 'Father of the Bride Col...</td>
      <td>0</td>
      <td>[{'id': 35, 'name': 'Comedy'}]</td>
      <td>NaN</td>
      <td>11862</td>
      <td>tt0113041</td>
      <td>en</td>
      <td>Father of the Bride Part II</td>
      <td>Just when George Banks has recovered from his ...</td>
      <td>...</td>
      <td>2/10/1995</td>
      <td>76578911.0</td>
      <td>106.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>Just When His World Is Back To Normal... He's ...</td>
      <td>Father of the Bride Part II</td>
      <td>False</td>
      <td>5.7</td>
      <td>173.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



# Simple Recommender

#### Generalized recommendation based on movie popularity and genre to every user. For example, movies that are more popular and more critically acclaimed will have a higher probability of being liked by the average audience.

Sort movies based on ratings (The Movie Database (TMDb) Ratings) and popularity, then display the top movies list. 


```python
md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])\
```

Now, determine an appropriate value for m, the minimum votes required to be listed in the chart. 95th percentile will be used as cutoff. For a movie to feature in the charts, it must have more votes than at least 95% of the movies in the list.


```python
vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
C
```




    5.244896612406511



#### Average rating for a movie on TMDB is 5.244 on a scale of 10


```python
m = vote_counts.quantile(0.95)
m
```




    434.0



#### As shown, to qualify to be considered for the chart, a movie has to have at least 434 votes on TMDB


```python
md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
```


```python
qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
qualified.shape
```




    (2274, 6)



#### 2274 Movies qualify to be on the top chart

### IMDB's weighted rating formula is used.


```python
def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)
qualified['wr'] = qualified.apply(weighted_rating, axis=1)

#Choose top 250
qualified = qualified.sort_values('wr', ascending=False).head(250)
```


```python
len(qualified)
```




    250




```python
qualified.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>year</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>popularity</th>
      <th>genres</th>
      <th>wr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15480</th>
      <td>Inception</td>
      <td>2010</td>
      <td>14075</td>
      <td>8</td>
      <td>29.108149</td>
      <td>[Action, Thriller, Science Fiction, Mystery, A...</td>
      <td>7.917588</td>
    </tr>
    <tr>
      <th>12481</th>
      <td>The Dark Knight</td>
      <td>2008</td>
      <td>12269</td>
      <td>8</td>
      <td>123.167259</td>
      <td>[Drama, Action, Crime, Thriller]</td>
      <td>7.905871</td>
    </tr>
    <tr>
      <th>22878</th>
      <td>Interstellar</td>
      <td>2014</td>
      <td>11187</td>
      <td>8</td>
      <td>32.213481</td>
      <td>[Adventure, Drama, Science Fiction]</td>
      <td>7.897107</td>
    </tr>
    <tr>
      <th>2843</th>
      <td>Fight Club</td>
      <td>1999</td>
      <td>9678</td>
      <td>8</td>
      <td>63.869599</td>
      <td>[Drama]</td>
      <td>7.881753</td>
    </tr>
    <tr>
      <th>4863</th>
      <td>The Lord of the Rings: The Fellowship of the Ring</td>
      <td>2001</td>
      <td>8892</td>
      <td>8</td>
      <td>32.070725</td>
      <td>[Adventure, Fantasy, Action]</td>
      <td>7.871787</td>
    </tr>
    <tr>
      <th>292</th>
      <td>Pulp Fiction</td>
      <td>1994</td>
      <td>8670</td>
      <td>8</td>
      <td>140.950236</td>
      <td>[Thriller, Crime]</td>
      <td>7.868660</td>
    </tr>
    <tr>
      <th>314</th>
      <td>The Shawshank Redemption</td>
      <td>1994</td>
      <td>8358</td>
      <td>8</td>
      <td>51.645403</td>
      <td>[Drama, Crime]</td>
      <td>7.864000</td>
    </tr>
    <tr>
      <th>7000</th>
      <td>The Lord of the Rings: The Return of the King</td>
      <td>2003</td>
      <td>8226</td>
      <td>8</td>
      <td>29.324358</td>
      <td>[Adventure, Fantasy, Action]</td>
      <td>7.861927</td>
    </tr>
    <tr>
      <th>351</th>
      <td>Forrest Gump</td>
      <td>1994</td>
      <td>8147</td>
      <td>8</td>
      <td>48.307194</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>7.860656</td>
    </tr>
    <tr>
      <th>5814</th>
      <td>The Lord of the Rings: The Two Towers</td>
      <td>2002</td>
      <td>7641</td>
      <td>8</td>
      <td>29.423537</td>
      <td>[Adventure, Fantasy, Action]</td>
      <td>7.851924</td>
    </tr>
    <tr>
      <th>256</th>
      <td>Star Wars</td>
      <td>1977</td>
      <td>6778</td>
      <td>8</td>
      <td>42.149697</td>
      <td>[Adventure, Action, Science Fiction]</td>
      <td>7.834205</td>
    </tr>
    <tr>
      <th>1225</th>
      <td>Back to the Future</td>
      <td>1985</td>
      <td>6239</td>
      <td>8</td>
      <td>25.778509</td>
      <td>[Adventure, Comedy, Science Fiction, Family]</td>
      <td>7.820813</td>
    </tr>
    <tr>
      <th>834</th>
      <td>The Godfather</td>
      <td>1972</td>
      <td>6024</td>
      <td>8</td>
      <td>41.109264</td>
      <td>[Drama, Crime]</td>
      <td>7.814847</td>
    </tr>
    <tr>
      <th>1154</th>
      <td>The Empire Strikes Back</td>
      <td>1980</td>
      <td>5998</td>
      <td>8</td>
      <td>19.470959</td>
      <td>[Adventure, Action, Science Fiction]</td>
      <td>7.814099</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Se7en</td>
      <td>1995</td>
      <td>5915</td>
      <td>8</td>
      <td>18.457430</td>
      <td>[Crime, Mystery, Thriller]</td>
      <td>7.811669</td>
    </tr>
  </tbody>
</table>
</div>



The chart indicates a strong bias of TMDB Users towards particular genres and directors (Christopher Nolan)

### Generate top chart based on genre


```python
def build_chart(genre, percentile=0.85):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    
    return qualified
```

Use 85th percentile instead, and split movie with multiple genres into seperate row


```python
s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = md.drop('genres', axis=1).join(s)
```


```python
gen_md
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>adult</th>
      <th>belongs_to_collection</th>
      <th>budget</th>
      <th>homepage</th>
      <th>id</th>
      <th>imdb_id</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>...</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>video</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>year</th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>{'id': 10194, 'name': 'Toy Story Collection', ...</td>
      <td>30000000</td>
      <td>http://toystory.disney.com/toy-story</td>
      <td>862</td>
      <td>tt0114709</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>Led by Woody, Andy's toys live happily in his ...</td>
      <td>21.946943</td>
      <td>...</td>
      <td>81.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Toy Story</td>
      <td>False</td>
      <td>7.7</td>
      <td>5415.0</td>
      <td>1995</td>
      <td>Animation</td>
    </tr>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>{'id': 10194, 'name': 'Toy Story Collection', ...</td>
      <td>30000000</td>
      <td>http://toystory.disney.com/toy-story</td>
      <td>862</td>
      <td>tt0114709</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>Led by Woody, Andy's toys live happily in his ...</td>
      <td>21.946943</td>
      <td>...</td>
      <td>81.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Toy Story</td>
      <td>False</td>
      <td>7.7</td>
      <td>5415.0</td>
      <td>1995</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>{'id': 10194, 'name': 'Toy Story Collection', ...</td>
      <td>30000000</td>
      <td>http://toystory.disney.com/toy-story</td>
      <td>862</td>
      <td>tt0114709</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>Led by Woody, Andy's toys live happily in his ...</td>
      <td>21.946943</td>
      <td>...</td>
      <td>81.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Toy Story</td>
      <td>False</td>
      <td>7.7</td>
      <td>5415.0</td>
      <td>1995</td>
      <td>Family</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>NaN</td>
      <td>65000000</td>
      <td>NaN</td>
      <td>8844</td>
      <td>tt0113497</td>
      <td>en</td>
      <td>Jumanji</td>
      <td>When siblings Judy and Peter discover an encha...</td>
      <td>17.015539</td>
      <td>...</td>
      <td>104.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}, {'iso...</td>
      <td>Released</td>
      <td>Roll the dice and unleash the excitement!</td>
      <td>Jumanji</td>
      <td>False</td>
      <td>6.9</td>
      <td>2413.0</td>
      <td>1995</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>NaN</td>
      <td>65000000</td>
      <td>NaN</td>
      <td>8844</td>
      <td>tt0113497</td>
      <td>en</td>
      <td>Jumanji</td>
      <td>When siblings Judy and Peter discover an encha...</td>
      <td>17.015539</td>
      <td>...</td>
      <td>104.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}, {'iso...</td>
      <td>Released</td>
      <td>Roll the dice and unleash the excitement!</td>
      <td>Jumanji</td>
      <td>False</td>
      <td>6.9</td>
      <td>2413.0</td>
      <td>1995</td>
      <td>Fantasy</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>45460</th>
      <td>False</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>67758</td>
      <td>tt0303758</td>
      <td>en</td>
      <td>Betrayal</td>
      <td>When one of her hits goes wrong, a professiona...</td>
      <td>0.903007</td>
      <td>...</td>
      <td>90.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>A deadly game of wits.</td>
      <td>Betrayal</td>
      <td>False</td>
      <td>3.8</td>
      <td>6.0</td>
      <td>2003</td>
      <td>Action</td>
    </tr>
    <tr>
      <th>45460</th>
      <td>False</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>67758</td>
      <td>tt0303758</td>
      <td>en</td>
      <td>Betrayal</td>
      <td>When one of her hits goes wrong, a professiona...</td>
      <td>0.903007</td>
      <td>...</td>
      <td>90.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>A deadly game of wits.</td>
      <td>Betrayal</td>
      <td>False</td>
      <td>3.8</td>
      <td>6.0</td>
      <td>2003</td>
      <td>Drama</td>
    </tr>
    <tr>
      <th>45460</th>
      <td>False</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>67758</td>
      <td>tt0303758</td>
      <td>en</td>
      <td>Betrayal</td>
      <td>When one of her hits goes wrong, a professiona...</td>
      <td>0.903007</td>
      <td>...</td>
      <td>90.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>A deadly game of wits.</td>
      <td>Betrayal</td>
      <td>False</td>
      <td>3.8</td>
      <td>6.0</td>
      <td>2003</td>
      <td>Thriller</td>
    </tr>
    <tr>
      <th>45461</th>
      <td>False</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>227506</td>
      <td>tt0008536</td>
      <td>en</td>
      <td>Satana likuyushchiy</td>
      <td>In a small town live two brothers, one a minis...</td>
      <td>0.003503</td>
      <td>...</td>
      <td>87.0</td>
      <td>[]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Satan Triumphant</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1917</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>45462</th>
      <td>False</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>461257</td>
      <td>tt6980792</td>
      <td>en</td>
      <td>Queerama</td>
      <td>50 years after decriminalisation of homosexual...</td>
      <td>0.163015</td>
      <td>...</td>
      <td>75.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Queerama</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2017</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>93536 rows × 25 columns</p>
</div>




```python
build_chart('Action').head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>year</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>popularity</th>
      <th>wr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15480</th>
      <td>Inception</td>
      <td>2010</td>
      <td>14075</td>
      <td>8</td>
      <td>29.108149</td>
      <td>7.955099</td>
    </tr>
    <tr>
      <th>12481</th>
      <td>The Dark Knight</td>
      <td>2008</td>
      <td>12269</td>
      <td>8</td>
      <td>123.167259</td>
      <td>7.948610</td>
    </tr>
    <tr>
      <th>4863</th>
      <td>The Lord of the Rings: The Fellowship of the Ring</td>
      <td>2001</td>
      <td>8892</td>
      <td>8</td>
      <td>32.070725</td>
      <td>7.929579</td>
    </tr>
    <tr>
      <th>7000</th>
      <td>The Lord of the Rings: The Return of the King</td>
      <td>2003</td>
      <td>8226</td>
      <td>8</td>
      <td>29.324358</td>
      <td>7.924031</td>
    </tr>
    <tr>
      <th>5814</th>
      <td>The Lord of the Rings: The Two Towers</td>
      <td>2002</td>
      <td>7641</td>
      <td>8</td>
      <td>29.423537</td>
      <td>7.918382</td>
    </tr>
    <tr>
      <th>256</th>
      <td>Star Wars</td>
      <td>1977</td>
      <td>6778</td>
      <td>8</td>
      <td>42.149697</td>
      <td>7.908327</td>
    </tr>
    <tr>
      <th>1154</th>
      <td>The Empire Strikes Back</td>
      <td>1980</td>
      <td>5998</td>
      <td>8</td>
      <td>19.470959</td>
      <td>7.896841</td>
    </tr>
    <tr>
      <th>4135</th>
      <td>Scarface</td>
      <td>1983</td>
      <td>3017</td>
      <td>8</td>
      <td>11.299673</td>
      <td>7.802046</td>
    </tr>
    <tr>
      <th>9430</th>
      <td>Oldboy</td>
      <td>2003</td>
      <td>2000</td>
      <td>8</td>
      <td>10.616859</td>
      <td>7.711649</td>
    </tr>
    <tr>
      <th>1910</th>
      <td>Seven Samurai</td>
      <td>1954</td>
      <td>892</td>
      <td>8</td>
      <td>15.017770</td>
      <td>7.426145</td>
    </tr>
    <tr>
      <th>43187</th>
      <td>Band of Brothers</td>
      <td>2001</td>
      <td>725</td>
      <td>8</td>
      <td>7.903731</td>
      <td>7.325485</td>
    </tr>
    <tr>
      <th>1215</th>
      <td>M</td>
      <td>1931</td>
      <td>465</td>
      <td>8</td>
      <td>12.752421</td>
      <td>7.072073</td>
    </tr>
    <tr>
      <th>14551</th>
      <td>Avatar</td>
      <td>2009</td>
      <td>12114</td>
      <td>7</td>
      <td>185.070892</td>
      <td>6.966363</td>
    </tr>
    <tr>
      <th>17818</th>
      <td>The Avengers</td>
      <td>2012</td>
      <td>12000</td>
      <td>7</td>
      <td>89.887648</td>
      <td>6.966049</td>
    </tr>
    <tr>
      <th>26563</th>
      <td>Deadpool</td>
      <td>2016</td>
      <td>11444</td>
      <td>7</td>
      <td>187.860492</td>
      <td>6.964431</td>
    </tr>
  </tbody>
</table>
</div>



# Content Based Recommender

### Personalized recommendations - Computes similarity between movies based on certain metrics and suggests movies that are most similar to a particular movie that a user liked.


```python
#links.csv contains the reference bewteen imdbId	tmdbId
links_small = pd.read_csv('dataset/movie/links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
```


```python
links_small
```




    0          862
    1         8844
    2        15602
    3        31357
    4        11862
             ...  
    9120    402672
    9121    315011
    9122    391698
    9123    137608
    9124    410803
    Name: tmdbId, Length: 9112, dtype: int32




```python
md['id'] = md['id'].astype('int')
small_md = md[md['id'].isin(links_small)]
small_md.shape
```




    (9099, 25)




```python
small_md
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>adult</th>
      <th>belongs_to_collection</th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>imdb_id</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>...</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>video</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>{'id': 10194, 'name': 'Toy Story Collection', ...</td>
      <td>30000000</td>
      <td>[Animation, Comedy, Family]</td>
      <td>http://toystory.disney.com/toy-story</td>
      <td>862</td>
      <td>tt0114709</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>Led by Woody, Andy's toys live happily in his ...</td>
      <td>...</td>
      <td>373554033.0</td>
      <td>81.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Toy Story</td>
      <td>False</td>
      <td>7.7</td>
      <td>5415.0</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>NaN</td>
      <td>65000000</td>
      <td>[Adventure, Fantasy, Family]</td>
      <td>NaN</td>
      <td>8844</td>
      <td>tt0113497</td>
      <td>en</td>
      <td>Jumanji</td>
      <td>When siblings Judy and Peter discover an encha...</td>
      <td>...</td>
      <td>262797249.0</td>
      <td>104.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}, {'iso...</td>
      <td>Released</td>
      <td>Roll the dice and unleash the excitement!</td>
      <td>Jumanji</td>
      <td>False</td>
      <td>6.9</td>
      <td>2413.0</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>{'id': 119050, 'name': 'Grumpy Old Men Collect...</td>
      <td>0</td>
      <td>[Romance, Comedy]</td>
      <td>NaN</td>
      <td>15602</td>
      <td>tt0113228</td>
      <td>en</td>
      <td>Grumpier Old Men</td>
      <td>A family wedding reignites the ancient feud be...</td>
      <td>...</td>
      <td>0.0</td>
      <td>101.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>Still Yelling. Still Fighting. Still Ready for...</td>
      <td>Grumpier Old Men</td>
      <td>False</td>
      <td>6.5</td>
      <td>92.0</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>NaN</td>
      <td>16000000</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>NaN</td>
      <td>31357</td>
      <td>tt0114885</td>
      <td>en</td>
      <td>Waiting to Exhale</td>
      <td>Cheated on, mistreated and stepped on, the wom...</td>
      <td>...</td>
      <td>81452156.0</td>
      <td>127.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>Friends are the people who let you be yourself...</td>
      <td>Waiting to Exhale</td>
      <td>False</td>
      <td>6.1</td>
      <td>34.0</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>{'id': 96871, 'name': 'Father of the Bride Col...</td>
      <td>0</td>
      <td>[Comedy]</td>
      <td>NaN</td>
      <td>11862</td>
      <td>tt0113041</td>
      <td>en</td>
      <td>Father of the Bride Part II</td>
      <td>Just when George Banks has recovered from his ...</td>
      <td>...</td>
      <td>76578911.0</td>
      <td>106.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>Just When His World Is Back To Normal... He's ...</td>
      <td>Father of the Bride Part II</td>
      <td>False</td>
      <td>5.7</td>
      <td>173.0</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>40221</th>
      <td>False</td>
      <td>NaN</td>
      <td>15000000</td>
      <td>[Action, Adventure, Drama, Horror, Science Fic...</td>
      <td>NaN</td>
      <td>315011</td>
      <td>tt4262980</td>
      <td>ja</td>
      <td>シン・ゴジラ</td>
      <td>From the mind behind Evangelion comes a hit la...</td>
      <td>...</td>
      <td>77000000.0</td>
      <td>120.0</td>
      <td>[{'iso_639_1': 'it', 'name': 'Italiano'}, {'is...</td>
      <td>Released</td>
      <td>A god incarnate. A city doomed.</td>
      <td>Shin Godzilla</td>
      <td>False</td>
      <td>6.6</td>
      <td>152.0</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>40500</th>
      <td>False</td>
      <td>NaN</td>
      <td>0</td>
      <td>[Documentary, Music]</td>
      <td>http://www.thebeatlesliveproject.com/</td>
      <td>391698</td>
      <td>tt2531318</td>
      <td>en</td>
      <td>The Beatles: Eight Days a Week - The Touring Y...</td>
      <td>The band stormed Europe in 1963, and, in 1964,...</td>
      <td>...</td>
      <td>0.0</td>
      <td>99.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>The band you know. The story you don't.</td>
      <td>The Beatles: Eight Days a Week - The Touring Y...</td>
      <td>False</td>
      <td>7.6</td>
      <td>92.0</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>44818</th>
      <td>False</td>
      <td>{'id': 34055, 'name': 'Pokémon Collection', 'p...</td>
      <td>16000000</td>
      <td>[Adventure, Fantasy, Animation, Action, Family]</td>
      <td>http://movies.warnerbros.com/pk3/</td>
      <td>10991</td>
      <td>tt0235679</td>
      <td>ja</td>
      <td>Pokémon 3: The Movie</td>
      <td>When Molly Hale's sadness of her father's disa...</td>
      <td>...</td>
      <td>68411275.0</td>
      <td>93.0</td>
      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>
      <td>Released</td>
      <td>Pokémon: Spell of the Unknown</td>
      <td>Pokémon: Spell of the Unknown</td>
      <td>False</td>
      <td>6.0</td>
      <td>144.0</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>44823</th>
      <td>False</td>
      <td>{'id': 34055, 'name': 'Pokémon Collection', 'p...</td>
      <td>0</td>
      <td>[Adventure, Fantasy, Animation, Science Fictio...</td>
      <td>http://www.pokemon.com/us/movies/movie-pokemon...</td>
      <td>12600</td>
      <td>tt0287635</td>
      <td>ja</td>
      <td>劇場版ポケットモンスター セレビィ 時を越えた遭遇（であい）</td>
      <td>All your favorite Pokémon characters are back,...</td>
      <td>...</td>
      <td>28023563.0</td>
      <td>75.0</td>
      <td>[{'iso_639_1': 'ja', 'name': '日本語'}]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Pokémon 4Ever: Celebi - Voice of the Forest</td>
      <td>False</td>
      <td>5.7</td>
      <td>82.0</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>45262</th>
      <td>False</td>
      <td>NaN</td>
      <td>0</td>
      <td>[Comedy, Drama]</td>
      <td>NaN</td>
      <td>265189</td>
      <td>tt2121382</td>
      <td>sv</td>
      <td>Turist</td>
      <td>While holidaying in the French Alps, a Swedish...</td>
      <td>...</td>
      <td>1359497.0</td>
      <td>118.0</td>
      <td>[{'iso_639_1': 'fr', 'name': 'Français'}, {'is...</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Force Majeure</td>
      <td>False</td>
      <td>6.8</td>
      <td>255.0</td>
      <td>2014</td>
    </tr>
  </tbody>
</table>
<p>9099 rows × 25 columns</p>
</div>



## Movie Description Based Recommender

Movie descriptions and taglines based recommender


```python
small_md['tagline'] = small_md['tagline'].fillna('')
small_md['description'] = small_md['overview'] + small_md['tagline']
small_md['description'] = small_md['description'].fillna('')
```

TF-IDF Vectorizer


```python
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(small_md['description'])
tfidf_matrix.shape
```




    (9099, 268123)



Cosine Similarity
Cosine Similarity willl be used to calculate a numeric quantity that denotes the similarity between two movies.


```python
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim[0]
```




    array([1.        , 0.00680476, 0.        , ..., 0.        , 0.00344913,
           0.        ])



Now all the movies in the dataset has a pairwise cosine similarity matrix 


```python
small_md = small_md.reset_index()
titles = small_md['title']
indices = pd.Series(small_md.index, index=small_md['title'])
```


```python
indices
```




    title
    Toy Story                                                0
    Jumanji                                                  1
    Grumpier Old Men                                         2
    Waiting to Exhale                                        3
    Father of the Bride Part II                              4
                                                          ... 
    Shin Godzilla                                         9094
    The Beatles: Eight Days a Week - The Touring Years    9095
    Pokémon: Spell of the Unknown                         9096
    Pokémon 4Ever: Celebi - Voice of the Forest           9097
    Force Majeure                                         9098
    Length: 9099, dtype: int64



 #### Function that returns the 30 most similar movies based on the cosine similarity score of input movie


```python
def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]
```


```python
get_recommendations('The Dark Knight')
```




    7931                      The Dark Knight Rises
    132                              Batman Forever
    1113                             Batman Returns
    8227    Batman: The Dark Knight Returns, Part 2
    7565                 Batman: Under the Red Hood
    524                                      Batman
    7901                           Batman: Year One
    2579               Batman: Mask of the Phantasm
    2696                                        JFK
    8165    Batman: The Dark Knight Returns, Part 1
    6144                              Batman Begins
    7933         Sherlock Holmes: A Game of Shadows
    5511                            To End All Wars
    4489                                      Q & A
    7344                        Law Abiding Citizen
    7242                  The File on Thelma Jordon
    3537                               Criminal Law
    2893                              Flying Tigers
    1135                   Night Falls on Manhattan
    8680                          The Young Savages
    8917         Batman v Superman: Dawn of Justice
    1240                             Batman & Robin
    6740                                Rush Hour 3
    1652                            The Shaggy D.A.
    6667                                   Fracture
    4028                                 The Rookie
    8371       Justice League: Crisis on Two Earths
    8719                                 By the Gun
    3730                    Dr. Mabuse, the Gambler
    4160                     The Master of Disguise
    Name: title, dtype: object



As observed, the system takes conderation of the description and taglines of Batman: Dark knight and recommend all other Batman movie, followed by Detective, Superhero, Crime, etc

## Metadata Based Recommender

### Based on genre, keywords, cast and crew 


```python
credits = pd.read_csv('dataset/movie/credits.csv')
keywords = pd.read_csv('dataset/movie/keywords.csv')
```


```python
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
md['id'] = md['id'].astype('int')
md.shape
```




    (45463, 25)




```python
md = md.merge(credits, on='id')
md = md.merge(keywords, on='id')
sub_md = md[md['id'].isin(links_small)]
sub_md.shape
```




    (9219, 28)




```python
sub_md.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>adult</th>
      <th>belongs_to_collection</th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>imdb_id</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>...</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>video</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>year</th>
      <th>cast</th>
      <th>crew</th>
      <th>keywords</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>{'id': 10194, 'name': 'Toy Story Collection', ...</td>
      <td>30000000</td>
      <td>[Animation, Comedy, Family]</td>
      <td>http://toystory.disney.com/toy-story</td>
      <td>862</td>
      <td>tt0114709</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>Led by Woody, Andy's toys live happily in his ...</td>
      <td>...</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Toy Story</td>
      <td>False</td>
      <td>7.7</td>
      <td>5415.0</td>
      <td>1995</td>
      <td>[{'cast_id': 14, 'character': 'Woody (voice)',...</td>
      <td>[{'credit_id': '52fe4284c3a36847f8024f49', 'de...</td>
      <td>[{'id': 931, 'name': 'jealousy'}, {'id': 4290,...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>NaN</td>
      <td>65000000</td>
      <td>[Adventure, Fantasy, Family]</td>
      <td>NaN</td>
      <td>8844</td>
      <td>tt0113497</td>
      <td>en</td>
      <td>Jumanji</td>
      <td>When siblings Judy and Peter discover an encha...</td>
      <td>...</td>
      <td>Released</td>
      <td>Roll the dice and unleash the excitement!</td>
      <td>Jumanji</td>
      <td>False</td>
      <td>6.9</td>
      <td>2413.0</td>
      <td>1995</td>
      <td>[{'cast_id': 1, 'character': 'Alan Parrish', '...</td>
      <td>[{'credit_id': '52fe44bfc3a36847f80a7cd1', 'de...</td>
      <td>[{'id': 10090, 'name': 'board game'}, {'id': 1...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>{'id': 119050, 'name': 'Grumpy Old Men Collect...</td>
      <td>0</td>
      <td>[Romance, Comedy]</td>
      <td>NaN</td>
      <td>15602</td>
      <td>tt0113228</td>
      <td>en</td>
      <td>Grumpier Old Men</td>
      <td>A family wedding reignites the ancient feud be...</td>
      <td>...</td>
      <td>Released</td>
      <td>Still Yelling. Still Fighting. Still Ready for...</td>
      <td>Grumpier Old Men</td>
      <td>False</td>
      <td>6.5</td>
      <td>92.0</td>
      <td>1995</td>
      <td>[{'cast_id': 2, 'character': 'Max Goldman', 'c...</td>
      <td>[{'credit_id': '52fe466a9251416c75077a89', 'de...</td>
      <td>[{'id': 1495, 'name': 'fishing'}, {'id': 12392...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>NaN</td>
      <td>16000000</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>NaN</td>
      <td>31357</td>
      <td>tt0114885</td>
      <td>en</td>
      <td>Waiting to Exhale</td>
      <td>Cheated on, mistreated and stepped on, the wom...</td>
      <td>...</td>
      <td>Released</td>
      <td>Friends are the people who let you be yourself...</td>
      <td>Waiting to Exhale</td>
      <td>False</td>
      <td>6.1</td>
      <td>34.0</td>
      <td>1995</td>
      <td>[{'cast_id': 1, 'character': "Savannah 'Vannah...</td>
      <td>[{'credit_id': '52fe44779251416c91011acb', 'de...</td>
      <td>[{'id': 818, 'name': 'based on novel'}, {'id':...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>{'id': 96871, 'name': 'Father of the Bride Col...</td>
      <td>0</td>
      <td>[Comedy]</td>
      <td>NaN</td>
      <td>11862</td>
      <td>tt0113041</td>
      <td>en</td>
      <td>Father of the Bride Part II</td>
      <td>Just when George Banks has recovered from his ...</td>
      <td>...</td>
      <td>Released</td>
      <td>Just When His World Is Back To Normal... He's ...</td>
      <td>Father of the Bride Part II</td>
      <td>False</td>
      <td>5.7</td>
      <td>173.0</td>
      <td>1995</td>
      <td>[{'cast_id': 1, 'character': 'George Banks', '...</td>
      <td>[{'credit_id': '52fe44959251416c75039ed7', 'de...</td>
      <td>[{'id': 1009, 'name': 'baby'}, {'id': 1599, 'n...</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>



To make things lless complicated,
1. Crew will be represented by the director
2. Only top 3 actors willl be choseden to represent the cast.


```python
sub_md['cast'] = sub_md['cast'].apply(literal_eval)
sub_md['crew'] = sub_md['crew'].apply(literal_eval)
sub_md['keywords'] = sub_md['keywords'].apply(literal_eval)
sub_md['cast_size'] = sub_md['cast'].apply(lambda x: len(x))
sub_md['crew_size'] = sub_md['crew'].apply(lambda x: len(x))
```


```python
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
sub_md['director'] = sub_md['crew'].apply(get_director)
```


```python
sub_md['cast'] = sub_md['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
sub_md['cast'] = sub_md['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
```


```python
sub_md['keywords'] = sub_md['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
```


```python
sub_md.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>adult</th>
      <th>belongs_to_collection</th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>imdb_id</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>...</th>
      <th>video</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>year</th>
      <th>cast</th>
      <th>crew</th>
      <th>keywords</th>
      <th>cast_size</th>
      <th>crew_size</th>
      <th>director</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>{'id': 10194, 'name': 'Toy Story Collection', ...</td>
      <td>30000000</td>
      <td>[Animation, Comedy, Family]</td>
      <td>http://toystory.disney.com/toy-story</td>
      <td>862</td>
      <td>tt0114709</td>
      <td>en</td>
      <td>Toy Story</td>
      <td>Led by Woody, Andy's toys live happily in his ...</td>
      <td>...</td>
      <td>False</td>
      <td>7.7</td>
      <td>5415.0</td>
      <td>1995</td>
      <td>[Tom Hanks, Tim Allen, Don Rickles]</td>
      <td>[{'credit_id': '52fe4284c3a36847f8024f49', 'de...</td>
      <td>[jealousy, toy, boy, friendship, friends, riva...</td>
      <td>13</td>
      <td>106</td>
      <td>John Lasseter</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>NaN</td>
      <td>65000000</td>
      <td>[Adventure, Fantasy, Family]</td>
      <td>NaN</td>
      <td>8844</td>
      <td>tt0113497</td>
      <td>en</td>
      <td>Jumanji</td>
      <td>When siblings Judy and Peter discover an encha...</td>
      <td>...</td>
      <td>False</td>
      <td>6.9</td>
      <td>2413.0</td>
      <td>1995</td>
      <td>[Robin Williams, Jonathan Hyde, Kirsten Dunst]</td>
      <td>[{'credit_id': '52fe44bfc3a36847f80a7cd1', 'de...</td>
      <td>[board game, disappearance, based on children'...</td>
      <td>26</td>
      <td>16</td>
      <td>Joe Johnston</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>{'id': 119050, 'name': 'Grumpy Old Men Collect...</td>
      <td>0</td>
      <td>[Romance, Comedy]</td>
      <td>NaN</td>
      <td>15602</td>
      <td>tt0113228</td>
      <td>en</td>
      <td>Grumpier Old Men</td>
      <td>A family wedding reignites the ancient feud be...</td>
      <td>...</td>
      <td>False</td>
      <td>6.5</td>
      <td>92.0</td>
      <td>1995</td>
      <td>[Walter Matthau, Jack Lemmon, Ann-Margret]</td>
      <td>[{'credit_id': '52fe466a9251416c75077a89', 'de...</td>
      <td>[fishing, best friend, duringcreditsstinger, o...</td>
      <td>7</td>
      <td>4</td>
      <td>Howard Deutch</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>NaN</td>
      <td>16000000</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>NaN</td>
      <td>31357</td>
      <td>tt0114885</td>
      <td>en</td>
      <td>Waiting to Exhale</td>
      <td>Cheated on, mistreated and stepped on, the wom...</td>
      <td>...</td>
      <td>False</td>
      <td>6.1</td>
      <td>34.0</td>
      <td>1995</td>
      <td>[Whitney Houston, Angela Bassett, Loretta Devine]</td>
      <td>[{'credit_id': '52fe44779251416c91011acb', 'de...</td>
      <td>[based on novel, interracial relationship, sin...</td>
      <td>10</td>
      <td>10</td>
      <td>Forest Whitaker</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>{'id': 96871, 'name': 'Father of the Bride Col...</td>
      <td>0</td>
      <td>[Comedy]</td>
      <td>NaN</td>
      <td>11862</td>
      <td>tt0113041</td>
      <td>en</td>
      <td>Father of the Bride Part II</td>
      <td>Just when George Banks has recovered from his ...</td>
      <td>...</td>
      <td>False</td>
      <td>5.7</td>
      <td>173.0</td>
      <td>1995</td>
      <td>[Steve Martin, Diane Keaton, Martin Short]</td>
      <td>[{'credit_id': '52fe44959251416c75039ed7', 'de...</td>
      <td>[baby, midlife crisis, confidence, aging, daug...</td>
      <td>12</td>
      <td>7</td>
      <td>Charles Shyer</td>
    </tr>
    <tr>
      <th>5</th>
      <td>False</td>
      <td>NaN</td>
      <td>60000000</td>
      <td>[Action, Crime, Drama, Thriller]</td>
      <td>NaN</td>
      <td>949</td>
      <td>tt0113277</td>
      <td>en</td>
      <td>Heat</td>
      <td>Obsessive master thief, Neil McCauley leads a ...</td>
      <td>...</td>
      <td>False</td>
      <td>7.7</td>
      <td>1886.0</td>
      <td>1995</td>
      <td>[Al Pacino, Robert De Niro, Val Kilmer]</td>
      <td>[{'credit_id': '52fe4292c3a36847f802916d', 'de...</td>
      <td>[robbery, detective, bank, obsession, chase, s...</td>
      <td>65</td>
      <td>71</td>
      <td>Michael Mann</td>
    </tr>
    <tr>
      <th>6</th>
      <td>False</td>
      <td>NaN</td>
      <td>58000000</td>
      <td>[Comedy, Romance]</td>
      <td>NaN</td>
      <td>11860</td>
      <td>tt0114319</td>
      <td>en</td>
      <td>Sabrina</td>
      <td>An ugly duckling having undergone a remarkable...</td>
      <td>...</td>
      <td>False</td>
      <td>6.2</td>
      <td>141.0</td>
      <td>1995</td>
      <td>[Harrison Ford, Julia Ormond, Greg Kinnear]</td>
      <td>[{'credit_id': '52fe44959251416c75039da9', 'de...</td>
      <td>[paris, brother brother relationship, chauffeu...</td>
      <td>57</td>
      <td>53</td>
      <td>Sydney Pollack</td>
    </tr>
    <tr>
      <th>7</th>
      <td>False</td>
      <td>NaN</td>
      <td>0</td>
      <td>[Action, Adventure, Drama, Family]</td>
      <td>NaN</td>
      <td>45325</td>
      <td>tt0112302</td>
      <td>en</td>
      <td>Tom and Huck</td>
      <td>A mischievous young boy, Tom Sawyer, witnesses...</td>
      <td>...</td>
      <td>False</td>
      <td>5.4</td>
      <td>45.0</td>
      <td>1995</td>
      <td>[Jonathan Taylor Thomas, Brad Renfro, Rachael ...</td>
      <td>[{'credit_id': '52fe46bdc3a36847f810f797', 'de...</td>
      <td>[]</td>
      <td>7</td>
      <td>4</td>
      <td>Peter Hewitt</td>
    </tr>
    <tr>
      <th>8</th>
      <td>False</td>
      <td>NaN</td>
      <td>35000000</td>
      <td>[Action, Adventure, Thriller]</td>
      <td>NaN</td>
      <td>9091</td>
      <td>tt0114576</td>
      <td>en</td>
      <td>Sudden Death</td>
      <td>International action superstar Jean Claude Van...</td>
      <td>...</td>
      <td>False</td>
      <td>5.5</td>
      <td>174.0</td>
      <td>1995</td>
      <td>[Jean-Claude Van Damme, Powers Boothe, Dorian ...</td>
      <td>[{'credit_id': '52fe44dbc3a36847f80ae0f1', 'de...</td>
      <td>[terrorist, hostage, explosive, vice president]</td>
      <td>6</td>
      <td>9</td>
      <td>Peter Hyams</td>
    </tr>
    <tr>
      <th>9</th>
      <td>False</td>
      <td>{'id': 645, 'name': 'James Bond Collection', '...</td>
      <td>58000000</td>
      <td>[Adventure, Action, Thriller]</td>
      <td>http://www.mgm.com/view/movie/757/Goldeneye/</td>
      <td>710</td>
      <td>tt0113189</td>
      <td>en</td>
      <td>GoldenEye</td>
      <td>James Bond must unmask the mysterious head of ...</td>
      <td>...</td>
      <td>False</td>
      <td>6.6</td>
      <td>1194.0</td>
      <td>1995</td>
      <td>[Pierce Brosnan, Sean Bean, Izabella Scorupco]</td>
      <td>[{'credit_id': '52fe426ec3a36847f801e14b', 'de...</td>
      <td>[cuba, falsely accused, secret identity, compu...</td>
      <td>20</td>
      <td>46</td>
      <td>Martin Campbell</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 31 columns</p>
</div>



Strip Spaces and Convert to Lowercase from all our features


```python
sub_md['cast'] = sub_md['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
```

Mention Director 3 times to give it more weight relative to the entire cast.


```python
sub_md['director'] = sub_md['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
sub_md['director'] = sub_md['director'].apply(lambda x: [x,x, x])
```

Calculate the frequenct counts of every keyword that appears in the dataset


```python
s = sub_md.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()
s[:5]
```




    independent film        610
    woman director          550
    murder                  399
    duringcreditsstinger    327
    based on novel          318
    Name: keyword, dtype: int64



Remove keyword that only occur once


```python
s = s[s > 1]
```

Convert every word to its stem so that words such as heroes and hero are considered the same.


```python
stemmer = SnowballStemmer('english')
stemmer.stem('heroes')
```




    'hero'




```python
def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words
sub_md['keywords'] = sub_md['keywords'].apply(filter_keywords)
sub_md['keywords'] = sub_md['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
sub_md['keywords'] = sub_md['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
```

 Create a metadata dump for every movie which consists of genres, director, main actors and keywords.


```python
sub_md['metadata'] = sub_md['keywords'] + sub_md['cast'] + sub_md['director'] + sub_md['genres']
sub_md['metadata'] = sub_md['metadata'].apply(lambda x: ' '.join(x))
```

Use Count Vectorizer to create count matrix. Then,calculate the cosine similarities and return movies that are most similar.


```python
count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(sub_md['metadata'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)
```


```python
sub_md = sub_md.reset_index()
titles = sub_md['title']
indices = pd.Series(sub_md.index, index=sub_md['title'])
```


```python
def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]
```


```python
get_recommendations('The Dark Knight').head(10)
```




    8031         The Dark Knight Rises
    6218                 Batman Begins
    6623                  The Prestige
    2085                     Following
    7648                     Inception
    4145                      Insomnia
    3381                       Memento
    8613                  Interstellar
    7659    Batman: Under the Red Hood
    1134                Batman Returns
    Name: title, dtype: object



As observed, more Christopher Nolan's movies made it to the list. Besides, these movies appeared to share same genres and required more thinking. 

## Adding Popularity and Ratings to recommender

Take the top 25 movies based on similarity scores and calculate the vote of the 60th percentile movie. Then, using this as the value of m, calculate the weighted rating of each movie using IMDB's formula.


```python
def improved_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = sub_md.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified
```


```python
improved_recommendations('The Dark Knight')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>year</th>
      <th>wr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7648</th>
      <td>Inception</td>
      <td>14075</td>
      <td>8</td>
      <td>2010</td>
      <td>7.917588</td>
    </tr>
    <tr>
      <th>8613</th>
      <td>Interstellar</td>
      <td>11187</td>
      <td>8</td>
      <td>2014</td>
      <td>7.897107</td>
    </tr>
    <tr>
      <th>6623</th>
      <td>The Prestige</td>
      <td>4510</td>
      <td>8</td>
      <td>2006</td>
      <td>7.758148</td>
    </tr>
    <tr>
      <th>3381</th>
      <td>Memento</td>
      <td>4168</td>
      <td>8</td>
      <td>2000</td>
      <td>7.740175</td>
    </tr>
    <tr>
      <th>8031</th>
      <td>The Dark Knight Rises</td>
      <td>9263</td>
      <td>7</td>
      <td>2012</td>
      <td>6.921448</td>
    </tr>
    <tr>
      <th>6218</th>
      <td>Batman Begins</td>
      <td>7511</td>
      <td>7</td>
      <td>2005</td>
      <td>6.904127</td>
    </tr>
    <tr>
      <th>1134</th>
      <td>Batman Returns</td>
      <td>1706</td>
      <td>6</td>
      <td>1992</td>
      <td>5.846862</td>
    </tr>
    <tr>
      <th>132</th>
      <td>Batman Forever</td>
      <td>1529</td>
      <td>5</td>
      <td>1995</td>
      <td>5.054144</td>
    </tr>
    <tr>
      <th>9024</th>
      <td>Batman v Superman: Dawn of Justice</td>
      <td>7189</td>
      <td>5</td>
      <td>2016</td>
      <td>5.013943</td>
    </tr>
    <tr>
      <th>1260</th>
      <td>Batman &amp; Robin</td>
      <td>1447</td>
      <td>4</td>
      <td>1997</td>
      <td>4.287233</td>
    </tr>
  </tbody>
</table>
</div>



# Collaborative Filtering

Predict using simillar users data, using Surprise and Singular Value Decomposition (SVD) algorithm


```python
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

reader = Reader()
```


```python
ratings = pd.read_csv('dataset/movie/ratings_small.csv')
ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>31</td>
      <td>2.5</td>
      <td>1260759144</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1029</td>
      <td>3.0</td>
      <td>1260759179</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1061</td>
      <td>3.0</td>
      <td>1260759182</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1129</td>
      <td>2.0</td>
      <td>1260759185</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1172</td>
      <td>4.0</td>
      <td>1260759205</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5 , verbose=True) #5 splits
```

    Evaluating RMSE, MAE of algorithm SVD on 5 split(s).
    
                      Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
    RMSE (testset)    0.8997  0.8943  0.8981  0.9034  0.8894  0.8970  0.0048  
    MAE (testset)     0.6924  0.6879  0.6923  0.6949  0.6866  0.6908  0.0031  
    Fit time          5.50    5.39    5.82    5.68    6.19    5.72    0.28    
    Test time         0.14    0.26    0.16    0.15    0.27    0.20    0.06    
    




    {'test_rmse': array([0.89971588, 0.89433794, 0.89810361, 0.90339806, 0.88944449]),
     'test_mae': array([0.69242199, 0.68794254, 0.69231266, 0.6948946 , 0.68659872]),
     'fit_time': (5.496105194091797,
      5.386150598526001,
      5.822059869766235,
      5.684120178222656,
      6.186817169189453),
     'test_time': (0.14284086227416992,
      0.2593069076538086,
      0.15757989883422852,
      0.1506178379058838,
      0.26779866218566895)}



 Root Mean Sqaure Error = 0.8963, which is pretty good


```python
trainset = data.build_full_trainset()
svd.fit(trainset)
```




    <surprise.prediction_algorithms.matrix_factorization.SVD at 0x1a0f52151c8>




```python
ratings[ratings['userId'] == 1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>31</td>
      <td>2.5</td>
      <td>1260759144</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1029</td>
      <td>3.0</td>
      <td>1260759179</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1061</td>
      <td>3.0</td>
      <td>1260759182</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1129</td>
      <td>2.0</td>
      <td>1260759185</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1172</td>
      <td>4.0</td>
      <td>1260759205</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1263</td>
      <td>2.0</td>
      <td>1260759151</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1287</td>
      <td>2.0</td>
      <td>1260759187</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>1293</td>
      <td>2.0</td>
      <td>1260759148</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>1339</td>
      <td>3.5</td>
      <td>1260759125</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>1343</td>
      <td>2.0</td>
      <td>1260759131</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>1371</td>
      <td>2.5</td>
      <td>1260759135</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>1405</td>
      <td>1.0</td>
      <td>1260759203</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>1953</td>
      <td>4.0</td>
      <td>1260759191</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>2105</td>
      <td>4.0</td>
      <td>1260759139</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>2150</td>
      <td>3.0</td>
      <td>1260759194</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>2193</td>
      <td>2.0</td>
      <td>1260759198</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>2294</td>
      <td>2.0</td>
      <td>1260759108</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>2455</td>
      <td>2.5</td>
      <td>1260759113</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>2968</td>
      <td>1.0</td>
      <td>1260759200</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>3671</td>
      <td>3.0</td>
      <td>1260759117</td>
    </tr>
  </tbody>
</table>
</div>




```python
svd.predict(1, 302, 3)
```




    Prediction(uid=1, iid=302, r_ui=3, est=2.575412635819967, details={'was_impossible': False})



Prediction of user ID = 1 on movie ID =302 returns an estimated prediction of 2.7202 based on how the other users have predicted the movie.

# Hybrid Recommender

### Content based + collaborative filter based recommender
#### Input: User ID and the Title of a Movie
#### Output: Similar movies sorted on the basis of expected ratings by that particular user.


```python
def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan
    
def hybrid(userId, title):
    idx = indices[title]
    tmdbId = id_map.loc[title]['id']
    #print(idx)
    movie_id = id_map.loc[title]['movieId']
    
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = sub_md.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)
```


```python
id_map = pd.read_csv('dataset/movie/links_small.csv')[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(sub_md[['title', 'id']], on='id').set_index('title')
#id_map = id_map.set_index('tmdbId')
indices_map = id_map.set_index('id')
```


```python
hybrid(1, 'The Dark Knight')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>year</th>
      <th>id</th>
      <th>est</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3381</th>
      <td>Memento</td>
      <td>4168.0</td>
      <td>8.1</td>
      <td>2000</td>
      <td>77</td>
      <td>3.518882</td>
    </tr>
    <tr>
      <th>7648</th>
      <td>Inception</td>
      <td>14075.0</td>
      <td>8.1</td>
      <td>2010</td>
      <td>27205</td>
      <td>3.253708</td>
    </tr>
    <tr>
      <th>6623</th>
      <td>The Prestige</td>
      <td>4510.0</td>
      <td>8.0</td>
      <td>2006</td>
      <td>1124</td>
      <td>3.230209</td>
    </tr>
    <tr>
      <th>8613</th>
      <td>Interstellar</td>
      <td>11187.0</td>
      <td>8.1</td>
      <td>2014</td>
      <td>157336</td>
      <td>3.094515</td>
    </tr>
    <tr>
      <th>6218</th>
      <td>Batman Begins</td>
      <td>7511.0</td>
      <td>7.5</td>
      <td>2005</td>
      <td>272</td>
      <td>3.030609</td>
    </tr>
    <tr>
      <th>5943</th>
      <td>Thursday</td>
      <td>84.0</td>
      <td>7.0</td>
      <td>1998</td>
      <td>9812</td>
      <td>3.003159</td>
    </tr>
    <tr>
      <th>8031</th>
      <td>The Dark Knight Rises</td>
      <td>9263.0</td>
      <td>7.6</td>
      <td>2012</td>
      <td>49026</td>
      <td>2.864251</td>
    </tr>
    <tr>
      <th>7362</th>
      <td>Gangster's Paradise: Jerusalema</td>
      <td>16.0</td>
      <td>6.8</td>
      <td>2008</td>
      <td>22600</td>
      <td>2.856210</td>
    </tr>
    <tr>
      <th>5098</th>
      <td>The Enforcer</td>
      <td>21.0</td>
      <td>7.4</td>
      <td>1951</td>
      <td>26712</td>
      <td>2.817666</td>
    </tr>
    <tr>
      <th>7561</th>
      <td>Harry Brown</td>
      <td>351.0</td>
      <td>6.7</td>
      <td>2009</td>
      <td>25941</td>
      <td>2.729585</td>
    </tr>
  </tbody>
</table>
</div>




```python
hybrid(500, 'The Dark Knight')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>year</th>
      <th>id</th>
      <th>est</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6623</th>
      <td>The Prestige</td>
      <td>4510.0</td>
      <td>8.0</td>
      <td>2006</td>
      <td>1124</td>
      <td>3.823928</td>
    </tr>
    <tr>
      <th>3381</th>
      <td>Memento</td>
      <td>4168.0</td>
      <td>8.1</td>
      <td>2000</td>
      <td>77</td>
      <td>3.601578</td>
    </tr>
    <tr>
      <th>8613</th>
      <td>Interstellar</td>
      <td>11187.0</td>
      <td>8.1</td>
      <td>2014</td>
      <td>157336</td>
      <td>3.519702</td>
    </tr>
    <tr>
      <th>7648</th>
      <td>Inception</td>
      <td>14075.0</td>
      <td>8.1</td>
      <td>2010</td>
      <td>27205</td>
      <td>3.330051</td>
    </tr>
    <tr>
      <th>5943</th>
      <td>Thursday</td>
      <td>84.0</td>
      <td>7.0</td>
      <td>1998</td>
      <td>9812</td>
      <td>3.322717</td>
    </tr>
    <tr>
      <th>7561</th>
      <td>Harry Brown</td>
      <td>351.0</td>
      <td>6.7</td>
      <td>2009</td>
      <td>25941</td>
      <td>3.210024</td>
    </tr>
    <tr>
      <th>2448</th>
      <td>Nighthawks</td>
      <td>87.0</td>
      <td>6.4</td>
      <td>1981</td>
      <td>21610</td>
      <td>3.034214</td>
    </tr>
    <tr>
      <th>2131</th>
      <td>Superman</td>
      <td>1042.0</td>
      <td>6.9</td>
      <td>1978</td>
      <td>1924</td>
      <td>2.998992</td>
    </tr>
    <tr>
      <th>2085</th>
      <td>Following</td>
      <td>363.0</td>
      <td>7.2</td>
      <td>1998</td>
      <td>11660</td>
      <td>2.974581</td>
    </tr>
    <tr>
      <th>8031</th>
      <td>The Dark Knight Rises</td>
      <td>9263.0</td>
      <td>7.6</td>
      <td>2012</td>
      <td>49026</td>
      <td>2.967465</td>
    </tr>
  </tbody>
</table>
</div>



As observed, different recommendation lists were offered to different user watching the same movie, indicating that the recommendation more tailored and personalized for each user
