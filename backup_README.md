# Readme

Recommendation system tutorial on a movies db.


```python
# import libs

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
```


```python
# import movies ratings data

ratings = pd.read_csv('./ml-latest-small/ratings.csv', sep=',')
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
      <td>1</td>
      <td>4.0</td>
      <td>964982703</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>4.0</td>
      <td>964981247</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>6</td>
      <td>4.0</td>
      <td>964982224</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>47</td>
      <td>5.0</td>
      <td>964983815</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>50</td>
      <td>5.0</td>
      <td>964982931</td>
    </tr>
  </tbody>
</table>
</div>




```python
# import movies names data

movies = pd.read_csv('./ml-latest-small/movies.csv', sep=',')
movies.head()
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
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>




```python
# merge names & ratings

df = pd.merge(ratings, movies, on='movieId')
df.head()
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
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
      <td>964982703</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>1</td>
      <td>4.0</td>
      <td>847434962</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>1</td>
      <td>4.5</td>
      <td>1106635946</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>1</td>
      <td>2.5</td>
      <td>1510577970</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17</td>
      <td>1</td>
      <td>4.5</td>
      <td>1305696483</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.rating.describe()
```




    count    100836.000000
    mean          3.501557
    std           1.042529
    min           0.500000
    25%           3.000000
    50%           3.500000
    75%           4.000000
    max           5.000000
    Name: rating, dtype: float64




```python
# computing mean rating for every movie

rated = pd.DataFrame(df.groupby('title')['rating'].mean())
rated.head()
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
      <th>rating</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'71 (2014)</th>
      <td>4.0</td>
    </tr>
    <tr>
      <th>'Hellboy': The Seeds of Creation (2004)</th>
      <td>4.0</td>
    </tr>
    <tr>
      <th>'Round Midnight (1986)</th>
      <td>3.5</td>
    </tr>
    <tr>
      <th>'Salem's Lot (2004)</th>
      <td>5.0</td>
    </tr>
    <tr>
      <th>'Til There Was You (1997)</th>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# computing number of ratings for every movie

rated['number_of_ratings'] = df.groupby('title')['rating'].count()
rated.head()
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
      <th>rating</th>
      <th>number_of_ratings</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'71 (2014)</th>
      <td>4.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>'Hellboy': The Seeds of Creation (2004)</th>
      <td>4.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>'Round Midnight (1986)</th>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>'Salem's Lot (2004)</th>
      <td>5.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>'Til There Was You (1997)</th>
      <td>4.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# histogram of average ratings

import matplotlib.pyplot as plt
%matplotlib inline
rated['rating'].hist(bins=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb329872860>




![png](output_8_1.png)



```python
# histogram of number of ratings

rated['number_of_ratings'].hist(bins=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb328fd58d0>




![png](output_9_1.png)



```python
# numbers vs. means

import seaborn as sns
sns.jointplot(x='rating', y='number_of_ratings', data=rated)
```




    <seaborn.axisgrid.JointGrid at 0x7fb328f2f898>




![png](output_10_1.png)



```python
# rating by user for every movie matrix

movie_matrix = df.pivot_table(index='userId', columns='title', values='rating')
movie_matrix.head()
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
      <th>title</th>
      <th>'71 (2014)</th>
      <th>'Hellboy': The Seeds of Creation (2004)</th>
      <th>'Round Midnight (1986)</th>
      <th>'Salem's Lot (2004)</th>
      <th>'Til There Was You (1997)</th>
      <th>'Tis the Season for Love (2015)</th>
      <th>'burbs, The (1989)</th>
      <th>'night Mother (1986)</th>
      <th>(500) Days of Summer (2009)</th>
      <th>*batteries not included (1987)</th>
      <th>...</th>
      <th>Zulu (2013)</th>
      <th>[REC] (2007)</th>
      <th>[REC]² (2009)</th>
      <th>[REC]³ 3 Génesis (2012)</th>
      <th>anohana: The Flower We Saw That Day - The Movie (2013)</th>
      <th>eXistenZ (1999)</th>
      <th>xXx (2002)</th>
      <th>xXx: State of the Union (2005)</th>
      <th>¡Three Amigos! (1986)</th>
      <th>À nous la liberté (Freedom for Us) (1931)</th>
    </tr>
    <tr>
      <th>userId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 9719 columns</p>
</div>




```python
# sorted by number of ratings

rated.sort_values('number_of_ratings', ascending=False).head(10)
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
      <th>rating</th>
      <th>number_of_ratings</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Forrest Gump (1994)</th>
      <td>4.164134</td>
      <td>329</td>
    </tr>
    <tr>
      <th>Shawshank Redemption, The (1994)</th>
      <td>4.429022</td>
      <td>317</td>
    </tr>
    <tr>
      <th>Pulp Fiction (1994)</th>
      <td>4.197068</td>
      <td>307</td>
    </tr>
    <tr>
      <th>Silence of the Lambs, The (1991)</th>
      <td>4.161290</td>
      <td>279</td>
    </tr>
    <tr>
      <th>Matrix, The (1999)</th>
      <td>4.192446</td>
      <td>278</td>
    </tr>
    <tr>
      <th>Star Wars: Episode IV - A New Hope (1977)</th>
      <td>4.231076</td>
      <td>251</td>
    </tr>
    <tr>
      <th>Jurassic Park (1993)</th>
      <td>3.750000</td>
      <td>238</td>
    </tr>
    <tr>
      <th>Braveheart (1995)</th>
      <td>4.031646</td>
      <td>237</td>
    </tr>
    <tr>
      <th>Terminator 2: Judgment Day (1991)</th>
      <td>3.970982</td>
      <td>224</td>
    </tr>
    <tr>
      <th>Schindler's List (1993)</th>
      <td>4.225000</td>
      <td>220</td>
    </tr>
  </tbody>
</table>
</div>




```python
# users rating for the movie "Forrest Gump"

forrest_gump_ratings = movie_matrix['Forrest Gump (1994)']
forrest_gump_ratings.head()
```




    userId
    1    4.0
    2    NaN
    3    NaN
    4    NaN
    5    NaN
    Name: Forrest Gump (1994), dtype: float64




```python
# pointing movies similar to "Forrest Gump"

movies_like_forrest_gump = movie_matrix.corrwith(forrest_gump_ratings)
corr_forrest_gump = pd.DataFrame(movies_like_forrest_gump, columns=['Correlation'])
corr_forrest_gump.dropna(inplace=True)
corr_forrest_gump.head()
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
      <th>Correlation</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'burbs, The (1989)</th>
      <td>0.197712</td>
    </tr>
    <tr>
      <th>(500) Days of Summer (2009)</th>
      <td>0.234095</td>
    </tr>
    <tr>
      <th>*batteries not included (1987)</th>
      <td>0.892710</td>
    </tr>
    <tr>
      <th>...And Justice for All (1979)</th>
      <td>0.928571</td>
    </tr>
    <tr>
      <th>10 Cent Pistol (2015)</th>
      <td>-1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# sorted

corr_forrest_gump.sort_values('Correlation', ascending=False).head()
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
      <th>Correlation</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Lost &amp; Found (1999)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Cercle Rouge, Le (Red Circle, The) (1970)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Play Time (a.k.a. Playtime) (1967)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Killers (2010)</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Playing God (1997)</th>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# joining the number of ratings

corr_forrest_gump = corr_forrest_gump.join(rated['number_of_ratings'])
corr_forrest_gump.head()
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
      <th>Correlation</th>
      <th>number_of_ratings</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'burbs, The (1989)</th>
      <td>0.197712</td>
      <td>17</td>
    </tr>
    <tr>
      <th>(500) Days of Summer (2009)</th>
      <td>0.234095</td>
      <td>42</td>
    </tr>
    <tr>
      <th>*batteries not included (1987)</th>
      <td>0.892710</td>
      <td>7</td>
    </tr>
    <tr>
      <th>...And Justice for All (1979)</th>
      <td>0.928571</td>
      <td>3</td>
    </tr>
    <tr>
      <th>10 Cent Pistol (2015)</th>
      <td>-1.000000</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# sorted by correlation with more than 50 ratings registered

corr_forrest_gump[corr_forrest_gump['number_of_ratings']>50].sort_values('Correlation', ascending=False).head()
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
      <th>Correlation</th>
      <th>number_of_ratings</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Forrest Gump (1994)</th>
      <td>1.000000</td>
      <td>329</td>
    </tr>
    <tr>
      <th>Mr. Holland's Opus (1995)</th>
      <td>0.652144</td>
      <td>80</td>
    </tr>
    <tr>
      <th>Pocahontas (1995)</th>
      <td>0.550118</td>
      <td>68</td>
    </tr>
    <tr>
      <th>Grumpier Old Men (1995)</th>
      <td>0.534682</td>
      <td>52</td>
    </tr>
    <tr>
      <th>Caddyshack (1980)</th>
      <td>0.520328</td>
      <td>52</td>
    </tr>
  </tbody>
</table>
</div>


