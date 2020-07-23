Recommendation system tutorial on a sample movies db.

#### libraries
* pandas
* numpy
* warnings
* matplotlib
* seaborn

#### code

##### import libs
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
```

##### import data
```python
ratings = pd.read_csv('./ml-latest-small/ratings.csv', sep=',')
movies = pd.read_csv('./ml-latest-small/movies.csv', sep=',')
```

##### processing data
```python
# create dataframe
df = pd.merge(ratings, movies, on='movieId')

# computes mean rating for every movie
rated = pd.DataFrame(df.groupby('title')['rating'].mean())

# computes number of ratings for every movie
rated['number_of_ratings'] = df.groupby('title')['rating'].count()
```

##### visualizing data
```python
# histogram of average ratings
%matplotlib inline
rated['rating'].hist(bins=50)
```
![png](img/output_8_1.png)

```python
# histogram of number of ratings
rated['number_of_ratings'].hist(bins=50)
```
![png](img/output_9_1.png)

```python
# numbers vs. means
sns.jointplot(x='rating', y='number_of_ratings', data=rated)
```
![png](img/output_10_1.png)

##### exploiting data
```python
# rating by user for every movie matrix
movie_matrix = df.pivot_table(index='userId', columns='title', values='rating')

# users rating for the movie "Forrest Gump"
forrest_gump_ratings = movie_matrix['Forrest Gump (1994)']

# pointing movies similar to "Forrest Gump"
movies_like_forrest_gump = movie_matrix.corrwith(forrest_gump_ratings)
corr_forrest_gump = pd.DataFrame(movies_like_forrest_gump, columns=['Correlation'])
corr_forrest_gump.dropna(inplace=True)

# joining the number of ratings
corr_forrest_gump = corr_forrest_gump.join(rated['number_of_ratings'])

# sorted by correlation with more than 50 ratings registered
corr_forrest_gump[corr_forrest_gump['number_of_ratings']>50].sort_values('Correlation', ascending=False).head()
```
![png](img/output_11_1.png)