# Readme

Recommendation system tutorial on a movies db.

```{.python .input  n=1}
# import libs

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
```

```{.python .input  n=2}
# import movies ratings data

ratings = pd.read_csv('./ml-latest-small/ratings.csv', sep=',')
ratings.head()
```

```{.json .output n=2}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>userId</th>\n      <th>movieId</th>\n      <th>rating</th>\n      <th>timestamp</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>1</td>\n      <td>1</td>\n      <td>4.0</td>\n      <td>964982703</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>1</td>\n      <td>3</td>\n      <td>4.0</td>\n      <td>964981247</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>1</td>\n      <td>6</td>\n      <td>4.0</td>\n      <td>964982224</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>1</td>\n      <td>47</td>\n      <td>5.0</td>\n      <td>964983815</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>1</td>\n      <td>50</td>\n      <td>5.0</td>\n      <td>964982931</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "   userId  movieId  rating  timestamp\n0       1        1     4.0  964982703\n1       1        3     4.0  964981247\n2       1        6     4.0  964982224\n3       1       47     5.0  964983815\n4       1       50     5.0  964982931"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=3}
# import movies names data

movies = pd.read_csv('./ml-latest-small/movies.csv', sep=',')
movies.head()
```

```{.json .output n=3}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>movieId</th>\n      <th>title</th>\n      <th>genres</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>1</td>\n      <td>Toy Story (1995)</td>\n      <td>Adventure|Animation|Children|Comedy|Fantasy</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>2</td>\n      <td>Jumanji (1995)</td>\n      <td>Adventure|Children|Fantasy</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>3</td>\n      <td>Grumpier Old Men (1995)</td>\n      <td>Comedy|Romance</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>4</td>\n      <td>Waiting to Exhale (1995)</td>\n      <td>Comedy|Drama|Romance</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>5</td>\n      <td>Father of the Bride Part II (1995)</td>\n      <td>Comedy</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "   movieId                               title  \\\n0        1                    Toy Story (1995)   \n1        2                      Jumanji (1995)   \n2        3             Grumpier Old Men (1995)   \n3        4            Waiting to Exhale (1995)   \n4        5  Father of the Bride Part II (1995)   \n\n                                        genres  \n0  Adventure|Animation|Children|Comedy|Fantasy  \n1                   Adventure|Children|Fantasy  \n2                               Comedy|Romance  \n3                         Comedy|Drama|Romance  \n4                                       Comedy  "
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=4}
# merge names & ratings

df = pd.merge(ratings, movies, on='movieId')
df.head()
```

```{.json .output n=4}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>userId</th>\n      <th>movieId</th>\n      <th>rating</th>\n      <th>timestamp</th>\n      <th>title</th>\n      <th>genres</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>1</td>\n      <td>1</td>\n      <td>4.0</td>\n      <td>964982703</td>\n      <td>Toy Story (1995)</td>\n      <td>Adventure|Animation|Children|Comedy|Fantasy</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>5</td>\n      <td>1</td>\n      <td>4.0</td>\n      <td>847434962</td>\n      <td>Toy Story (1995)</td>\n      <td>Adventure|Animation|Children|Comedy|Fantasy</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>7</td>\n      <td>1</td>\n      <td>4.5</td>\n      <td>1106635946</td>\n      <td>Toy Story (1995)</td>\n      <td>Adventure|Animation|Children|Comedy|Fantasy</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>15</td>\n      <td>1</td>\n      <td>2.5</td>\n      <td>1510577970</td>\n      <td>Toy Story (1995)</td>\n      <td>Adventure|Animation|Children|Comedy|Fantasy</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>17</td>\n      <td>1</td>\n      <td>4.5</td>\n      <td>1305696483</td>\n      <td>Toy Story (1995)</td>\n      <td>Adventure|Animation|Children|Comedy|Fantasy</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "   userId  movieId  rating   timestamp             title  \\\n0       1        1     4.0   964982703  Toy Story (1995)   \n1       5        1     4.0   847434962  Toy Story (1995)   \n2       7        1     4.5  1106635946  Toy Story (1995)   \n3      15        1     2.5  1510577970  Toy Story (1995)   \n4      17        1     4.5  1305696483  Toy Story (1995)   \n\n                                        genres  \n0  Adventure|Animation|Children|Comedy|Fantasy  \n1  Adventure|Animation|Children|Comedy|Fantasy  \n2  Adventure|Animation|Children|Comedy|Fantasy  \n3  Adventure|Animation|Children|Comedy|Fantasy  \n4  Adventure|Animation|Children|Comedy|Fantasy  "
  },
  "execution_count": 4,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=5}
df.rating.describe()
```

```{.json .output n=5}
[
 {
  "data": {
   "text/plain": "count    100836.000000\nmean          3.501557\nstd           1.042529\nmin           0.500000\n25%           3.000000\n50%           3.500000\n75%           4.000000\nmax           5.000000\nName: rating, dtype: float64"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=6}
# computing mean rating for every movie

rated = pd.DataFrame(df.groupby('title')['rating'].mean())
rated.head()
```

```{.json .output n=6}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>rating</th>\n    </tr>\n    <tr>\n      <th>title</th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>'71 (2014)</th>\n      <td>4.0</td>\n    </tr>\n    <tr>\n      <th>'Hellboy': The Seeds of Creation (2004)</th>\n      <td>4.0</td>\n    </tr>\n    <tr>\n      <th>'Round Midnight (1986)</th>\n      <td>3.5</td>\n    </tr>\n    <tr>\n      <th>'Salem's Lot (2004)</th>\n      <td>5.0</td>\n    </tr>\n    <tr>\n      <th>'Til There Was You (1997)</th>\n      <td>4.0</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "                                         rating\ntitle                                          \n'71 (2014)                                  4.0\n'Hellboy': The Seeds of Creation (2004)     4.0\n'Round Midnight (1986)                      3.5\n'Salem's Lot (2004)                         5.0\n'Til There Was You (1997)                   4.0"
  },
  "execution_count": 6,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=7}
# computing number of ratings for every movie

rated['number_of_ratings'] = df.groupby('title')['rating'].count()
rated.head()
```

```{.json .output n=7}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>rating</th>\n      <th>number_of_ratings</th>\n    </tr>\n    <tr>\n      <th>title</th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>'71 (2014)</th>\n      <td>4.0</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>'Hellboy': The Seeds of Creation (2004)</th>\n      <td>4.0</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>'Round Midnight (1986)</th>\n      <td>3.5</td>\n      <td>2</td>\n    </tr>\n    <tr>\n      <th>'Salem's Lot (2004)</th>\n      <td>5.0</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <th>'Til There Was You (1997)</th>\n      <td>4.0</td>\n      <td>2</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "                                         rating  number_of_ratings\ntitle                                                             \n'71 (2014)                                  4.0                  1\n'Hellboy': The Seeds of Creation (2004)     4.0                  1\n'Round Midnight (1986)                      3.5                  2\n'Salem's Lot (2004)                         5.0                  1\n'Til There Was You (1997)                   4.0                  2"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=8}
# histogram of average ratings

import matplotlib.pyplot as plt
%matplotlib inline
rated['rating'].hist(bins=50)
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "<matplotlib.axes._subplots.AxesSubplot at 0x7fb329872860>"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<Figure size 432x288 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

```{.python .input  n=9}
# histogram of number of ratings

rated['number_of_ratings'].hist(bins=50)
```

```{.json .output n=9}
[
 {
  "data": {
   "text/plain": "<matplotlib.axes._subplots.AxesSubplot at 0x7fb328fd58d0>"
  },
  "execution_count": 9,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<Figure size 432x288 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

```{.python .input  n=10}
# numbers vs. means

import seaborn as sns
sns.jointplot(x='rating', y='number_of_ratings', data=rated)
```

```{.json .output n=10}
[
 {
  "data": {
   "text/plain": "<seaborn.axisgrid.JointGrid at 0x7fb328f2f898>"
  },
  "execution_count": 10,
  "metadata": {},
  "output_type": "execute_result"
 },
 {
  "data": {
   "image/png": "iVBORw0KGgoAAAANSUhEUgAAAasAAAGoCAYAAAD4hcrDAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjMsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+AADFEAAAgAElEQVR4nO3df5RcZZkn8O/TlQqpjpAOY2STgkyiwyTIBNISAU/27BocCYpAD4jBX+POcmT2rI4GmZ5t1JXExaGdHoUzc2bcwUHFASGBYAuGMTAkHseMQTt0hxhJFuRHSCWHBJOCkC6S6u5n/6h7O7er7s+qe+veqvv9nNOnq27dqnoL0vep932f93lFVUFERJRkHXE3gIiIyAuDFRERJR6DFRERJR6DFRERJR6DFRERJd60uBsQAqYzElE7kbgbkETsWRERUeIxWBERUeK1wzAgESXQD57ca3v8YxfNb3JLqB0wWBG1KQYLaiccBiQiosRjsCIiosRjsCIiosRjsCIiosRjggVRRJwSHAAmORAFxZ4VERElHoMVERElHoMVERElHoMVERElHoMVERElHoMVERElHoMVERElHoMVERElHoMVERElHoMVERElHoMVERElHoMVERElHoMVERElHoMVERElHoMVERElHoMVERElHoMVERElHoMVERElHoMVERElHoMVEREl3rS4G0AUlR88udfxsY9dNL+JLSGiRrFnRUREicdgRUREicdgRUREicc5K4qU07wR54yIKAgGK6IEiTu4x/3+RE44DEhERInHnhURtR0uW2g/DFZEKeN2ISdKKg4DEhFR4rFnRUSemHhBcWOwIvIh7jkQDt0lE4N483AYkIiIEo89KyKqG3t81CwMVkQx4EWeKBgGqzbHMXUiagcMVtQykhp42UsKR9xJLJRsDFZE1FQM7lQPBquI8dtia+GFlCiZmLpORESJl+qeVVLnQCh67EERtZZUBytKnnqCCANP++MXS2KwooYxwBBR1BisyDcGGGpnTIZKNgarAJp1sW7G+zDwUDtgrz49RFXjbkNDROQnAN4adzvq8FYAr8bdiBjwc6cLP3dwr6rqZWE2ph20fLBqVSIypKrL4m5Hs/Fzpws/N4WF66yIiCjxGKyIiCjxGKzic2fcDYgJP3e68HNTKDhnRUREiceeFRERJR6D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\n",
   "text/plain": "<Figure size 432x432 with 3 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

```{.python .input  n=11}
# rating by user for every movie matrix

movie_matrix = df.pivot_table(index='userId', columns='title', values='rating')
movie_matrix.head()
```

```{.json .output n=11}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th>title</th>\n      <th>'71 (2014)</th>\n      <th>'Hellboy': The Seeds of Creation (2004)</th>\n      <th>'Round Midnight (1986)</th>\n      <th>'Salem's Lot (2004)</th>\n      <th>'Til There Was You (1997)</th>\n      <th>'Tis the Season for Love (2015)</th>\n      <th>'burbs, The (1989)</th>\n      <th>'night Mother (1986)</th>\n      <th>(500) Days of Summer (2009)</th>\n      <th>*batteries not included (1987)</th>\n      <th>...</th>\n      <th>Zulu (2013)</th>\n      <th>[REC] (2007)</th>\n      <th>[REC]\u00b2 (2009)</th>\n      <th>[REC]\u00b3 3 G\u00e9nesis (2012)</th>\n      <th>anohana: The Flower We Saw That Day - The Movie (2013)</th>\n      <th>eXistenZ (1999)</th>\n      <th>xXx (2002)</th>\n      <th>xXx: State of the Union (2005)</th>\n      <th>\u00a1Three Amigos! (1986)</th>\n      <th>\u00c0 nous la libert\u00e9 (Freedom for Us) (1931)</th>\n    </tr>\n    <tr>\n      <th>userId</th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>1</th>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>4.0</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>5</th>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>...</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n    </tr>\n  </tbody>\n</table>\n<p>5 rows \u00d7 9719 columns</p>\n</div>",
   "text/plain": "title   '71 (2014)  'Hellboy': The Seeds of Creation (2004)  \\\nuserId                                                        \n1              NaN                                      NaN   \n2              NaN                                      NaN   \n3              NaN                                      NaN   \n4              NaN                                      NaN   \n5              NaN                                      NaN   \n\ntitle   'Round Midnight (1986)  'Salem's Lot (2004)  \\\nuserId                                                \n1                          NaN                  NaN   \n2                          NaN                  NaN   \n3                          NaN                  NaN   \n4                          NaN                  NaN   \n5                          NaN                  NaN   \n\ntitle   'Til There Was You (1997)  'Tis the Season for Love (2015)  \\\nuserId                                                               \n1                             NaN                              NaN   \n2                             NaN                              NaN   \n3                             NaN                              NaN   \n4                             NaN                              NaN   \n5                             NaN                              NaN   \n\ntitle   'burbs, The (1989)  'night Mother (1986)  (500) Days of Summer (2009)  \\\nuserId                                                                          \n1                      NaN                   NaN                          NaN   \n2                      NaN                   NaN                          NaN   \n3                      NaN                   NaN                          NaN   \n4                      NaN                   NaN                          NaN   \n5                      NaN                   NaN                          NaN   \n\ntitle   *batteries not included (1987)  ...  Zulu (2013)  [REC] (2007)  \\\nuserId                                  ...                              \n1                                  NaN  ...          NaN           NaN   \n2                                  NaN  ...          NaN           NaN   \n3                                  NaN  ...          NaN           NaN   \n4                                  NaN  ...          NaN           NaN   \n5                                  NaN  ...          NaN           NaN   \n\ntitle   [REC]\u00b2 (2009)  [REC]\u00b3 3 G\u00e9nesis (2012)  \\\nuserId                                           \n1                 NaN                      NaN   \n2                 NaN                      NaN   \n3                 NaN                      NaN   \n4                 NaN                      NaN   \n5                 NaN                      NaN   \n\ntitle   anohana: The Flower We Saw That Day - The Movie (2013)  \\\nuserId                                                           \n1                                                     NaN        \n2                                                     NaN        \n3                                                     NaN        \n4                                                     NaN        \n5                                                     NaN        \n\ntitle   eXistenZ (1999)  xXx (2002)  xXx: State of the Union (2005)  \\\nuserId                                                                \n1                   NaN         NaN                             NaN   \n2                   NaN         NaN                             NaN   \n3                   NaN         NaN                             NaN   \n4                   NaN         NaN                             NaN   \n5                   NaN         NaN                             NaN   \n\ntitle   \u00a1Three Amigos! (1986)  \u00c0 nous la libert\u00e9 (Freedom for Us) (1931)  \nuserId                                                                    \n1                         4.0                                        NaN  \n2                         NaN                                        NaN  \n3                         NaN                                        NaN  \n4                         NaN                                        NaN  \n5                         NaN                                        NaN  \n\n[5 rows x 9719 columns]"
  },
  "execution_count": 11,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=12}
# sorted by number of ratings

rated.sort_values('number_of_ratings', ascending=False).head(10)
```

```{.json .output n=12}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>rating</th>\n      <th>number_of_ratings</th>\n    </tr>\n    <tr>\n      <th>title</th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>Forrest Gump (1994)</th>\n      <td>4.164134</td>\n      <td>329</td>\n    </tr>\n    <tr>\n      <th>Shawshank Redemption, The (1994)</th>\n      <td>4.429022</td>\n      <td>317</td>\n    </tr>\n    <tr>\n      <th>Pulp Fiction (1994)</th>\n      <td>4.197068</td>\n      <td>307</td>\n    </tr>\n    <tr>\n      <th>Silence of the Lambs, The (1991)</th>\n      <td>4.161290</td>\n      <td>279</td>\n    </tr>\n    <tr>\n      <th>Matrix, The (1999)</th>\n      <td>4.192446</td>\n      <td>278</td>\n    </tr>\n    <tr>\n      <th>Star Wars: Episode IV - A New Hope (1977)</th>\n      <td>4.231076</td>\n      <td>251</td>\n    </tr>\n    <tr>\n      <th>Jurassic Park (1993)</th>\n      <td>3.750000</td>\n      <td>238</td>\n    </tr>\n    <tr>\n      <th>Braveheart (1995)</th>\n      <td>4.031646</td>\n      <td>237</td>\n    </tr>\n    <tr>\n      <th>Terminator 2: Judgment Day (1991)</th>\n      <td>3.970982</td>\n      <td>224</td>\n    </tr>\n    <tr>\n      <th>Schindler's List (1993)</th>\n      <td>4.225000</td>\n      <td>220</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "                                             rating  number_of_ratings\ntitle                                                                 \nForrest Gump (1994)                        4.164134                329\nShawshank Redemption, The (1994)           4.429022                317\nPulp Fiction (1994)                        4.197068                307\nSilence of the Lambs, The (1991)           4.161290                279\nMatrix, The (1999)                         4.192446                278\nStar Wars: Episode IV - A New Hope (1977)  4.231076                251\nJurassic Park (1993)                       3.750000                238\nBraveheart (1995)                          4.031646                237\nTerminator 2: Judgment Day (1991)          3.970982                224\nSchindler's List (1993)                    4.225000                220"
  },
  "execution_count": 12,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=15}
# users rating for the movie "Forrest Gump"

forrest_gump_ratings = movie_matrix['Forrest Gump (1994)']
forrest_gump_ratings.head()
```

```{.json .output n=15}
[
 {
  "data": {
   "text/plain": "userId\n1    4.0\n2    NaN\n3    NaN\n4    NaN\n5    NaN\nName: Forrest Gump (1994), dtype: float64"
  },
  "execution_count": 15,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=20}
# pointing movies similar to "Forrest Gump"

movies_like_forrest_gump = movie_matrix.corrwith(forrest_gump_ratings)
corr_forrest_gump = pd.DataFrame(movies_like_forrest_gump, columns=['Correlation'])
corr_forrest_gump.dropna(inplace=True)
corr_forrest_gump.head()
```

```{.json .output n=20}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Correlation</th>\n    </tr>\n    <tr>\n      <th>title</th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>'burbs, The (1989)</th>\n      <td>0.197712</td>\n    </tr>\n    <tr>\n      <th>(500) Days of Summer (2009)</th>\n      <td>0.234095</td>\n    </tr>\n    <tr>\n      <th>*batteries not included (1987)</th>\n      <td>0.892710</td>\n    </tr>\n    <tr>\n      <th>...And Justice for All (1979)</th>\n      <td>0.928571</td>\n    </tr>\n    <tr>\n      <th>10 Cent Pistol (2015)</th>\n      <td>-1.000000</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "                                Correlation\ntitle                                      \n'burbs, The (1989)                 0.197712\n(500) Days of Summer (2009)        0.234095\n*batteries not included (1987)     0.892710\n...And Justice for All (1979)      0.928571\n10 Cent Pistol (2015)             -1.000000"
  },
  "execution_count": 20,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=22}
# sorted

corr_forrest_gump.sort_values('Correlation', ascending=False).head()
```

```{.json .output n=22}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Correlation</th>\n    </tr>\n    <tr>\n      <th>title</th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>Lost &amp; Found (1999)</th>\n      <td>1.0</td>\n    </tr>\n    <tr>\n      <th>Cercle Rouge, Le (Red Circle, The) (1970)</th>\n      <td>1.0</td>\n    </tr>\n    <tr>\n      <th>Play Time (a.k.a. Playtime) (1967)</th>\n      <td>1.0</td>\n    </tr>\n    <tr>\n      <th>Killers (2010)</th>\n      <td>1.0</td>\n    </tr>\n    <tr>\n      <th>Playing God (1997)</th>\n      <td>1.0</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "                                           Correlation\ntitle                                                 \nLost & Found (1999)                                1.0\nCercle Rouge, Le (Red Circle, The) (1970)          1.0\nPlay Time (a.k.a. Playtime) (1967)                 1.0\nKillers (2010)                                     1.0\nPlaying God (1997)                                 1.0"
  },
  "execution_count": 22,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=23}
# joining the number of ratings

corr_forrest_gump = corr_forrest_gump.join(rated['number_of_ratings'])
corr_forrest_gump.head()
```

```{.json .output n=23}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Correlation</th>\n      <th>number_of_ratings</th>\n    </tr>\n    <tr>\n      <th>title</th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>'burbs, The (1989)</th>\n      <td>0.197712</td>\n      <td>17</td>\n    </tr>\n    <tr>\n      <th>(500) Days of Summer (2009)</th>\n      <td>0.234095</td>\n      <td>42</td>\n    </tr>\n    <tr>\n      <th>*batteries not included (1987)</th>\n      <td>0.892710</td>\n      <td>7</td>\n    </tr>\n    <tr>\n      <th>...And Justice for All (1979)</th>\n      <td>0.928571</td>\n      <td>3</td>\n    </tr>\n    <tr>\n      <th>10 Cent Pistol (2015)</th>\n      <td>-1.000000</td>\n      <td>2</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "                                Correlation  number_of_ratings\ntitle                                                         \n'burbs, The (1989)                 0.197712                 17\n(500) Days of Summer (2009)        0.234095                 42\n*batteries not included (1987)     0.892710                  7\n...And Justice for All (1979)      0.928571                  3\n10 Cent Pistol (2015)             -1.000000                  2"
  },
  "execution_count": 23,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=24}
# sorted by correlation with more than 50 ratings registered

corr_forrest_gump[corr_forrest_gump['number_of_ratings']>50].sort_values('Correlation', ascending=False).head()
```

```{.json .output n=24}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Correlation</th>\n      <th>number_of_ratings</th>\n    </tr>\n    <tr>\n      <th>title</th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>Forrest Gump (1994)</th>\n      <td>1.000000</td>\n      <td>329</td>\n    </tr>\n    <tr>\n      <th>Mr. Holland's Opus (1995)</th>\n      <td>0.652144</td>\n      <td>80</td>\n    </tr>\n    <tr>\n      <th>Pocahontas (1995)</th>\n      <td>0.550118</td>\n      <td>68</td>\n    </tr>\n    <tr>\n      <th>Grumpier Old Men (1995)</th>\n      <td>0.534682</td>\n      <td>52</td>\n    </tr>\n    <tr>\n      <th>Caddyshack (1980)</th>\n      <td>0.520328</td>\n      <td>52</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "                           Correlation  number_of_ratings\ntitle                                                    \nForrest Gump (1994)           1.000000                329\nMr. Holland's Opus (1995)     0.652144                 80\nPocahontas (1995)             0.550118                 68\nGrumpier Old Men (1995)       0.534682                 52\nCaddyshack (1980)             0.520328                 52"
  },
  "execution_count": 24,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```
