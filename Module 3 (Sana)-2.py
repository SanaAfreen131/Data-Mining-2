#!/usr/bin/env python
# coding: utf-8

# In[186]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from collections import defaultdict


# # Data Loading & Understanding

# In[187]:


df_artists = pd.read_csv('artists.csv')
df_songs = pd.read_csv('tracks.csv')


# In[188]:


df_artists.head()


# In[189]:


df_songs.head()


# In[190]:


df_artists.shape


# In[191]:


df_songs.shape


# In[192]:


df_artists.isnull().sum()


# In[193]:


df_songs.isnull().sum()


# In[194]:


print(df_artists[df_artists.isnull().any(axis=1)])


# In[195]:


df_artists_cleaned = df_artists.dropna()


# In[196]:


df_artists_cleaned.shape


# In[197]:


unique_artists_id = df_artists['id'].nunique()
print(unique_artists_id)


# In[198]:


unique_song_id = df_songs['id'].nunique()
print(unique_song_id)


# In[199]:


# Drop the 'Genre' column
df_artists = df_artists_cleaned.drop(columns=['genres', 'id'])
df_artists.head()


# In[200]:


# Rename specific columns using rename() method
df_artists = df_artists.rename(columns={'name': 'artist_name', 'popularity': 'artist_popularity', 'followers': 'artist_followers'})

# Display the DataFrame with renamed columns
print("\nArtists DataFrame with renamed columns:")
df_artists.head()


# In[201]:


# Split artists into separate rows from song dataset
df_songs_split = df_songs.assign(artists=df_songs['artists'].str.split(';')).explode('artists')

# Reset index if needed
df_songs_split = df_songs_split.reset_index(drop=True)

# View the resulting dataframe
df_songs_split.head(50)


# In[202]:


df_songs = df_songs_split
df_songs.head()


# In[220]:


# Merge DataFrames on 'artist' and 'artist_name'
merged_df = pd.merge(df_songs, df_artists, left_on='artists', right_on='artist_name', how='left')
# Display the merged DataFrame
merged_df.head(50)


# In[203]:


merged_df.isnull().sum()


# In[204]:


# Show rows with missing values
missing_rows = merged_df[merged_df.isnull().any(axis=1)]
missing_rows.head(30)


# In[205]:


merged_df.shape


# In[206]:


# Remove rows with missing values
cleaned_df = merged_df.dropna()

# Display cleaned DataFrame
cleaned_df.head(50)


# In[207]:


cleaned_df.shape


# In[208]:


cleaned_df.isnull().sum()


# In[210]:


# Display data types of columns
df.dtypes


# In[211]:


# Select numerical columns
numerical_df = df.select_dtypes(include=['int64', 'float64'])



# In[212]:


# Calculate correlation matrix for numerical columns
corr_matrix = numerical_df.corr()


# In[213]:


print("Correlation Matrix for Numerical Values:")
print(corr_matrix)


# In[214]:


# Plot heatmap
plt.figure(figsize=(25, 20))  # Adjust the figure size as needed
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap for Numerical Values')
plt.show()


# In[215]:


# Create a mask to display only the upper triangular part
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Plot heatmap of the upper triangular correlation matrix
plt.figure(figsize=(25, 20))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Upper Triangular Correlation Heatmap')
plt.show()


# In[216]:


# Select numerical columns for outlier detection
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Calculate the number of subplots needed
num_cols = len(numerical_cols)
num_rows = (num_cols // 3) + (1 if num_cols % 3 > 0 else 0)  # Adjust rows based on the number of columns

# Plot histograms for each numerical column
plt.figure(figsize=(20, 20))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(num_rows, 3, i)  # Adjust the number of columns (3 in each row)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[217]:


#Plotting the most popular genres
expl_genre = df.groupby('genre')['popularity'].mean().reset_index()

expl_genre = expl_genre.sort_values(by='popularity', ascending=False).head(20)

sns.set_style("whitegrid")

# Create bar plot
plt.figure(figsize=(10, 6))
sns.barplot(data= expl_genre, x='genre', y='popularity', palette='viridis')
plt.xlabel('Genre', fontsize=14)
plt.ylabel('Average Popularity', fontsize=14)
plt.title("Genres' tracks  with the highest popularity ", fontsize=16)
plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12)
plt.show()


# In[218]:


# @title artist_popularity vs artist_followers
df.plot(kind='scatter', x='artist_popularity', y='artist_followers', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)


# In[227]:


# @title artist_followers vs artist_popularity 
df.plot(kind='scatter', x='artist_followers', y='artist_popularity', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)


# In[222]:


# @title followers vs song popularity
df.plot(kind='scatter', x='artist_followers', y='popularity', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)


# In[223]:


# @title artist_popularity vs song popularity
df.plot(kind='scatter', x='artist_popularity', y='popularity', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)


# # Outlier Detection 

# In[ ]:





# ## Data Preparation

# In[147]:


# Assuming 'target' is your target column and 'data' are your features
X = df.drop(columns=['popularity']).values  # Adjust columns as per your dataset structure
y = df['popularity'].values  # Adjust column name as per your dataset structure


# In[181]:


y


# In[148]:


from sklearn.preprocessing import StandardScaler


# In[167]:


# Identify numerical columns (assuming here only columns 'A' and 'B' are numerical)
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Apply StandardScaler to numerical columns
scl = StandardScaler()
df[numerical_cols] = scl.fit_transform(df[numerical_cols])

# Show the transformed DataFrame
df.head()


# # Visual Approaches

# ## Box Plot

# In[177]:


# Example StandardScaler application
scl = StandardScaler()
df[numerical_cols] = scl.fit_transform(df[numerical_cols])

# Create box plots for numerical columns
plt.figure(figsize=(60, 25))
sns.boxplot(data=df[numerical_cols])
plt.title('Box Plot of Numerical Columns')
plt.xlabel('Column Names')
plt.ylabel('Scaled Values')
plt.show()


# In[163]:


pip install pyod


# In[164]:


from pyod.models.hbos import HBOS


# # Statistical Approaches

# ## Grubbs Test

# In[ ]:




