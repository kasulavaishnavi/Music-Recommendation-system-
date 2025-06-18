
# This library provides support for working with arrays and matrices, along with various mathematical functions
# to operate on these arrays
import numpy as np

# Pandas is a powerful data manipulation and analysis library. It provides data structures like DataFrames and
# Series that allow for easy data handling and analysis
import pandas as pd

# This is part of the scikit-learn library and is used to perform K-means clustering, an unsupervised machine
# learning algorithm that groups similar data points into clusters.
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Matplotlib is a widely used plotting library in Python. The "pyplot" submodule provides functions to create various
# types of plots and visualizations
import matplotlib.pyplot as plt

# Another part of scikit-learn, PCA (Principal Component Analysis) is used for dimensionality reduction. It helps
# transform high-dimensional data into a lower-dimensional representation while preserving as much of the variance as possible.
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

pd.set_option("display.max_columns", None)    # Sets the maximum number of columns to be displayed in a Pandas DataFrame to be unlimited
pd.set_option("display.max_rows", None)       # Sets the maximum number of rows to be displayed in a Pandas DataFrame to be unlimited
pd.set_option("display.width", None)          # Sets the maximum width of the display for a Pandas DataFrame to be unlimited
pd.set_option("display.max_colwidth", None)   # Sets the maximum width of column contents to be unlimited, allowing for complete display of text data
from sklearn.metrics import silhouette_score
print("You have imported all the libraries.")


# ## Loading the Data into a Pandas DataFrame

# In[2]:


# Load the CSV file into a pandas DataFrame
df = pd.read_csv("https://www.sciencebuddies.org/ai/colab/spotify.csv?t=AQUFFSDzPhzwVowq4o-UCEuUPGblqkJu37owVnVhPpgMyg")

# We can see what the dataframe looks like by using the head function
df.head()


# ## Preprocessing the Dataset

# Dropping NaN Values
# Display the shape before dropping NaN values
print("Shape before dropping NaN values:", df.shape)

# Drop NaN values from the DataFrame
df.dropna(inplace=True)

# Display the shape after dropping NaN values
print("Shape after dropping NaN values:", df.shape)
df.head()


# Dropping Features

# In[4]:


# TODO: List the columns that you want to drop
columns_to_drop = ['Artist', 'Album', 'Album_type', 'Title', 'Channel', 'Licensed', 'official_video', 'most_playedon']

# Create a new DataFrame that excludes the specified columns
dropped_df = df.drop(columns=columns_to_drop)

# Let's check if our specified columns are no longer there!
dropped_df.head()


# Since KMeans is a distance-based algorithm, it is crucial to normalize or scale the features to ensure that all features contribute equally to the distance calculations.

# In[5]:


# We can use the describe() function to provide a summary of statistical information about the numerical columns in the DataFrame
dropped_df.describe()


# In[6]:


# TODO: Identify the numerical feature columns you want to normalize
numerical_columns = ['Loudness', 'Tempo', 'Duration_min', 'Views', 'Likes', 'Comments', 'Energy', 'Liveness', 'Stream', 'EnergyLiveness']

# Create a copy of the dropped_df
final_df = dropped_df

# Apply min-max scaling to the selected numerical feature columns
final_df[numerical_columns] = (dropped_df[numerical_columns] - dropped_df[numerical_columns].min()) / (dropped_df[numerical_columns].max() - dropped_df[numerical_columns].min())

# Let's see what our normalization did!
final_df.describe()


# In[7]:


def preprocess_data(df):
    # Encode categorical variables
    label_encoder = LabelEncoder()
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            df_encoded[col] = label_encoder.fit_transform(df_encoded[col])
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_encoded)
    # Convert the scaled array back to a DataFrame
    df_processed = pd.DataFrame(df_scaled, columns=df_encoded.columns)
    return df_processed
# Preprocess the data and encode categorical variables
final_df_encoded = preprocess_data(final_df)
# Preprocess the data and encode categorical variables
final_df_encoded = preprocess_data(final_df)


# In[8]:


# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
if 'Cluster' in final_df_encoded.columns:
    reduced_features = pca.fit_transform(final_df_encoded.drop('Cluster', axis=1)) # Exclude the cluster labels
else:
    reduced_features = pca.fit_transform(final_df_encoded)
# Add the reduced components to the DataFrame
final_df_encoded['pca_1'] = reduced_features[:, 0]
if 'Cluster' in final_df_encoded.columns:
    final_df_encoded['pca_2'] = reduced_features[:, 1]


# ## Clustering the Data

# In[9]:


# Function that works out optimum number of clusters
def optimise_k_means(data, max_k):
    means = []
    inertias = []

    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, n_init=max_k)
        kmeans.fit(data)

        means.append(k)
        inertias.append(kmeans.inertia_)

    # Generate the elbow plot
    plt.figure(figsize=(10, 5))  # Create a new figure
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()
    # Preprocess the data
    final_df_encoded = preprocess_data(final_df)
   

print("This code block has been run and the optimise_k_means() function is now available for use.")


# In[10]:


optimise_k_means(final_df_encoded, 10)


# ## Applying K-Means Clustering

# In[11]:


from sklearn.cluster import KMeans

# Assuming 'final_df_encoded' contains your preprocessed data including encoded categorical variables
kmeans = KMeans(n_clusters=3, n_init='auto') # Set the number of clusters
kmeans.fit(final_df_encoded)
df['Cluster'] = kmeans.labels_
final_df_encoded['Cluster'] = kmeans.labels_
df.head()


# In[13]:


X = final_df_encoded.drop('Cluster', axis=1)
y = final_df_encoded['Cluster']


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Split the data into training and testing sets

# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Train Random Forest Classifier

# In[16]:


rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)


# ### Make predictions

# In[17]:


rf_predictions = rf_classifier.predict(X_test)


# ### Evaluate Random Forest Classifier

# In[18]:


rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_classification_report = classification_report(y_test, rf_predictions)


# In[19]:


# Display results
print("Random Forest Classifier Accuracy:", rf_accuracy)


# In[20]:


print("Random Forest Classifier Classification Report:")
print(rf_classification_report)


# ## Train Decision Tree Classifier

# In[21]:


dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)


# ### Make predictions

# In[22]:


dt_predictions = dt_classifier.predict(X_test)


# ### Evaluate Decision Tree Classifier

# In[23]:


dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_classification_report = classification_report(y_test, dt_predictions)


# In[24]:


print("Decision Tree Classifier Accuracy:", dt_accuracy)


# In[25]:


print("Decision Tree Classifier Classification Report:")
print(dt_classification_report)


# ## Visualize the Model

# In[49]:


# Apply PCA for dimensionality reduction
pca = PCA(n_components=4)
reduced_features = pca.fit_transform(final_df_encoded.drop('Cluster', axis=1)) # Exclude the cluster labels
final_df_encoded['pca_1'] = reduced_features[:, 0]
final_df_encoded['pca_2'] = reduced_features[:, 1]
final_df_encoded['pca_3'] = reduced_features[:, 2]
final_df_encoded['pca_4'] = reduced_features[:, 3]

plt.scatter(final_df_encoded['pca_1'], final_df_encoded['pca_2'], final_df_encoded['pca_3'], c = final_df_encoded['Cluster'], cmap = 'viridis')

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA Components')

# Set x-axis limits
plt.xlim(final_df_encoded['pca_1'].min() - 0, final_df_encoded['pca_1'].max() + 0.5)  

# Set y-axis limits
plt.ylim(final_df_encoded['pca_2'].min() - 0, final_df_encoded['pca_2'].max() + 0.5)  

plt.show()


# In[50]:


# Apply PCA for dimensionality reduction
pca = PCA(n_components=4)
reduced_features = pca.fit_transform(final_df_encoded.drop('Cluster', axis=1)) # Exclude the cluster labels
final_df_encoded['pca_3'] = reduced_features[:, 2]
final_df_encoded['pca_4'] = reduced_features[:, 3]

plt.scatter(final_df_encoded['pca_3'], final_df_encoded['pca_4'], c = final_df_encoded['Cluster'], cmap = 'viridis')

plt.xlabel('PCA Component 3')
plt.ylabel('PCA Component 4')
plt.title('PCA Components')

# Set x-axis limits
plt.xlim(final_df_encoded['pca_3'].min() - 0, final_df_encoded['pca_4'].max() + 0.5)  

# Set y-axis limits
plt.ylim(final_df_encoded['pca_4'].min() - 0, final_df_encoded['pca_3'].max() + 0.5)  

plt.show()


# ## Creating Our Song Recommendation Function

# In[27]:


# This function attemps to find the index of a given track name in the 'Track' column of the dataframe
def find_track_index(track_name, df):
    try:
        # Attempt to find the index of the first occurence of 'track_name' in the 'Track' column of 'df'
        track_index = df[df['Track'] == track_name].index[0]
        # Return the index if found
        return track_index
    except IndexError:
        # If the track name is not found, return None
        return None


# In[28]:


# This function finds song recommendations based on a given track name and the DataFrame 'df'
def find_song_recommendation(track_name, df):
    # Call the 'find_track_index' function to get the index of the provided 'track_name'
    track_index = find_track_index(track_name, df)

    # Retrieve the cluster label of the provided track using its index
    cluster = df.loc[track_index]['Cluster']

    # Create a filter to select rows in 'df' that belong to the same cluster as the provided track
    filter = (df['Cluster'] == cluster)

    # Apply the filter to 'df' to get a DataFrame containing songs from the same cluster
    filtered_df = df[filter]

    # Generate song recommendations by randomly selecting tracks from the same cluster
    for i in range(5):
        # Randomly sample a track from the shuffled DataFrame
        recommendation = filtered_df.sample()
        # Print the recommended track's title and artist
        print(recommendation.iloc[0]['Track'] + ' by ' + recommendation.iloc[0]['Artist'])


# In[29]:


# TODO: Experiment with inputting different song names!
find_song_recommendation('Clint Eastwood', df)


# ## Creating Our Song Randomizer Function

# In[30]:


def find_random_song(track_name, df):
    # Call the 'find_track_index' function to get the index of the provided 'track_name'
    track_index = find_track_index(track_name, df)

    # Retrieve the cluster label of the provided track using its index
    cluster = df.loc[track_index]['Cluster']

    # Create a filter to select rows in 'df' that don't belong to the same cluster as the provided track
    filter = (df['Cluster'] != cluster)

    # Apply the filter to 'df' to get a DataFrame containing songs from different clusters
    filtered_df = df[filter]

    # Generate song recommendations by randomly selecting tracks from the filtered dataframe
    for i in range(5):
        # Randomly sample a track from the shuffled DataFrame
        random_song = filtered_df.sample()
        # Print the random song track's title and artist
        print(random_song.iloc[0]['Track'] + ' by ' + random_song.iloc[0]['Artist'])


# In[31]:


# TODO: Experiment with inputting different song names!
find_random_song('Blue Flame', df)


# ## Evaluating the Model

# In[32]:


# Assuming final_df_encoded is your preprocessed DataFrame
# Assuming n_clusters is the number of clusters you want to use
n_clusters = 4

# Fit KMeans model to preprocessed data
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(final_df_encoded)

# Get cluster labels
cluster_labels = kmeans.labels_

# Calculate silhouette score
silhouette_avg = silhouette_score(final_df_encoded, cluster_labels)
print("Silhouette Score:", silhouette_avg)

