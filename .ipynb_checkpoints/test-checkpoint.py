import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Sample data for demonstration
data = pd.DataFrame([[1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8],
                     [1/2, 1, 2, 3, 3, 4, 5, 6, 7, 8, 8],
                     [1/2, 1/2, 1, 2, 3, 4, 4, 5, 6, 7, 7],
                     [1/3, 1/3, 1/2, 1, 2, 3, 3, 4, 5, 7, 8],
                     [1/4, 1/3, 1/3, 1/2, 1, 2, 2, 3, 4, 5, 6],
                     [1/5, 1/4, 1/4, 1/3, 1/2, 1, 2, 4, 5, 7, 7],
                     [1/5, 1/5, 1/4, 1/3, 1/2, 1/2, 1, 2, 3, 5, 6],
                     [1/6, 1/6, 1/5, 1/4, 1/3, 1/4, 1/2, 1, 2, 4, 5],
                     [1/7, 1/7, 1/6, 1/5, 1/4, 1/5, 1/3, 1/2, 1, 3, 4],
                     [1/8, 1/8, 1/7, 1/7, 1/5, 1/7, 1/5, 1/4, 1/3, 1, 3],
                     [1/8, 1/8, 1/7, 1/8, 1/6, 1/7, 1/6, 1/5, 1/4, 1/3, 1]])

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Perform PCA
pca = PCA()  # Adjust number of components as needed
principal_components = pca.fit_transform(data_scaled)

# Create a DataFrame with the principal components
pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11'])

# Explained variance
explained_variance = pca.explained_variance_ratio_

# Output results
print("Principal Components:\n", pc_df.round(3))
print("Explained Variance:\n", explained_variance)

# Interactive 3D Plotting the principal components
fig = px.scatter_3d(
    pc_df, x='PC1', y='PC2', z='PC3',
    color='PC4',  # Using the third component for color
    title='3D PCA of Flood Mapping Data (Transposed)',
    labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2', 'PC3': 'Principal Component 3'}
)

fig.show()
