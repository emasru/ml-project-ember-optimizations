from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import ember
from ember import features

# Assuming the feature names for X_encoded are generated in the same way
# Initialize PEFeatureExtractor and get feature names for all features in X_encoded
extractor = features.PEFeatureExtractor()

feature_names = []
for feature in extractor.features:
    for i in range(feature.dim):
        feature_names.append(f"{feature.name}_{i}")

# Load the dataset (X_encoded in this case)
dataset_path = './data/ember2018/'
X_train, y_train, X_test, y_test = ember.read_vectorized_features(dataset_path)
metadata_dataframe = ember.read_metadata(dataset_path)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)  # Standardize all features

# Apply PCA to reduce dimensionality on all features
pca = PCA(n_components=2)  # Adjust the number of components as needed
X_pca = pca.fit_transform(X_scaled)

# View explained variance ratio for each principal component
explained_variance = pca.explained_variance_ratio_
print("Explained variance by each PCA component:", explained_variance)

# Display the relationship between each principal component and original features
components_df = pd.DataFrame(pca.components_, columns=feature_names, index=[f'PC-{i+1}' for i in range(pca.n_components_)])
print("Principal components relation with all features:")
print(components_df)

explained_variance = pca.explained_variance_ratio_
print("Explained variance by each component:", explained_variance)
