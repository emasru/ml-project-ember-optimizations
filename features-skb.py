import ember
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import pandas as pd
from ember import features

extractor = features.PEFeatureExtractor()

dataset_path = './data/ember2018/'
X_train, y_train, X_test, y_test = ember.read_vectorized_features(dataset_path)
metadata_dataframe = ember.read_metadata(dataset_path)

feature_names = []

for feature in extractor.features:
    for i in range(feature.dim):
        feature_names.append(f"{feature.name}_{i}")

# Perform feature selection
skb = SelectKBest(f_classif, k=100)
skb.fit(X_train, y_train)
X_best = skb.transform(X_train)

# Get the indices of the selected features
selected_indices = skb.get_support(indices=True)
selected_feature_names = [feature_names[i] for i in selected_indices]

print("Selected feature names:", selected_feature_names)
