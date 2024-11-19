import ember
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import pandas as pd
from ember import features
import matplotlib.pyplot as plt
import lightgbm as lgb

# Extract features
extractor = features.PEFeatureExtractor()

# Set dataset path
dataset_path = './data/ember2018/'
X_train, y_train, X_test, y_test = ember.read_vectorized_features(dataset_path)
metadata_dataframe = ember.read_metadata(dataset_path)

# Create feature names
feature_names = []
for feature in extractor.features:
    for i in range(feature.dim):
        feature_names.append(f"{feature.name}_{i}")

# Train LightGBM model
train_data = lgb.Dataset(X_train, label=y_train)
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 1024,
    'learning_rate': 0.05,
    'feature_fraction': 1
}
lgbm_model = lgb.train(params, train_data)

# Get feature importance (gain method)
feature_importances = lgbm_model.feature_importance(importance_type='gain')

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# Sort by importance and select the top 50
top_50_features = feature_importance_df.sort_values(by='Importance',
                                                     ascending=False).head(50)

# Plot the top 50 features
plt.figure(figsize=(10, 20))
plt.barh(top_50_features['Feature'], top_50_features['Importance'], align='center')
plt.xlabel('Importance (Gain)')
plt.ylabel('Feature')
plt.title('Top 100 Features Ranked by Gain (LightGBM)')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature at the top
plt.tight_layout()
plt.show()