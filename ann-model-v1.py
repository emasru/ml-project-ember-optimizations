import ember
from sklearn.feature_selection import SelectKBest, RFE, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
from ember import features

# Initialize PEFeatureExtractor to get feature names
extractor = features.PEFeatureExtractor()

# Load the EMBER dataset
dataset_path = './data/ember2018/'
X_train, y_train, X_test, y_test = ember.read_vectorized_features(dataset_path)
metadata_dataframe = ember.read_metadata(dataset_path)

# Filter out samples with labels -1 in both training and test sets
train_mask = y_train != -1
test_mask = y_test != -1
X_train, y_train = X_train[train_mask], y_train[train_mask]
X_test, y_test = X_test[test_mask], y_test[test_mask]

# Generate and print all feature names
feature_names = []
for feature in extractor.features:
    for i in range(feature.dim):
        feature_names.append(f"{feature.name}_{i}")

# print("All Feature Names:")
    # for name in feature_names:
# print(name)

# Handpick specific features by name (function to select by names)
def select_features_by_name(X, feature_names, selected_names):
    indices = [feature_names.index(name) for name in selected_names if name in feature_names]
    return X[:, indices]

# Example usage: specify the features you want to use
# selected_feature_names = ['feature1_name', 'feature2_name', 'feature3_name']  # Replace with actual feature names
# X_train_handpicked = select_features_by_name(X_train, feature_names, selected_feature_names)
# X_test_handpicked = select_features_by_name(X_test, feature_names, selected_feature_names)

# Feature selection using Recursive Feature Elimination (RFE) with a Random Forest classifier
estimator = RandomForestClassifier(n_estimators=50, random_state=42)
rfe = RFE(estimator, n_features_to_select=50)  # Select top 50 features
rfe.fit(X_train, y_train)
X_train_best = rfe.transform(X_train)
X_test_best = rfe.transform(X_test)

# Standardize the selected features
scaler = StandardScaler()
X_train_best = scaler.fit_transform(X_train_best)
X_test_best = scaler.transform(X_test_best)

# Define a simple ANN model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_best.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train_best, y_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
y_pred = (model.predict(X_test_best) > 0.5).astype(int).reshape(-1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Calculate TPR and FPR
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
tpr = tp / (tp + fn)  # True Positive Rate (Sensitivity)
fpr = fp / (fp + tn)  # False Positive Rate
print(f"True Positive Rate (TPR): {tpr:.4f}")
print(f"False Positive Rate (FPR): {fpr:.4f}")

# Save model weights
model.save_weights('ann_weights.h5')

