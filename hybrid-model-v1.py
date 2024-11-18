import ember
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from ember import features
from sklearn.metrics import accuracy_score

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

# Generate feature names
feature_names = []
for feature in extractor.features:
    for i in range(feature.dim):
        feature_names.append(f"{feature.name}_{i}")

# Select the top 150 most discriminative features
skb = SelectKBest(f_classif, k=150)
skb.fit(X_train, y_train)
X_train_best = skb.transform(X_train)
X_test_best = skb.transform(X_test)

# Standardize the selected features
scaler = StandardScaler()
X_train_best = scaler.fit_transform(X_train_best)
X_test_best = scaler.transform(X_test_best)

# Reshape data for CNN input (samples, timesteps, features)
X_train_rnn = X_train_best.reshape(-1, 150, 1)
X_test_rnn = X_test_best.reshape(-1, 150, 1)

# Build the Hybrid CNN + LSTM model
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(150, 1), kernel_regularizer=l2(0.001)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Conv1D(32, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    LSTM(32, kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=3, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(X_train_rnn, y_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
y_pred = (model.predict(X_test_rnn) > 0.5).astype(int).reshape(-1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

from sklearn.metrics import confusion_matrix

# Step 1: Generate binary predictions from the model
y_pred = (model.predict(X_test_rnn) > 0.5).astype(int).reshape(-1)

# Step 2: Compute the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Step 3: Calculate TPR and FPR
tpr = tp / (tp + fn)  # True Positive Rate (Sensitivity)
fpr = fp / (fp + tn)  # False Positive Rate

print(f"True Positive Rate (TPR): {tpr:.4f}")
print(f"False Positive Rate (FPR): {fpr:.4f}")
