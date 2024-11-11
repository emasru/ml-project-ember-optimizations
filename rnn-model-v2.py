import ember
from ember import features
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Initialize PEFeatureExtractor
extractor = features.PEFeatureExtractor()

# Load the EMBER dataset
dataset_path = './data/ember2018/'
X_train, y_train, X_test, y_test = ember.read_vectorized_features(dataset_path)
metadata_dataframe = ember.read_metadata(dataset_path)

# Generate feature names
feature_names = []
for feature in extractor.features:
    for i in range(feature.dim):
        feature_names.append(f"{feature.name}_{i}")

# Perform feature selection to get top 100 features
skb = SelectKBest(f_classif, k=100)
skb.fit(X_train, y_train)
X_train_best = skb.transform(X_train)
X_test_best = skb.transform(X_test)

# Standardize the selected features
scaler = StandardScaler()
X_train_best = scaler.fit_transform(X_train_best)
X_test_best = scaler.transform(X_test_best)

# Reshape data for RNN input (samples, timesteps, features)
# Here, we treat each feature as a timestep for the RNN
X_train_rnn = X_train_best.reshape(-1, 100, 1)
X_test_rnn = X_test_best.reshape(-1, 100, 1)

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Define the RNN model with additional regularization
model = Sequential([
    LSTM(64, input_shape=(100, 1), return_sequences=True, kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    LSTM(32, kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Train the model with early stopping
model.fit(X_train_rnn, y_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
y_pred = model.predict(X_test_rnn)
y_pred = (y_pred > 0.5).astype(int).reshape(-1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

