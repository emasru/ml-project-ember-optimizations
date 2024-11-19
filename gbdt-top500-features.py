import ember
import lightgbm as lgb
import numpy as np
from sklearn.metrics import confusion_matrix

dataset_path = './data/ember2018/'


X_train, y_train, X_test, y_test = ember.read_vectorized_features(dataset_path)

# Train the initial LightGBM model
lgbm_model = ember.train_model(dataset_path)

# Extract feature importance scores
feature_importances = lgbm_model.feature_importance(importance_type='gain')  # Get feature importances
feature_indices = np.argsort(feature_importances)[::-1]
top_100_features = feature_indices[:500]  # Get the indices of the top n features

# Subset the dataset to include only the top 100 features
X_train_reduced = X_train[:, top_100_features]
X_test_reduced = X_test[:, top_100_features]

# Train the LightGBM model with the reduced feature set
train_data = lgb.Dataset(X_train_reduced, label=y_train)
params = {
    'objective': 'binary', 
    'metric': 'auc',  
    'boosting_type': 'gbdt',
    'num_leaves': 2048,
    'learning_rate': 0.05,
    'feature_fraction': 1,
    'num_iterations': 500
}
lgbm_model_reduced = lgb.train(params, train_data)

# Evaluate the model on the test set
y_pred = lgbm_model_reduced.predict(X_test_reduced)

# Define thresholding function to map predictions to discrete classes
def classify_prediction(pred):
    if pred <= -0.5:
        return -1
    elif -0.5 < pred <= 0.5:
        return 0
    else:
        return 1

# Apply thresholding to predictions
y_pred_discrete = np.array([classify_prediction(pred) for pred in y_pred])

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_discrete, labels=[-1, 0, 1])

# Calculate TPR and FPR for each class
tpr = {}
fpr = {}

for idx, class_label in enumerate([-1, 0, 1]):
    tp = conf_matrix[idx, idx]
    fn = np.sum(conf_matrix[idx, :]) - tp
    fp = np.sum(conf_matrix[:, idx]) - tp
    tn = np.sum(conf_matrix) - (tp + fn + fp)

    tpr[class_label] = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr[class_label] = fp / (fp + tn) if (fp + tn) > 0 else 0

# Print TPR and FPR for each class
for class_label in [-1, 0, 1]:
    print(f"Class {class_label}: TPR = {tpr[class_label]:.2f}, FPR = {fpr[class_label]:.2f}")

