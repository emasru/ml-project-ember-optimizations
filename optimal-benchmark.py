import ember
import numpy as np
from sklearn.metrics import confusion_matrix

dataset_path = './data/ember2018/'
X_train, y_train, X_test, y_test = ember.read_vectorized_features(dataset_path)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 1024,
    'learning_rate': 0.05,
    'feature_fraction': 0.5,
    'num_iterations': 500,
    'bagging_fraction': 0.5
}

lgbm_model = ember.train_model(dataset_path, params=params)

# Generate predictions (continuous values)
y_pred = lgbm_model.predict(X_test)


def classify_prediction(pred):
    if pred <= -0.5:
        return -1
    elif -0.5 < pred <= 0.5:
        return 0
    else:
        return 1

# Apply thresholding to predictions
y_pred_discrete = np.array([classify_prediction(pred) for pred in y_pred])

# Evaluate the Model Using Confusion Matrix and Compute TPR/FPR

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