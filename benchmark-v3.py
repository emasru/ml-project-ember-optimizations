import signal

def print_linenum(signum, frame):
    print ("Currently at line", frame.f_lineno)

signal.signal(signal.SIGINT, print_linenum)

import ember
import lightgbm as lgb
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.metrics import confusion_matrix

# Load data
dataset_path = './data/ember2018/'
X_train, y_train, X_test, y_test = ember.read_vectorized_features(dataset_path)
print('Made it past reading the vectorized features')
# metadata_dataframe = ember.read_metadata(dataset_path)


# params = ember.optimize_model(dataset_path)
# print(params)
# exit()

# Step 1: Feature Selection

# 1.1 Variance Thresholding to remove low-variance features
# var_thresholder = VarianceThreshold(threshold=0.01)
# X_train_var = var_thresholder.fit_transform(X_train)
# X_test_var = var_thresholder.transform(X_test)

# 1.2 Univariate Feature Selection using SelectKBest and mutual information
# mutual_info_classif is too intensive
selector = SelectKBest(score_func=f_classif, k=150)  # Adjust 'k' as needed
X_train_kbest = selector.fit_transform(X_train, y_train)
X_test_kbest = selector.transform(X_test)

print('Made it past skb')
# Step 2: Model Training with Optimized Hyperparameters

# Define LightGBM parameters
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 1024,
    'learning_rate': 0.05,
    'num_iterations': 500,
    'feature_fraction': 1,
    'bagging_fraction': 0.5
}

# Create LightGBM dataset
lgb_train = lgb.Dataset(X_train_kbest, label=y_train)
lgb_test = lgb.Dataset(X_test_kbest, label=y_test, reference=lgb_train)

# Train the model
# removed early stopping rounds
lgbm_model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_test])

# Step 3: Generate Predictions and Map to Discrete Classes

# Generate predictions (continuous values)
y_pred = lgbm_model.predict(X_test_kbest)

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

# Step 4: Evaluate the Model Using Confusion Matrix and Compute TPR/FPR

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

