import ember
dataset_path = './data/ember2018/'
# ember.create_vectorized_features(dataset_path)
# ember.create_metadata(dataset_path)
X_train, y_train, X_test, y_test = ember.read_vectorized_features(dataset_path)
# metadata_dataframe = ember.read_metadata(dataset_path)
lgbm_model = ember.train_model(dataset_path)
print(lgbm_model.params)

# lgbm_model.save_model('lgbm_model_it1.txt')

# Check the portable version of WinDirStat
# windirstat_data = open("./windirstat_portable.exe", "rb").read()
# print(round(ember.predict_sample(lgbm_model, windirstat_data), 2))
# wannacry_data = open("./wannacry.exe", "rb").read()
# print(round(ember.predict_sample(lgbm_model, wannacry_data), 2))

import numpy as np
from sklearn.metrics import confusion_matrix

# Generate predictions (continuous values)
y_pred = lgbm_model.predict(X_test)

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

# print("Feature Importance Scores:")
# feature_importances = lgbm_model.feature_importance(importance_type='split')  # 'split' for split count, 'gain' for average gain
# for i, importance in enumerate(feature_importances):
#    print(f"Feature {i}: {importance}")
