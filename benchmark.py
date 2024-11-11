import ember
dataset_path = './data/ember2018/'
# ember.create_vectorized_features(dataset_path)
# ember.create_metadata(dataset_path)
X_train, y_train, X_test, y_test = ember.read_vectorized_features(dataset_path)
metadata_dataframe = ember.read_metadata(dataset_path)
lgbm_model = ember.train_model(dataset_path)
# Check the portable version of WinDirStat
windirstat_data = open("./windirstat_portable.exe", "rb").read()
print(round(ember.predict_sample(lgbm_model, windirstat_data), 2))
wannacry_data = open("./wannacry.exe", "rb").read()
print(round(ember.predict_sample(lgbm_model, wannacry_data), 2))
