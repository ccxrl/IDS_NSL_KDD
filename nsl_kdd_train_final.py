# final ish may 12, 1:43 pm

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectFromModel

# Load from local CSV file
df = pd.read_csv("NSL_KDD_Train.csv", header=None)

# Define column names
column_names = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
    'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
    'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
    'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
    'dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label'
]

df.columns = column_names

# Convert label to binary: 0 = normal, 1 = attack
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Encode categorical columns
categorical_cols = ['protocol_type', 'service', 'flag']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features and labels
X = df.drop(['label'], axis=1)
y = df['label']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Output shape info
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("First 5 rows of scaled features (X_train):")
print(X_train[:5])


# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Create Random Forest model
rf = RandomForestClassifier(random_state=42)

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                          cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get best model
best_rf = grid_search.best_estimator_

# Feature selection using SelectFromModel
from sklearn.feature_selection import SelectFromModel

# Select features with importance above the median
sfm = SelectFromModel(best_rf, threshold='median', prefit=True)
X_train_selected = sfm.transform(X_train)
X_test_selected = sfm.transform(X_test)

# Retrain model on selected features
rf_selected = RandomForestClassifier(random_state=42)
rf_selected.fit(X_train_selected, y_train)
y_pred_selected = rf_selected.predict(X_test_selected)

# Print metrics for model with selected features
print("\nAccuracy with Selected Features:", accuracy_score(y_test, y_pred_selected))
print("\nClassification Report (Selected Features):\n")
print(classification_report(y_test, y_pred_selected))
print("\nConfusion Matrix (Selected Features):\n")
print(confusion_matrix(y_test, y_pred_selected))

# Optional: Save the selected model
joblib.dump(rf_selected, 'rf_model_selected_features.joblib')
joblib.dump(sfm, 'feature_selector.joblib')

# Print best parameters
print("\nBest Parameters:", grid_search.best_params_)

# Make predictions
y_pred = best_rf.predict(X_test)

# Print metrics
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Feature importance visualization
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Top 10 Most Important Features')
plt.tight_layout()
plt.show()

# Save the model
joblib.dump(best_rf, 'intrusion_detection_rf_model_fs.joblib')
joblib.dump(scaler, 'scaler_fs.joblib')
joblib.dump(label_encoders, 'label_encoders_fs.joblib')