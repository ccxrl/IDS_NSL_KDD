import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load saved model and preprocessing tools
model = joblib.load('rf_model_selected_features.joblib')
scaler = joblib.load('scaler_fs.joblib')
sfm = joblib.load('feature_selector.joblib')
label_encoders = joblib.load('label_encoders_fs.joblib')

# Load new/test data
df_test = pd.read_csv("NSL_KDD_Test.csv", header=None)

# Define the same column names
column_names = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
    'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
    'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
    'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
    'dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label'
]
df_test.columns = column_names

# Convert label to binary: 0 = normal, 1 = attack
df_test['label'] = df_test['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Apply label encoding to categorical columns
for col in ['protocol_type', 'service', 'flag']:
    le = label_encoders[col]
    df_test[col] = le.transform(df_test[col])

# Split features and labels
X_new = df_test.drop(['label'], axis=1)
y_true = df_test['label']

# Scale features
X_scaled_new = scaler.transform(X_new)

# Select important features
X_selected_new = sfm.transform(X_scaled_new)

# Predict using the saved model
y_pred = model.predict(X_selected_new)

# Evaluate performance
print("\nTest Accuracy on New Data:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))
