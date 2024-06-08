import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import numpy as np

# Load data
file_path = 'weibo_data_sample.csv'
data = pd.read_csv(file_path)

# Preprocess data
# 'label' and 'false' is the target column and other columns are features
X = data.drop(['index', 'label', 'false'], axis=1)

# Convert array-like strings to actual numerical features
def parse_array_column(column):
    if isinstance(column, str) and column.startswith('['):
        return np.array(eval(column))
    return float(column)

for col in X.columns:
    if isinstance(X[col][0], str) and X[col][0].startswith('['):
        # Expand array columns into multiple columns
        expanded_cols = X[col].apply(parse_array_column).apply(pd.Series)
        expanded_cols.columns = [f"{col}_{i}" for i in range(expanded_cols.shape[1])]
        X = X.drop(col, axis=1).join(expanded_cols)

# Handle missing values by filling with the mean for numerical columns
X = X.apply(lambda col: col.fillna(col.mean()) if col.dtype in ['float64', 'int64'] else col.fillna('Unknown'))

# Convert categorical 'topic' column to numerical using one-hot encoding
X = pd.get_dummies(X, columns=['topic'], drop_first=True)

# Extract target labels
y = data['label'].apply(lambda x: 1 if 'T' in x else 0)  # 1 for true, 0 for fake

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Predict on test data
y_pred = svm_model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Evaluate feature importance
importance = permutation_importance(svm_model, X_test, y_test, n_repeats=10, random_state=42)
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance.importances_mean})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

print("Feature Importance:")
print(feature_importance)

# Save the feature importance to a CSV file
feature_importance.to_csv('feature_importance.csv', index=False)
