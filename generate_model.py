from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle

# Load data and save as CSV
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')
df = pd.concat([X, y], axis=1)
df.to_csv('breast_cancer.csv', index=False)

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model + column info
pickle.dump({'model': model, 'columns': X.columns.tolist()}, open('model.pkl', 'wb'))

print("✅ CSV and model.pkl generated successfully.")
