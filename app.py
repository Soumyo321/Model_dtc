from flask import Flask, render_template, request, redirect
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    model_file = request.files['model']
    data_file = request.files['dataset']

    if not model_file or not data_file:
        return render_template('index.html', error="Both files are required.")

    model_path = os.path.join(app.config['UPLOAD_FOLDER'], model_file.filename)
    data_path = os.path.join(app.config['UPLOAD_FOLDER'], data_file.filename)

    model_file.save(model_path)
    data_file.save(data_path)

    try:
        model_data = pickle.load(open(model_path, 'rb'))
        model = model_data['model']
        expected_columns = model_data['columns']
    except Exception as e:
        return render_template('index.html', error=f"Error loading model: {str(e)}")

    try:
        df = pd.read_csv(data_path)
        if 'target' not in df.columns:
            return render_template('index.html', error="CSV must contain a 'target' column.")
        X = df.drop('target', axis=1)
        y = df['target']

        # Ensure columns match
        if list(X.columns) != expected_columns:
            return render_template('index.html', error="CSV columns do not match model's training columns.")

        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)

        return render_template('dashboard.html', accuracy=round(acc*100, 2), conf_matrix=cm)
    
    except Exception as e:
        return render_template('index.html', error=f"Error processing data: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
