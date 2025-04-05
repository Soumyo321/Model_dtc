# # from flask import Flask, render_template, request, redirect, url_for, flash
# # import os
# # import uuid
# # import json
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.cluster import KMeans
# # from sklearn.svm import SVC
# # from sklearn.naive_bayes import GaussianNB
# # from sklearn.tree import DecisionTreeClassifier
# # from sklearn.ensemble import RandomForestClassifier

# # app = Flask(__name__)
# # app.secret_key = 'secret'
# # UPLOAD_FOLDER = 'uploads'
# # PLOT_FOLDER = 'static/plots'
# # RESULTS_FILE = 'results_store.json'

# # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # os.makedirs(PLOT_FOLDER, exist_ok=True)

# # # Load existing results
# # if os.path.exists(RESULTS_FILE):
# #     with open(RESULTS_FILE, 'r') as f:
# #         results = json.load(f)
# # else:
# #     results = []

# # # 🔧 Helper: Get model by name
# # def get_model(model_name):
# #     if model_name == 'KMeans':
# #         return KMeans(n_clusters=3)
# #     elif model_name == 'LogisticRegression':
# #         return LogisticRegression(max_iter=1000)
# #     elif model_name == 'SVC':
# #         return SVC()
# #     elif model_name == 'DecisionTree':
# #         return DecisionTreeClassifier()
# #     elif model_name == 'RandomForest':
# #         return RandomForestClassifier()
# #     elif model_name == 'NaiveBayes':
# #         return GaussianNB()
# #     else:
# #         raise ValueError("Unsupported model selected!")

# # # 📊 Helper: Create & save confusion matrix
# # def save_confusion_plot(y_true, y_pred, id):
# #     cm = confusion_matrix(y_true, y_pred)
# #     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# #     disp.plot(cmap='Purples')
# #     path = f"{PLOT_FOLDER}/{id}_confusion.png"
# #     plt.savefig(path)
# #     plt.close()
# #     return f"plots/{id}_confusion.png"

# # # 🎯 Helper: Scatter plot (only if 2 features)
# # def save_scatter_plot(X, y_pred, id):
# #     if X.shape[1] != 2:
# #         return None
# #     plt.figure()
# #     plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred, cmap='plasma')
# #     plt.xlabel(X.columns[0])
# #     plt.ylabel(X.columns[1])
# #     path = f"{PLOT_FOLDER}/{id}_scatter.png"
# #     plt.savefig(path)
# #     plt.close()
# #     return f"plots/{id}_scatter.png"

# # @app.route('/')
# # def index():
# #     return render_template('index.html')

# # @app.route('/analyze', methods=['POST'])
# # def analyze():
# #     file = request.files['dataset']
# #     model_name = request.form['model']
# #     if not file or not file.filename.endswith('.csv'):
# #         flash("Upload a valid CSV file!", "error")
# #         return redirect(url_for('index'))

# #     path = os.path.join(UPLOAD_FOLDER, file.filename)
# #     file.save(path)

# #     try:
# #         data = pd.read_csv(path)
# #         if data.shape[1] < 2:
# #             flash("Dataset must have at least 2 columns", "error")
# #             return redirect(url_for('index'))

# #         # Assume last column is target
# #         X = data.iloc[:, :-1]
# #         y = data.iloc[:, -1]

# #         model = get_model(model_name)

# #         if model_name == "KMeans":
# #             y_pred = model.fit_predict(X)
# #             # KMeans doesn't use y_true — for evaluation, use dummy labels or skip scores
# #             y_true = y if y.nunique() <= 3 else pd.factorize(y)[0]
# #         else:
# #             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# #             model.fit(X_train, y_train)
# #             y_pred = model.predict(X_test)
# #             y_true = y_test

# #         acc = round(accuracy_score(y_true, y_pred) * 100, 2)
# #         prec = round(precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2)
# #         rec = round(recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2)
# #         f1 = round(f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2)

# #         id = str(uuid.uuid4())
# #         confusion_plot = save_confusion_plot(y_true, y_pred, id)
# #         scatter_plot = save_scatter_plot(X, y_pred, id)

# #         entry = {
# #             "id": id,
# #             "model": model_name,
# #             "accuracy": acc,
# #             "precision": prec,
# #             "recall": rec,
# #             "f1": f1,
# #             "confusion_plot": confusion_plot,
# #             "scatter_plot": scatter_plot
# #         }
# #         results.append(entry)

# #         # Save results to file
# #         with open(RESULTS_FILE, 'w') as f:
# #             json.dump(results, f)

# #         flash("✅ Analysis complete!", "success")
# #         return redirect(url_for('dashboard'))

# #     except Exception as e:
# #         flash(f"Error analyzing dataset: {e}", "error")
# #         return redirect(url_for('index'))

# # @app.route('/dashboard')
# # def dashboard():
# #     return render_template('dashboard.html', results=results)

# # @app.route('/details/<id>')
# # def details(id):
# #     result = next((r for r in results if r["id"] == id), None)
# #     if not result:
# #         flash("Result not found", "error")
# #         return redirect(url_for('dashboard'))
# #     return render_template('details.html', result=result)

# # @app.route('/reset')
# # def reset():
# #     global results
# #     results = []
# #     with open(RESULTS_FILE, 'w') as f:
# #         json.dump(results, f)
# #     flash("All results cleared!", "info")
# #     return redirect(url_for('dashboard'))


# # if __name__ == '__main__':
# #     app.run(debug=True)
# from flask import Flask, render_template, request, redirect, url_for, flash, send_file
# import os
# import uuid
# import json
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
# from sklearn.linear_model import LogisticRegression
# from sklearn.cluster import KMeans
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from fpdf import FPDF

# app = Flask(__name__)
# app.secret_key = 'secret'
# UPLOAD_FOLDER = 'uploads'
# PLOT_FOLDER = 'static/plots'
# RESULTS_FILE = 'results_store.json'

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(PLOT_FOLDER, exist_ok=True)

# # Load existing results
# if os.path.exists(RESULTS_FILE):
#     with open(RESULTS_FILE, 'r') as f:
#         results = json.load(f)
# else:
#     results = []

# # 🔧 Helper: Get model by name
# def get_model(model_name):
#     if model_name == 'KMeans':
#         return KMeans(n_clusters=3)
#     elif model_name == 'LogisticRegression':
#         return LogisticRegression(max_iter=1000)
#     elif model_name == 'SVC':
#         return SVC()
#     elif model_name == 'DecisionTree':
#         return DecisionTreeClassifier()
#     elif model_name == 'RandomForest':
#         return RandomForestClassifier()
#     elif model_name == 'NaiveBayes':
#         return GaussianNB()
#     else:
#         raise ValueError("Unsupported model selected!")

# # 📊 Helper: Create & save confusion matrix
# def save_confusion_plot(y_true, y_pred, id):
#     cm = confusion_matrix(y_true, y_pred)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#     disp.plot(cmap='Purples')
#     path = f"{PLOT_FOLDER}/{id}_confusion.png"
#     plt.savefig(path)
#     plt.close()
#     return f"plots/{id}_confusion.png"

# # 🎯 Helper: Scatter plot (only if 2 features)
# def save_scatter_plot(X, y_pred, id):
#     if X.shape[1] != 2:
#         return None
#     plt.figure()
#     plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred, cmap='plasma')
#     plt.xlabel(X.columns[0])
#     plt.ylabel(X.columns[1])
#     path = f"{PLOT_FOLDER}/{id}_scatter.png"
#     plt.savefig(path)
#     plt.close()
#     return f"plots/{id}_scatter.png"

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/analyze', methods=['POST'])
# def analyze():
#     file = request.files['dataset']
#     model_name = request.form['model']
#     if not file or not file.filename.endswith('.csv'):
#         flash("Upload a valid CSV file!", "error")
#         return redirect(url_for('index'))

#     path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(path)

#     try:
#         data = pd.read_csv(path)
#         if data.shape[1] < 2:
#             flash("Dataset must have at least 2 columns", "error")
#             return redirect(url_for('index'))

#         X = data.iloc[:, :-1]
#         y = data.iloc[:, -1]

#         model = get_model(model_name)

#         if model_name == "KMeans":
#             y_pred = model.fit_predict(X)
#             y_true = y if y.nunique() <= 3 else pd.factorize(y)[0]
#         else:
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#             y_true = y_test

#         acc = round(accuracy_score(y_true, y_pred) * 100, 2)
#         prec = round(precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2)
#         rec = round(recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2)
#         f1 = round(f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2)

#         id = str(uuid.uuid4())
#         confusion_plot = save_confusion_plot(y_true, y_pred, id)
#         scatter_plot = save_scatter_plot(X, y_pred, id)

#         entry = {
#             "id": id,
#             "model": model_name,
#             "accuracy": acc,
#             "precision": prec,
#             "recall": rec,
#             "f1": f1,
#             "confusion_plot": confusion_plot,
#             "scatter_plot": scatter_plot
#         }
#         results.append(entry)

#         with open(RESULTS_FILE, 'w') as f:
#             json.dump(results, f)

#         flash("✅ Analysis complete!", "success")
#         return redirect(url_for('dashboard'))

#     except Exception as e:
#         flash(f"Error analyzing dataset: {e}", "error")
#         return redirect(url_for('index'))

# @app.route('/dashboard')
# def dashboard():
#     return render_template('dashboard.html', results=results)

# @app.route('/details/<id>')
# def details(id):
#     result = next((r for r in results if r["id"] == id), None)
#     if not result:
#         flash("Result not found", "error")
#         return redirect(url_for('dashboard'))
#     return render_template('details.html', result=result)

# @app.route('/download/<id>')
# def download_report(id):
#     result = next((r for r in results if r["id"] == id), None)
#     if not result:
#         flash("Result not found", "error")
#         return redirect(url_for('dashboard'))

#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)

#     pdf.set_font("Arial", 'B', 16)
#     pdf.cell(200, 10, txt=f"Model Report - {result['model']}", ln=True, align='C')
#     pdf.ln(10)

#     pdf.set_font("Arial", '', 12)
#     pdf.cell(200, 10, txt=f"Accuracy: {result['accuracy']}%", ln=True)
#     pdf.cell(200, 10, txt=f"Precision: {result['precision']}%", ln=True)
#     pdf.cell(200, 10, txt=f"Recall: {result['recall']}%", ln=True)
#     pdf.cell(200, 10, txt=f"F1 Score: {result['f1']}%", ln=True)
#     pdf.ln(10)

#     confusion_path = os.path.join('static', result["confusion_plot"].split('/')[-1])
#     if os.path.exists(confusion_path):
#         pdf.image(confusion_path, x=10, w=180)
#         pdf.ln(10)

#     if result["scatter_plot"]:
#         scatter_path = os.path.join('static', result["scatter_plot"].split('/')[-1])
#         if os.path.exists(scatter_path):
#             pdf.image(scatter_path, x=10, w=180)
#             pdf.ln(10)

#     pdf_path = f"{PLOT_FOLDER}/{id}_report.pdf"
#     pdf.output(pdf_path)

#     return send_file(pdf_path, as_attachment=True)

# @app.route('/reset')
# def reset():
#     global results
#     results = []
#     with open(RESULTS_FILE, 'w') as f:
#         json.dump(results, f)
#     flash("All results cleared!", "info")
#     return redirect(url_for('dashboard'))

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import os
import uuid
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from fpdf import FPDF

app = Flask(__name__)
app.secret_key = 'secret'

# Folders
UPLOAD_FOLDER = 'uploads'
PLOT_FOLDER = 'static/plots'
RESULTS_FILE = 'results_store.json'
REPORT_FOLDER = 'static/reports'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Load existing results
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)
else:
    results = []



def get_model(model_name):
    models = {
        'KMeans': KMeans(n_clusters=3),
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'SVC': SVC(),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'NaiveBayes': GaussianNB()
    }
    if model_name not in models:
        raise ValueError("Unsupported model selected!")
    return models[model_name]



def save_confusion_plot(y_true, y_pred, id):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Purples')
    path = os.path.join(PLOT_FOLDER, f"{id}_confusion.png")
    plt.savefig(path)
    plt.close()
    return f"plots/{id}_confusion.png"

def save_scatter_plot(X, y_pred, id):
    if X.shape[1] != 2:
        return None
    plt.figure()
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred, cmap='plasma')
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    path = os.path.join(PLOT_FOLDER, f"{id}_scatter.png")
    plt.savefig(path)
    plt.close()
    return f"plots/{id}_scatter.png"



def generate_pdf_report(result):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="ML Model Report", ln=True, align='C')
    pdf.ln(10)

    for key in ['model', 'accuracy', 'precision', 'recall', 'f1']:
        pdf.cell(200, 10, txt=f"{key.capitalize()}: {result[key]}", ln=True)

    for plot_key in ['confusion_plot', 'scatter_plot']:
        plot_path = os.path.join("static", result[plot_key]) if result[plot_key] else None
        if plot_path and os.path.exists(plot_path):
            pdf.add_page()
            pdf.image(plot_path, x=10, y=30, w=180)

    report_path = os.path.join(REPORT_FOLDER, f"{result['id']}_report.pdf")
    pdf.output(report_path)
    return report_path



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['dataset']
    model_name = request.form['model']

    if not file or not file.filename.endswith('.csv'):
        flash("Upload a valid CSV file!", "error")
        return redirect(url_for('index'))

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    try:
        data = pd.read_csv(path)

        if data.shape[1] < 2:
            flash("Dataset must have at least 2 columns", "error")
            return redirect(url_for('index'))

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        model = get_model(model_name)

        if model_name == "KMeans":
            y_pred = model.fit_predict(X)
            y_true = y if y.nunique() <= 3 else pd.factorize(y)[0]
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_true = y_test

        # Metrics
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        prec = round(precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2)
        rec = round(recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2)
        f1 = round(f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2)

        id = str(uuid.uuid4())
        confusion_plot = save_confusion_plot(y_true, y_pred, id)
        scatter_plot = save_scatter_plot(X, y_pred, id)

        entry = {
            "id": id,
            "model": model_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_plot": confusion_plot,
            "scatter_plot": scatter_plot
        }

        results.append(entry)

        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f)

        flash("✅ Analysis complete!", "success")
        return redirect(url_for('dashboard'))

    except Exception as e:
        flash(f"Error analyzing dataset: {e}", "error")
        return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', results=results)

@app.route('/details/<id>')
def details(id):
    result = next((r for r in results if r["id"] == id), None)
    if not result:
        flash("Result not found", "error")
        return redirect(url_for('dashboard'))
    return render_template('details.html', result=result)

@app.route('/reset')
def reset():
    global results
    results = []
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f)
    flash("All results cleared!", "info")
    return redirect(url_for('dashboard'))

@app.route('/download_report/<id>')
def download_report(id):
    result = next((r for r in results if r["id"] == id), None)
    if not result:
        flash("Result not found", "error")
        return redirect(url_for('dashboard'))
    report_path = generate_pdf_report(result)
    return send_file(report_path, as_attachment=True)



if __name__ == '__main__':
    app.run(debug=True)
