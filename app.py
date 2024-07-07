import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from flask_restful import Api
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)
api = Api(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variables for ML models and data
df = None
cluster_model = None
classifier_model = None
regression_model = None


# Function to load and preprocess dataset
def load_dataset(file_path):
    global df
    df = pd.read_csv(file_path)
    clean_dataset()  # Clean the dataset after loading


def clean_dataset():
    global df
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)


def preprocess_dataset():
    global df
    if df is not None:
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = LabelEncoder().fit_transform(df[col])

        numerical_cols = df.select_dtypes(include=np.number).columns
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


# Function to train K-means clustering model
def train_clustering_model(n_clusters):
    global cluster_model
    if df is not None:
        X = df.select_dtypes(include=np.number).fillna(0)
        cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_model.fit(X)


# Function to train classification model
def train_classification_model(target_column):
    global classifier_model
    if df is not None:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        le = LabelEncoder()
        y = le.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        classifier_model = RandomForestClassifier(random_state=42)
        classifier_model.fit(X_train, y_train)
        y_pred = classifier_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return f"Classification model accuracy: {accuracy:.2f}"


# Function to train regression model
def train_regression_model(target_column):
    global regression_model
    if df is not None:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        regression_model = RandomForestRegressor(random_state=42)
        regression_model.fit(X_train, y_train)
        y_pred = regression_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return f"Regression model MSE: {mse:.2f}"


# Flask routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_data.csv')
        file.save(file_path)
        load_dataset(file_path)
        preprocess_dataset()  # Preprocess the dataset after loading
        return redirect(url_for('questions'))
    return redirect(url_for('index'))


@app.route('/questions', methods=['GET', 'POST'])
def questions():
    global df, cluster_model, classifier_model, regression_model
    if request.method == 'POST':
        question = request.form['question']
        answer = process_question(question)
        return render_template('questions.html', answer=answer)
    return render_template('questions.html', answer=None)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Function to process user questions
def process_question(question):
    global df, cluster_model, classifier_model, regression_model
    if df is None:
        return "No dataset uploaded."

    question = question.lower()

    # Define patterns for recognizing questions
    patterns = {
        'mean': r"mean of (.+)",
        'median': r"median of (.+)",
        'max': r"max(?:imum)? of (.+)",
        'min': r"min(?:imum)? of (.+)",
        'sum': r"sum of (.+)",
        'count_rows': r"count rows",
        'unique_values': r"unique values in (.+)",
        'count_unique_values': r"count unique values in (.+)",
        'describe': r"describe (.+)",
        'std': r"standard deviation of (.+)",
        'var': r"variance of (.+)",
        'column_names': r"column names",
        'missing_values': r"missing values in (.+)",
        'visualize': r"visualize (.+)",
        'scatter_plot': r"scatter plot of (.+) vs (.+)",
        'line_graph': r"line graph of (.+)",
        'cluster_data': r"cluster data using (.+)",
        'classify_data': r"classify data using (.+)",
        'predict_data': r"predict using (.+)"
    }

    def column_exists(column_name):
        return column_name in df.columns

    def generate_scatter_plot(x_column, y_column):
        plt.figure(figsize=(8, 4))
        plt.scatter(df[x_column], df[y_column], alpha=0.5)
        plt.title(f'Scatter Plot: {x_column} vs {y_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'scatter_plot.png')
        plt.savefig(plot_path, dpi=100)
        plt.close()
        return f'/uploads/scatter_plot.png'

    def generate_line_graph(column):
        plt.figure(figsize=(8, 4))
        plt.plot(df.index, df[column], marker='o', linestyle='-')
        plt.title(f'Line Graph: {column}')
        plt.xlabel('Index')
        plt.ylabel(column)
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'line_graph.png')
        plt.savefig(plot_path, dpi=100)
        plt.close()
        return f'/uploads/line_graph.png'

    for key, pattern in patterns.items():
        match = re.match(pattern, question)
        if match:
            if key == 'mean':
                column = match.group(1)
                if column_exists(column):
                    return f"The mean of {column} is {df[column].mean()}"
                else:
                    return f"Column {column} does not exist."

            if key == 'median':
                column = match.group(1)
                if column_exists(column):
                    return f"The median of {column} is {df[column].median()}"
                else:
                    return f"Column {column} does not exist."

            if key == 'max':
                column = match.group(1)
                if column_exists(column):
                    return f"The maximum value of {column} is {df[column].max()}"
                else:
                    return f"Column {column} does not exist."

            if key == 'min':
                column = match.group(1)
                if column_exists(column):
                    return f"The minimum value of {column} is {df[column].min()}"
                else:
                    return f"Column {column} does not exist."

            if key == 'sum':
                column = match.group(1)
                if column_exists(column):
                    return f"The sum of {column} is {df[column].sum()}"
                else:
                    return f"Column {column} does not exist."

            if key == 'count_rows':
                return f"The dataset has {len(df)} rows."

            if key == 'unique_values':
                column = match.group(1)
                if column_exists(column):
                    return f"The unique values in {column} are {df[column].unique().tolist()}"
                else:
                    return f"Column {column} does not exist."

            if key == 'count_unique_values':
                column = match.group(1)
                if column_exists(column):
                    return f"The number of unique values in {column} is {df[column].nunique()}"
                else:
                    return f"Column {column} does not exist."

            if key == 'describe':
                column = match.group(1)
                if column_exists(column):
                    return f"The description of {column} is {df[column].describe().to_dict()}"
                else:
                    return f"Column {column} does not exist."

            if key == 'std':
                column = match.group(1)
                if column_exists(column):
                    return f"The standard deviation of {column} is {df[column].std()}"
                else:
                    return f"Column {column} does not exist."

            if key == 'var':
                column = match.group(1)
                if column_exists(column):
                    return f"The variance of {column} is {df[column].var()}"
                else:
                    return f"Column {column} does not exist."

            if key == 'column_names':
                return f"The column names are {df.columns.tolist()}"

            if key == 'missing_values':
                column = match.group(1)
                if column_exists(column):
                    return f"The number of missing values in {column} is {df[column].isnull().sum()}"
                else:
                    return f"Column {column} does not exist."

            if key == 'visualize':
                column = match.group(1)
                if column_exists(column):
                    if len(df[column].unique()) > 10:  # Too many unique values, visualize distribution
                        plt.figure(figsize=(8, 4))
                        df[column].value_counts().plot(kind='bar')
                        plt.title(f'Distribution of {column}')
                        plt.xlabel(column)
                        plt.ylabel('Count')
                        plt.grid(True)
                        plt.tight_layout()
                        plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'distribution_plot.png')
                        plt.savefig(plot_path, dpi=100)
                        plt.close()
                        return f'/uploads/distribution_plot.png'
                    else:  # Few unique values, visualize scatter plot
                        return generate_scatter_plot(column, df.columns[0])  # Assuming df.columns[0] as y-axis

            if key == 'scatter_plot':
                x_column = match.group(1)
                y_column = match.group(2)
                if column_exists(x_column) and column_exists(y_column):
                    return generate_scatter_plot(x_column, y_column)
                else:
                    return f"Columns {x_column} or {y_column} do not exist."

            if key == 'line_graph':
                column = match.group(1)
                if column_exists(column):
                    return generate_line_graph(column)
                else:
                    return f"Column {column} does not exist."

            if key == 'cluster_data':
                algorithm = match.group(1)
                if algorithm.lower() == 'kmeans':
                    train_clustering_model(n_clusters=3)  # Example: K-means with 3 clusters
                    return "K-means clustering model trained."
                # Add other clustering algorithms as needed

            if key == 'classify_data':
                model_type = match.group(1)
                if model_type.lower() == 'random forest':
                    target_column = df.columns[-1]  # Example: Classify using last column as target
                    return train_classification_model(target_column)
                # Add other classification algorithms as needed

            if key == 'predict_data':
                model_type = match.group(1)
                if model_type.lower() == 'random forest':
                    target_column = df.columns[-1]  # Example: Predict using last column as target
                    return train_regression_model(target_column)
                # Add other prediction algorithms as needed

    return "Question not recognized or dataset not loaded."


if __name__ == '__main__':
    app.run(debug=True)