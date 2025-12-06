from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# -----------------------------
# Load Trained Model
# -----------------------------
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# -----------------------------
# Load Dataset
# -----------------------------
dataset = pd.read_csv("student_marks.csv")
dataset = dataset.fillna(dataset.mode().iloc[0])
dataset['Hours'] = dataset['Hours'].astype(int)
dataset['Scores'] = dataset['Scores'].astype(int)

# -----------------------------
# Function to Create Plots
# -----------------------------
def create_plots():
    plot_dir = 'static'

    # Scatter plot with regression line
    plt.figure(figsize=(8,5))
    plt.scatter(dataset['Hours'], dataset['Scores'], color='blue')
    plt.plot(dataset['Hours'], model.predict(dataset[['Hours']]), color='red', linewidth=2)
    plt.title("Hours vs Scores (Scatter + Regression)")
    plt.xlabel("Hours Studied")
    plt.ylabel("Scores")
    plt.grid(True)
    scatter_path = os.path.join(plot_dir, 'scatter.png')
    plt.savefig(scatter_path)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(5,4))
    sns.heatmap(dataset.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    heatmap_path = os.path.join(plot_dir, 'heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()

    # Histogram of Scores
    plt.figure(figsize=(8,5))
    plt.hist(dataset['Scores'], bins=10, color='orange', edgecolor='black')
    plt.title("Distribution of Scores")
    plt.xlabel("Scores")
    plt.ylabel("Frequency")
    plt.grid(True)
    histogram_path = os.path.join(plot_dir, 'histogram.png')
    plt.savefig(histogram_path)
    plt.close()

    # Boxplot of Hours and Scores
    plt.figure(figsize=(8,5))
    sns.boxplot(data=dataset)
    plt.title("Boxplot of Hours and Scores")
    boxplot_path = os.path.join(plot_dir, 'boxplot.png')
    plt.savefig(boxplot_path)
    plt.close()

    return scatter_path, heatmap_path, histogram_path, boxplot_path

# -----------------------------
# Flask Route
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            hours = int(request.form["hours"])
            prediction = int(model.predict(np.array([[hours]]))[0])
        except:
            prediction = "Invalid input. Enter a number."

    scatter, heatmap, histogram, boxplot = create_plots()
    return render_template("index.html", prediction=prediction,
                           scatter=scatter, heatmap=heatmap,
                           histogram=histogram, boxplot=boxplot)

if __name__ == "__main__":
    app.run(debug=True)
