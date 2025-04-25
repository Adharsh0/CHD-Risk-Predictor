from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Log Loss function
def log_loss(y_true, y_pred):
    epsilon = 1e-15  # To prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clipping values to prevent log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Gradient Descent for Logistic Regression
def logistic_regression(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    w = np.zeros(n)  # Initialize weights to 0
    b = 0  # Initialize bias to 0
    losses = []
    
    for epoch in range(epochs):
        # Compute the model prediction
        z = np.dot(X, w) + b
        y_pred = sigmoid(z)
        
        # Compute gradients
        dw = (1 / m) * np.dot(X.T, (y_pred - y))  # Gradient w.r.t weights
        db = (1 / m) * np.sum(y_pred - y)  # Gradient w.r.t bias
        
        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db
        
        # Compute the log loss for this epoch
        loss = log_loss(y, y_pred)
        losses.append(loss)
        
        # Print the loss every 100 epochs for monitoring
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Log Loss = {loss}")
    
    return w, b, losses

# Make predictions
def predict(X, w, b):
    z = np.dot(X, w) + b
    return sigmoid(z)

# Load and preprocess dataset
disease_df = pd.read_csv("framingham.csv")
disease_df.drop(['education'], axis=1, inplace=True)
disease_df.rename(columns={'male': 'Sex_male'}, inplace=True)
disease_df.dropna(inplace=True)

# Feature selection and scaling - including heartRate
X = np.asarray(disease_df[['age', 'Sex_male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose', 'heartRate']])
y = np.asarray(disease_df['TenYearCHD'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=4)

# Train the logistic regression model
w, b, losses = logistic_regression(X_train, y_train, learning_rate=0.01, epochs=1000)

# Make predictions on test data
y_pred_train = predict(X_train, w, b)
y_pred_test = predict(X_test, w, b)

# Convert predictions to binary (0 or 1)
y_pred_train_binary = (y_pred_train >= 0.5).astype(int)
y_pred_test_binary = (y_pred_test >= 0.5).astype(int)

# Model Evaluation
accuracy_train = np.mean(y_pred_train_binary == y_train) * 100
accuracy_test = np.mean(y_pred_test_binary == y_test) * 100
accuracy = accuracy_test  # For the web app

# Calculate classification report and other metrics
class_report = classification_report(y_test, y_pred_test_binary, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred_test_binary).tolist()
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test_binary))

# Extract precision, recall, F1-score for class 1 (CHD positive)
precision = class_report['1']['precision'] * 100
recall = class_report['1']['recall'] * 100
f1_score = class_report['1']['f1-score'] * 100

# Create loss plot
plt.figure(figsize=(10, 6))
plt.plot(range(1000), losses)
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.title('Log Loss vs Epochs')
img = io.BytesIO()
plt.savefig(img, format='png')
img.seek(0)
loss_plot = base64.b64encode(img.getvalue()).decode()

# Prediction with risk levels and probability
def predict_chd(age, Sex_male, cigsPerDay, totChol, sysBP, glucose, heartRate):
    input_data = np.array([[age, Sex_male, cigsPerDay, totChol, sysBP, glucose, heartRate]])
    input_scaled = scaler.transform(input_data)
    probability = predict(input_scaled, w, b)[0]  # Probability of class '1'
    risk_level = (
        "Not at Risk (0)" if probability < 0.5 else
        "High Risk(1) " 
        
    )
    return risk_level, probability * 100  # Return risk level and probability percentage

# Bar chart creation function
def create_bar_chart(data):
    categories = ['Age', 'Sex', 'Cigs per Day', 'Total Cholesterol', 'Systolic BP', 'Glucose', 'Heart Rate']
    fig = px.bar(x=categories, y=data, title="User Health Profile", labels={'x': 'Health Parameter', 'y': 'Value'})
    fig.update_traces(marker_color='blue')
    graph_html = pio.to_html(fig, full_html=False)
    return graph_html

@app.route('/')
def index():
    return render_template("index.html", loss_plot=loss_plot)

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        # Get user input
        age = float(request.form['age'])
        Sex_male = int(request.form['sex'])
        cigsPerDay = float(request.form['cigsPerDay'])
        totChol = float(request.form['totChol'])
        sysBP = float(request.form['sysBP'])
        glucose = float(request.form['glucose'])
        heartRate = float(request.form['heartRate'])

        # Get prediction result and probability
        result, probability = predict_chd(age, Sex_male, cigsPerDay, totChol, sysBP, glucose, heartRate)
        
        # Prepare data for bar chart
        user_data = [age, Sex_male, cigsPerDay, totChol, sysBP, glucose, heartRate]
        bar_chart = create_bar_chart(user_data)

        # Pass all metrics to template
        return render_template(
            "index.html",
            result=result,
            probability=probability,
            bar_chart=bar_chart,
            accuracy=accuracy,
            accuracy_train=accuracy_train,
            accuracy_test=accuracy_test,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            rmse=rmse,
            conf_matrix=conf_matrix,
            loss_plot=loss_plot
        )

    except Exception as e:
        return render_template("index.html", error=str(e), loss_plot=loss_plot)

if __name__ == '__main__':
    app.run(debug=True)