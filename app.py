from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import plotly.express as px  # Import plotly.express
import plotly.io as pio  # Import plotly.io

app = Flask(__name__)

# Load and preprocess dataset
disease_df = pd.read_csv("framingham.csv")
disease_df.drop(['education'], axis=1, inplace=True)
disease_df.rename(columns={'male': 'Sex_male'}, inplace=True)
disease_df.dropna(inplace=True)

# Feature selection and scaling
X = np.asarray(disease_df[['age', 'Sex_male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose']])
y = np.asarray(disease_df['TenYearCHD'])

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=4)

# Train the model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Model accuracy
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100  # Model accuracy in percentage

# Prediction with risk levels
def predict_chd(age, Sex_male, cigsPerDay, totChol, sysBP, glucose):
    input_data = np.array([[age, Sex_male, cigsPerDay, totChol, sysBP, glucose]])
    input_scaled = scaler.transform(input_data)
    probability = logreg.predict_proba(input_scaled)[0][1]  # Probability of class '1'

    if probability < 0.3:
        return "Not at Risk (0)"
    elif 0.3 <= probability < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk (1)"

# Bar chart creation function
def create_bar_chart(data):
    categories = ['Age', 'Sex', 'Cigs per Day', 'Total Cholesterol', 'Systolic BP', 'Glucose']
    fig = px.bar(x=categories, y=data, title="User Health Profile", labels={'x': 'Health Parameter', 'y': 'Value'})
    fig.update_traces(marker_color='blue')  # Change bar color if needed
    graph_html = pio.to_html(fig, full_html=False)
    return graph_html

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        age = float(request.form['age'])
        Sex_male = int(request.form['sex'])
        cigsPerDay = float(request.form['cigsPerDay'])
        totChol = float(request.form['totChol'])
        sysBP = float(request.form['sysBP'])
        glucose = float(request.form['glucose'])

        # Get prediction result
        result = predict_chd(age, Sex_male, cigsPerDay, totChol, sysBP, glucose)
        
        # Prepare data for bar chart
        user_data = [age, Sex_male, cigsPerDay, totChol, sysBP, glucose]
        bar_chart = create_bar_chart(user_data)

        return render_template("index.html", result=result, bar_chart=bar_chart, accuracy=accuracy)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
