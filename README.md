A web application that predicts the 10-year risk of Coronary Heart Disease (CHD) based on patient health data using Logistic Regression implemented from scratch with Gradient Descent and Log Loss.
Built with Flask, NumPy, Pandas, Plotly, and HTML/CSS.
Features
Predicts CHD risk: Not at Risk / Moderate Risk / High Risk

Logistic Regression implemented without using libraries like sklearn.linear_model

Gradient Descent and Log Loss training from scratch

Interactive data visualization with Plotly

Displays model evaluation metrics:

Accuracy (Train & Test)

Precision, Recall, F1-Score

RMSE (Root Mean Squared Error)

Confusion Matrix

Log Loss Curve

Clean and intuitive frontend built with HTML/CSS + Flask

📊 Input Parameters
The app uses the following health indicators as input:

Age

Gender (0 = Female, 1 = Male)

Cigarettes per day

Total cholesterol

Systolic Blood Pressure (sysBP)

Glucose level

Heart rate

🛠️ Technologies Used
Python (NumPy, Pandas, Matplotlib)

Flask (Web framework)

Plotly (Interactive chart)

HTML & CSS (Frontend)

Git & GitHub (Version control)

📁 Dataset
Framingham Heart Study dataset

Cleaned by removing education column and rows with missing values

⚙️ How It Works
Model Training:

Logistic Regression with sigmoid activation

Gradient Descent optimization

Log Loss function for performance

Model Evaluation:

On test set using accuracy, RMSE, confusion matrix, and classification report

Prediction:

User inputs health data via a web form

Inputs are standardized and passed to the model

Model returns probability and CHD risk level

Visualization:

Bar chart of user's health profile

Log loss vs. epochs line plot

# Clone the repository
git clone https://github.com/your-username/chd-predictor.git
cd chd-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
