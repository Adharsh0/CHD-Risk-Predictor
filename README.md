**🫀 CHD Risk Prediction Web App(Logistic Regression Model built from scratch)**

A web application that predicts the 10-year risk of Coronary Heart Disease (CHD) based on patient health data using Logistic Regression implemented from scratch with Gradient Descent and Log Loss.
Built with Flask, NumPy, Pandas, Plotly, and HTML/CSS.

**🚀 Features**
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

**📊 Input Parameters**
The app uses the following health indicators as input:

Age

Gender (0 = Female, 1 = Male)

Cigarettes per day

Total cholesterol

Systolic Blood Pressure (sysBP)

Glucose level

Heart rate

**🛠️ Technologies Used**
Python (NumPy, Pandas, Matplotlib)

Flask (Web framework)

Plotly (Interactive chart)

HTML & CSS (Frontend)

Git & GitHub (Version control)

**📁 Dataset**
Framingham Heart Study dataset

Cleaned by removing education column and rows with missing values

**⚙️ How It Works**
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

📷 Screenshots
![image](https://github.com/user-attachments/assets/da269b12-46c0-4e69-8535-ecafb9446343)
![image](https://github.com/user-attachments/assets/d64e8d1f-a8d9-4827-84ce-6b6e3ed1bd46)
![image](https://github.com/user-attachments/assets/5cc2ffc4-2805-4988-ab8f-e94c7c1632ba)





🔧 Installation & Run
bash
Copy
Edit
# Clone the repository
git clone https://github.com/Adharsh0/chd-predictor.git

cd chd-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
Then go to http://127.0.0.1:5000 in your browser.

📌 Future Improvements
Deploy online using Heroku or Render

Add more health parameters (e.g., BMI, diabetes)

Support CSV upload for batch prediction

User authentication & dashboard

