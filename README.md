# 🤖 AI Data Science Assistant

AI Data Science Assistant is a **Streamlit-based machine learning and data analysis web application** that allows users to upload a CSV dataset, perform exploratory data analysis, visualize data, train machine learning models, compare algorithms, and make predictions — all from an interactive interface.

It is designed as a beginner-friendly **Data Science + Machine Learning project** that demonstrates practical skills in **Python, Pandas, Scikit-learn, Streamlit, and Data Visualization**.

---

## 🚀 Features

### 📂 Dataset Upload
- Upload CSV datasets directly from the sidebar
- Preview uploaded data instantly
- Supports numerical datasets for analysis and ML tasks

### 📊 Data Analysis Module
- View dataset shape
- Check missing values
- Get statistical summary of the data
- Display full uploaded dataset in table format

### 📈 Visualization Module
- Bar chart visualization
- Correlation heatmap for numeric columns
- Helps understand relationships between features

### 🤖 Machine Learning Module
Supports both **Regression** and **Classification** problems.

#### Regression Models
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

#### Classification Models
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

### 🏆 Model Comparison
- Automatically compares multiple ML models
- Displays evaluation metrics:
  - **Regression:** MSE, R² Score
  - **Classification:** Accuracy
- Highlights the **best model**

### 💾 Model Saving & Download
- Trained models are saved using **Joblib**
- Download trained model directly from the app

### 🔮 Prediction System
- User can enter custom feature values
- App predicts output using the trained model
- Prediction result can also be downloaded as a CSV file

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Joblib**

---

## 📂 Project Structure

```bash
AI-DataScience-Assistant/
│── app.py
│── dashboard.py
│── prototype.py
│── students.csv
│── requirements.txt
│── README.md
│── models/
│   └── trained_model.pkl
│── screenshots/
│   ├── home.png
│   ├── data_analysis.png
│   ├── regression.png
│   ├── classification.png
│   └── visualization.png