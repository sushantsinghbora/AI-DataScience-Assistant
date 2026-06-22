# 🤖 AI Data Science Assistant

An interactive **AI-powered Data Science web application** built using **Python, Streamlit, Scikit-learn, Pandas, and Microsoft Azure**.  
The application allows users to upload CSV datasets, perform automated **data analysis**, generate **visualizations**, train **machine learning models**, compare model performance, and make **predictions** through an easy-to-use interface.

---

## 🚀 Live Demo
Deployed on **Microsoft Azure App Service**

---

## 📌 Features

### 📊 Data Analysis
- Upload CSV datasets directly from the sidebar
- View complete dataset preview
- Display dataset shape (rows and columns)
- Check missing values in each column
- Show dataset information (`info()`)
- Generate statistical summary using `describe()`

### 📈 Data Visualization
- Interactive bar chart for numerical columns
- Correlation heatmap using Seaborn and Matplotlib

### 🤖 Machine Learning – Regression
- Supports:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- Displays:
  - Mean Squared Error (MSE)
  - R² Score
  - Actual vs Predicted scatter plot
- Compares multiple regression models and highlights the best model

### 🧠 Machine Learning – Classification
- Supports:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
- Displays:
  - Accuracy score
  - Confusion Matrix heatmap
- Compares multiple classification models and highlights the best model

### 🎯 Prediction System
- User can select:
  - Target column
  - Feature columns
- Dynamic input fields generated for selected features
- Predicts target values or class labels
- Supports both regression and classification workflows

### 💾 Download Features
- Download trained machine learning model (`.pkl`)
- Download prediction result as CSV file

### ☁️ Deployment
- Source code managed using **Git & GitHub**
- Deployed on **Microsoft Azure App Service**
- CI/CD deployment using **GitHub Actions**

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Libraries & Frameworks
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

### Machine Learning Models
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

### Cloud / Deployment
- Microsoft Azure App Service
- GitHub Actions

### Tools
- VS Code
- Git
- GitHub

---

## 📂 Project Workflow

1. Upload dataset in CSV format  
2. Explore the dataset in the **Data Analysis** tab  
3. Visualize patterns using the **Visualization** tab  
4. Go to the **Machine Learning** tab  
5. Select:
   - Target column
   - Feature columns
   - Problem type (Regression / Classification)
   - Machine learning algorithm  
6. Train the model and evaluate performance  
7. Enter new feature values to generate predictions  
8. Download trained model or prediction results

---

## 📷 Screenshots
Add screenshots of:
- Home page / dashboard
- Data Analysis tab
- Regression model output
- Classification confusion matrix
- Azure deployed app

---

## ▶️ How to Run Locally

### 1. Clone the repository
```bash
git clone <your-github-repo-link>
cd AI-DataScience-Assistant
