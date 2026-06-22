import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io #for handling file uploads/io handles text streams or buffers
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


st.set_page_config(
    page_title="AI Data Science Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1 {
    color: #A81B4D;
}

.stButton>button {
    background-color: #A81B4D;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# Beautiful Title
st.markdown(
    "<h1 style='text-align: center; color: #A81B4D;'>🤖 AI Data Science Assistant</h1>",
    unsafe_allow_html=True
)

st.markdown("---")

st.sidebar.title("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    #read the csv file and display it
    df = pd.read_csv(uploaded_file)

    #select only numarical columns
    numeric_df = df.select_dtypes(include='number')

    tab1, tab2, tab3 = st.tabs(
        [
            "📊 Data Analysis",
            "🤖 Machine Learning",
            "📈 Visualization"
        ]
    )

    with tab1:
        #display success message and the dataframe
        st.success("File uploaded successfully!")

        st.write(df)

        #display the shape of the dataset
        st.write("Dataset Shape:")
        st.write(df.shape)

        #display the number of rows and columns in the dataset
        col1, col2 = st.columns(2)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])

        #display the number of missing values in each column
        st.write("Missing Values:")
        st.write(df.isnull().sum())

        #display information about the dataset using the info() method of the dataframe
        st.write("Dataset Information:")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        #display statistical summary of the dataset using the describe() method of the dataframe
        st.write("Statistical Summary:")
        st.write(df.describe())

    with tab2:
        # Machine Learning Section 
        st.write("Machine Learning Model")

        #check if dataset has at least 2 columns
        if numeric_df.shape[1] >= 2:

            #select target column
            target_column = st.selectbox(
                "Select Target Column",
                numeric_df.columns
            )

            #select feature columns
            feature_columns = st.multiselect(
                "Select Feature Columns",
                [col for col in numeric_df.columns if col != target_column],
                default=[col for col in numeric_df.columns if col != target_column]
            )

            if len(feature_columns) == 0:
                st.warning("Please select at least one feature column.")
                st.stop()

            #feature column
            x = numeric_df[feature_columns]

            #target column
            y = numeric_df[target_column]

            st.write("Feature (x):")
            st.write(x)

            st.write("Target (y):")
            st.write(y)

            # Split dataset into training and testing data
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42
            )

            # choose problem type
            problem_type = st.selectbox(
                "Select Problem Type",
                ["Regression", "Classification"]
            )

            # model comparison based on problem type
            if problem_type == "Regression":
                models = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
                    "Random Forest Regressor": RandomForestRegressor(random_state=42)
                }

                results = []

                for name, model_obj in models.items():
                    model_obj.fit(x_train, y_train)
                    predictions = model_obj.predict(x_test)
                    mse = mean_squared_error(y_test, predictions)
                    r2 = r2_score(y_test, predictions)

                    results.append({
                        "Model": name,
                        "MSE": mse,
                        "R2 Score": r2
                    })

                results_df = pd.DataFrame(results)

                st.write("Model Comparison")
                st.write(results_df)

                best_model_name = results_df.loc[results_df["MSE"].idxmin(), "Model"]
                st.success(f"Best Model: {best_model_name}")

            else:
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
                    "Random Forest Classifier": RandomForestClassifier(random_state=42)
                }

                results = []

                for name, model_obj in models.items():
                    model_obj.fit(x_train, y_train)
                    predictions = model_obj.predict(x_test)
                    acc = accuracy_score(y_test, predictions)

                    results.append({
                        "Model": name,
                        "Accuracy": acc
                    })

                results_df = pd.DataFrame(results)

                st.write("Model Comparison")
                st.write(results_df)

                best_model_name = results_df.loc[results_df["Accuracy"].idxmax(), "Model"]
                st.success(f"Best Model: {best_model_name}")


            # choose algorithm based on problem type
            if problem_type == "Regression":
                algorithm = st.selectbox(
                    "Choose Machine Learning Algorithm",
                    [
                        "Linear Regression",
                        "Decision Tree Regressor",
                        "Random Forest Regressor"
                    ]
                )
            else:
                algorithm = st.selectbox(
                    "Choose Machine Learning Algorithm",
                    [
                        "Logistic Regression",
                        "Decision Tree Classifier",
                        "Random Forest Classifier"
                    ]
                )

            #Create model based on selction
            if algorithm == "Linear Regression":
                model = LinearRegression()

            elif algorithm == "Decision Tree Regressor":
                model = DecisionTreeRegressor(random_state=42)

            elif algorithm == "Random Forest Regressor":
                model = RandomForestRegressor(random_state=42)

            elif algorithm == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)

            elif algorithm == "Decision Tree Classifier":
                model = DecisionTreeClassifier(random_state=42)

            else:
                model = RandomForestClassifier(random_state=42)

            # Train model with loading spinner
            with st.spinner("Training AI Model..."):
                model.fit(x_train, y_train)

            st.success(f"{algorithm} Model Trained Successfully!")

            # Save model
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, "models/trained_model.pkl")

            # Download trained model
            with open("models/trained_model.pkl", "rb") as file:
                st.download_button(
                    label="Download Trained Model",
                    data=file,
                    file_name="trained_model.pkl",
                    mime="application/octet-stream"
                )

            # Make predictions on test data
            predictions = model.predict(x_test)

            if problem_type == "Regression":
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)

                st.write("Mean Squared Error:")
                st.write(mse)

                st.write("R² Score:")
                st.write(r2)

                st.write("Actual vs Predicted Values")

                fig, ax = plt.subplots()
                ax.scatter(y_test, predictions, alpha=0.5)
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title("Actual vs Predicted Values")
                st.pyplot(fig)

            else:
                acc = accuracy_score(y_test, predictions)

                st.write("Accuracy:")
                st.write(acc)

                st.write("Confusion Matrix")

                cm = confusion_matrix(y_test, predictions)

                fig, ax = plt.subplots(figsize=(6, 4))

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    ax=ax
                )

                ax.set_xlabel("Predicted Values")
                ax.set_ylabel("Actual Values")
                ax.set_title("Confusion Matrix")

                st.pyplot(fig)

            #user Input Prediction System
            inputs = []

            for col in x.columns:
                value = st.number_input(col)
                inputs.append(value)

            #Predict button
            if st.button("Predict"):
                
                #create new input data
                new_data = [inputs]

                #load saved model
                loaded_model = joblib.load("models/trained_model.pkl")

                #make prediction
                result = loaded_model.predict(new_data)

                #Display reslut
                if problem_type == "Regression":
                    st.success(f"Predicted {target_column}: {result[0]:.2f}")
                else:
                    st.success(f"Predicted {target_column}: {result[0]}")

                st.balloons()

                #create prediction dataframe and provide download option-
                prediction_df = pd.DataFrame({
                    "Target": [target_column],
                    "Predicted Value": [result[0]]
                })

                csv = prediction_df.to_csv(index=False)

                st.download_button(
                    label="Download Prediction Result",
                    data=csv,
                    file_name="prediction_result.csv",
                    mime="text/csv"
                )

        else:
            st.warning("Dataset must contain at least 2 numerical columns for machine learning.")

    with tab3:
        #display a bar chart of the numerical columns in the dataset using the bar_chart() method of streamlit
        st.write("Bar Chart")

        # st.line_chart(df.select_dtypes(include='number'))
        st.bar_chart(df.select_dtypes(include="number"))

        # Correlation Heatmap
        st.write("Correlation Heatmap")

        correlation = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

else:
    st.info("Please upload a CSV file from the sidebar to start.")