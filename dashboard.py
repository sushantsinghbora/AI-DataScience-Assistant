import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io #for handling file uploads/io handles text streams or buffers
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

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

        with tab2:
            # Machine Learning Section 
            st.write("Machine Learning Model")


            #check if dataset has at least 2 columns
            if numeric_df.shape[1] >= 2:

                #feature column
                x = numeric_df.iloc[:,:-1]
                #target column
                y = numeric_df.iloc[:,-1]

                st.write("Feature (x):")
                st.write(x)
                st.write("Target (y):")
                st.write(y)


                # Split dataset into training and testing data
                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, test_size=0.2, random_state=42
                )

                #select model
                algorithm = st.selectbox(
                    "Choose Machine Learning Algorithm",
                    [
                        "Linear Regression",
                        "Decision Tree Regressor",
                        "Random Forest Regressor"
                    ]
                )

                #Create model based on selction
                if algorithm == "Linear Regression":
                    model = LinearRegression()
                elif algorithm == "Decision Tree Regressor":
                    model = DecisionTreeRegressor()
                else:
                    model = RandomForestRegressor(random_state=42)

                # Train model with loading spinner
                with st.spinner("Training AI Model..."):
                    model.fit(x_train, y_train)
                st.success(f"{algorithm} Model Trained Successfully!")
                

                # Save model
                joblib.dump(model, "trained_model.pkl")


                # Make predictions on test data
                predictions = model.predict(x_test)

                # Calculate and display mean squared error
                mse = mean_squared_error(y_test, predictions)
                st.write("Mean Squared Error:")
                st.write(mse)


                # Accuracy Visualization
                st.write("Actual vs Predicted Values")
                fig, ax = plt.subplots()
                ax.scatter(y_test, predictions,alpha=0.5)
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title("Actual vs Predicted Values")
                st.pyplot(fig)


                #user Input Prediction System
                st.write("Enter New Student Data")

                study_hours = st.number_input("Study Hours")
                attendance = st.number_input("Attendance")
                assignments = st.number_input("Assignment Score")

                #Predict button
                if st.button("Predict"):
                    #create new input data
                    new_data = [[study_hours, attendance, assignments]]
                    #load saved model
                    loaded_model = joblib.load("trained_model.pkl")
                    #make prediction
                    result = loaded_model.predict(new_data)
                    #Display reslut
                    st.success(f"Predicted Marks: {result[0]:.2f}")
                    st.balloons()