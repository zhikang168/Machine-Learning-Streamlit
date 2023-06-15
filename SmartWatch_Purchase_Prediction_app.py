# Importing the Streamlit library and pandas library
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, chi2
import seaborn as sns

# Create a global variable to store the data
gb = st.session_state


# Sidebar navigation
st.sidebar.title("Navigation")
options = ["Exploratory Data Analysis (EDA)", "Machine Learning Algorithm : Random Forest", "Machine Learning Algorithm : Naive Bayes"]
selected_option = st.sidebar.radio("Go to", options)

# EDA (Exploratory Data Analysis)
if selected_option == "Exploratory Data Analysis (EDA)":

    
    st.title("Smart Watch Purchase Dataset")
    st.markdown("***")
    st.subheader("Exploratory Data Analysis (EDA)")


    # Read CSV file and cache the result 
    st.text("Read Dataset")
    @st.cache_data
    def read_data(file_path):
        df = pd.read_csv(file_path)
        return df

    file_path = "SmartWatchPurchase.csv"  
    gb['df'] = read_data(file_path)

    # print the dataset
    st.write(gb['df'])

    # Drop column
    st.markdown("***")
    st.text("Drop Column")
    # get column names as a list
    columns = gb['df'].columns.tolist()

    # drop columns (hour)
    columns_to_drop = ['hour']
    gb['df_drop'] = gb['df'].drop(columns_to_drop, axis=1)
    st.write(gb['df_drop'])

    # Identify Missing Value
    st.markdown("***")
    st.text("Identify Missing Value")

    # Count the number of missing values in each column
    missing_values = gb['df_drop'].isnull().sum()

    # Filter columns with missing values
    missing_values = missing_values[missing_values != 0]

    if missing_values.empty:
         st.write("No missing values found.")
    else:
        st.write("Missing values:")
        st.write(missing_values)

    # Delete Missing Value
    st.markdown("***")
    st.text("Deleting Missing Value")
    gb['df_new'] = gb['df_drop'].dropna()
    st.write(gb['df_new'])

    # Categorical Conversion
    st.markdown("***")
    st.text("Categorical Conversion")

    # Mapping "female" to 0 and "male" to 1
    gb['df_new']['gender'] = gb['df_new']['gender'].replace({'female': 0, 'male': 1})

    # Mapping "single" to 0 , "married" to 1 and "divorced" to 2
    gb['df_new']['maritalStatus'] = gb['df_new']['maritalStatus'].replace({'single': 0, 'married': 1,'divorced': 2})
    
    # Mapping "False" to 0 and "True" to 1
    gb['df_new']['weekend'] = gb['df_new']['weekend'].replace({'FALSE': 0, 'TRUE': 1})

    # Mapping "no" to 0 and "yes" to 1
    gb['df_new']['buySmartWatch'] = gb['df_new']['buySmartWatch'].replace({'no': 0, 'yes': 1})

    st.write(gb['df_new'])

    st.markdown("***")

    # Graph Visualization
    st.subheader("Graph Visualization")
    # Scatter Plot
    st.text("Scatter Plot: Age vs Income")
    fig, ax = plt.subplots()
    ax.scatter(gb['df_new']['age'], gb['df_new']['income'])
    ax.set_xlabel("Age")
    ax.set_ylabel("Income")
    st.pyplot(fig)

    st.text("")
    # Bar chart 
    st.text("Bar chart: Age vs Income by Gender")
    # Filter data for male gender
    male_data = gb['df_new'][gb['df_new']['gender'] == 1] 
    # Filter data for female gender 
    female_data = gb['df_new'][gb['df_new']['gender'] == 0] 

    fig, ax = plt.subplots()
    ax.bar(male_data['age'], male_data['income'], color='blue', alpha=0.7, label='Male')
    ax.bar(female_data['age'], female_data['income'], color='red', alpha=0.7, label='Female')
    ax.set_xlabel("Age")
    ax.set_ylabel("Income")
    ax.legend()
    st.pyplot(fig)



# Algorithm
elif selected_option == "Machine Learning Algorithm : Random Forest":
    st.title("Machine Learning Algorithm: Random Forest")

    # Get data from Exploratory Data Analysis (EDA) page to Machine Learning Algorithm page
    if 'df_new' in gb:
        df_new2 = gb['df_new']  # Retrieve the DataFrame from the session state

    st.markdown("***")
    st.subheader("Random Forest")

    # Prepare the data
    X = df_new2[['age', 'income', 'gender', 'maritalStatus', 'weekend']]
    y = df_new2['buySmartWatch']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_rf = rf.predict(X_test)

    # Calculate accuracy, precision, recall, and F1-score
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    precision_rf = precision_score(y_test, y_pred_rf)
    recall_rf = recall_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)

    # Calculate confusion matrix
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()

    # Display Random Forest results
    st.write("**Random Forest Results:**")
    st.write("Accuracy:", accuracy_rf)
    st.write("Precision:", precision_rf)
    st.write("Recall:", recall_rf)
    st.write("F1-Score:", f1_rf)
    st.write("")
    st.write("**Confusion Matrix:**")
    st.write(cm_rf.astype(float))
    st.write("TN:", tn_rf, "FP:", fp_rf)
    st.write("FN:", fn_rf, "TP:", tp_rf)

    # Prediction
    st.markdown("***")
    st.subheader("Prediction")

    # Age input
    age = st.slider("Age:", min_value=18, max_value=80, value=18)
    st.write("")

    # Income input
    income = st.text_input("Income:", value="10000")
    st.write("")

    # Marital status input
    maritalStatus = st.radio("Marital Status:", ("single", "married", "divorced"))
    st.write("")

    # Create two columns
    col1, col2 = st.columns(2)

    # Gender input in the left column
    with col1:
        gender = st.radio("Gender:", ("male", "female"))

    # Weekend input in the right column
    with col2:
        weekend = st.radio("Is the purchase on the weekend:", ("Yes", "No"))

    # Convert gender input to numeric value
    gender_value = 1 if gender == "male" else 0

    # Convert marital status input to numeric value
    if maritalStatus == "single":
        maritalStatus_value = 0
    elif maritalStatus == "married":
        maritalStatus_value = 1
    else:  # divorced
        maritalStatus_value = 2

    # Convert weekend input to numeric value
    weekend_value = 1 if weekend == "Yes" else 0

    # Create a new sample DataFrame for prediction
    sample_data = pd.DataFrame({
        "age": [age],
        "income": [float(income)],
        "gender": [gender_value],
        "maritalStatus": [maritalStatus_value],
        "weekend": [weekend_value]
    })

    # Probability
    st.subheader("Probability")
    probabilities = rf.predict_proba(sample_data)
    probability_not_buy = probabilities[0][0]
    probability_buy = probabilities[0][1]

    prob = pd.DataFrame({"Probability": [probability_not_buy, probability_buy]})
    st.write(prob)
    st.write("0 = not purchase, 1 = purchase")

    st.subheader("Output")

    # Make predictions on the sample data
    prediction = rf.predict(sample_data)

    if prediction[0] == 0:
        result = "**Not purchase smartwatch.**"
    else:
        result = "**Purchase smartwatch.**"

    st.write("Based on the given inputs, the prediction is:", result)

    # Feature Selection
    st.markdown("***")
    st.subheader("Feature Selection")

    # Get feature importances from the Random Forest model
    feature_importances = rf.feature_importances_

    # Create a DataFrame to display feature importances
    feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importances})
    feature_importance_df = feature_importance_df.sort_values("Importance", ascending=False)

    # Display the feature importances
    st.write("**Feature Importances:**")
    st.dataframe(feature_importance_df)

    # Plot the feature importances
    st.write("**Feature Importances Plot:**")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importances")
    st.pyplot(plt)



# Machine Learning Algorithm : SVM
elif selected_option == "Machine Learning Algorithm : Naive Bayes":
    st.subheader("Machine Learning Algorithm : Naive Bayes")
   

     # Get data from Exploratory Data Analysis (EDA) page to Machine Learning Algorithm page
    if 'df_new' in st.session_state:
        df_new2 = st.session_state['df_new']  # Retrieve the DataFrame from the session state

    st.markdown("***")
    st.subheader("Naive Bayes")

    # Prepare the data
    X = df_new2[['age', 'income', 'gender', 'weekend','maritalStatus']]
    y = df_new2['buySmartWatch']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Naive Bayes Classifier
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_nb = nb.predict(X_test)

    # Calculate accuracy, precision, recall, and F1-score
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    precision_nb = precision_score(y_test, y_pred_nb)
    recall_nb = recall_score(y_test, y_pred_nb)
    f1_nb = f1_score(y_test, y_pred_nb)

    # Calculate confusion matrix
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    tn_nb, fp_nb, fn_nb, tp_nb = cm_nb.ravel()

    # Display Naive Bayes results
    st.write("**Naive Bayes Results:**")
    st.write("Accuracy:", accuracy_nb)
    st.write("Precision:", precision_nb)
    st.write("Recall:", recall_nb)
    st.write("F1-Score:", f1_nb)
    st.write("")
    st.write("**Confusion Matrix:**")
    st.write(cm_nb.astype(float))
    st.write("TN:", tn_nb, "FP:", fp_nb)
    st.write("FN:", fn_nb, "TP:", tp_nb)

    # Prediction
    st.markdown("***")
    st.subheader("Prediction")

    # Age input
    age = st.slider("Age:", min_value=18, max_value=80, value=18)
    st.write("")

    # Income input
    income = st.text_input("Income:", value="10000")
    st.write("")

    #maritalStatus input
    maritalStatus = st.radio("maritalStatus:", ("single", "married", "divorced"))
    st.write("")

    # create two columns
    col1, col2 = st.columns(2)

    # Gender input in the left column
    with col1:
        gender = st.radio("Gender:", ("male", "female"))

    # Weekend input in the right column
    with col2:
        weekend = st.radio("Is that purchase at weekend:", ("Yes", "No"))

    # Convert gender input to numeric value
    gender_value = 1 if gender == "male" else 0

    # Convert maritalStatus input to numeric value
    if maritalStatus == "single":
        maritalStatus_value = 0
    elif maritalStatus == "married":
        maritalStatus_value = 1
    else:  # divorced
        maritalStatus_value = 2

    # Convert weekend input to numeric value
    weekend_value = 1 if weekend == "Yes" else 0

    # Create a new sample DataFrame for prediction
    sample_data = pd.DataFrame({"age": [age], "income": [float(income)], "gender": [gender_value], "weekend": [weekend_value],"maritalStatus": [maritalStatus_value]})

    # Probability
    st.subheader("Probability")
    probabilities = nb.predict_proba(sample_data)
    probability_buy = probabilities[0][1]
    probability_not_buy = probabilities[0][0]

    prob = pd.DataFrame({"Probability": [probability_not_buy, probability_buy]})
    st.write(prob)
    st.write("0 = not purchase , 1 = purchase")

    st.subheader("Output")

    # Make predictions on the sample data
    prediction = nb.predict(sample_data)

    if prediction[0] == 0:
        result = "**Not purchase smartwatch.**"
    else:
        result = "**Purchase smartwatch.**"

    st.write("Based on the given inputs, the prediction is:", result)


    # Feature Selection
    st.markdown("***")
    st.subheader("Feature Selection")

    # Fit the classifier and perform feature selection
    selector = SelectKBest(score_func=chi2, k='all')
    X_new = selector.fit_transform(X, y)

    # Get the selected feature scores
    feature_scores = selector.scores_

    # Calculate the importance percentage for each feature
    total_score = sum(feature_scores)
    feature_importances = [(score / total_score) * 100 for score in feature_scores]

    # Create a DataFrame to display the features and their importance percentages
    features_df = pd.DataFrame({"Feature": X.columns, "Importance Percentage": feature_importances})
    st.write("**Feature Importance:**")
    st.dataframe(features_df)