
# This cell writes the Streamlit app code into a file named 'app.py'

import streamlit as st
import joblib # Changed from pickle to joblib for compressed model loading
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.express as px # Added for Plotly charts

# Load the trained model and scaler
try:
    # Assuming rfc_tune_model.joblib is the final chosen model
    model = joblib.load('rfc_tune_model.joblib') # Load model using joblib
    scaler = joblib.load('scaler.joblib') # Load scaler using joblib
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'rfc_tune_model.joblib' and 'scaler.joblib' are in the same directory.")
    st.stop()

# Define the expected columns for the model input (X) in the exact order
model_columns = [
    'age', 'capital_gain', 'capital_loss', 'hours_per_week',
    'workclass_Unknown', 'education_11th', 'education_12th', 'education_1st-4th',
    'education_5th-6th', 'education_7th-8th', 'education_9th', 'education_Assoc-acdm',
    'education_Assoc-voc', 'education_Bachelors', 'education_Doctorate',
    'education_HS-grad', 'education_Masters', 'education_Preschool',
    'education_Prof-school', 'education_Some-college', 'marital_status_not married',
    'occupation_Armed-Forces', 'occupation_Craft-repair', 'occupation_Exec-managerial',
    'occupation_Farming-fishing', 'occupation_Handlers-cleaners',
    'occupation_Machine-op-inspct', 'occupation_No-occupation', 'occupation_Other-service',
    'occupation_Priv-house-serv',
    'occupation_Prof-specialty',
    'occupation_Protective-serv', 'occupation_Sales', 'occupation_Tech-support',
    'occupation_Transport-moving', 'occupation_Unknown', 'relationship_Not-in-family',
    'relationship_Other-relative', 'relationship_Own-child', 'relationship_Unmarried',
    'relationship_Wife', 'race_Asian-Pac-Islander', 'race_Black', 'race_Other',
    'race_White', 'sex_Male', 'native_country_United-States'
]

# Numerical columns that need scaling
numerical_cols = ['age', 'capital_gain', 'capital_loss', 'hours_per_week']

# Categorical columns used for one-hot encoding (original column names)
categorical_cols = [
    'workclass', 'education', 'marital_status', 'occupation',
    'relationship', 'race', 'sex', 'native_country'
]

st.set_page_config(layout="wide") # Use wide layout for better space utilization

# --- Image placement ---
top_left_col, top_right_col = st.columns([4, 1])
with top_right_col:
    st.image('image1.png', width=400) # Adjust width as needed

st.title("📈 Income Prediction Dashboard")
st.markdown("""
Welcome to the Income Prediction Dashboard! Provide an individual's characteristics below,
and this application will predict whether their annual income is **>50K** or **<=50K**.
""")
st.markdown('---')

st.header("👤 Individual Characteristics")
st.markdown("Adjust the values and select the options below to describe the individual:")

# --- Logic to sort features by importance for UI display ---
# This block calculates the importance of original features and sorts them.
if model is not None:
    feature_importances = model.feature_importances_
    feature_importance_series = pd.Series(feature_importances, index=model_columns)

    original_feature_max_importance = {}
    all_original_features = categorical_cols + numerical_cols

    for original_col in all_original_features:
        max_imp = 0.0
        # Check if the original column name itself is present in the model_columns (e.g., numerical, or single-dummy for binary)
        if original_col in feature_importance_series.index:
            max_imp = max(max_imp, feature_importance_series[original_col])
        
        # Check for one-hot encoded columns that start with the original column name
        for encoded_col in model_columns:
            if encoded_col.startswith(f'{original_col}_'):
                max_imp = max(max_imp, feature_importance_series.get(encoded_col, 0.0))
        
        original_feature_max_importance[original_col] = max_imp

    # Ensure all original features are accounted for, even if they had zero importance or no direct encoded column with importance
    for of in all_original_features:
        if of not in original_feature_max_importance:
            original_feature_max_importance[of] = 0.0 # Assign 0 if no importance found

    sorted_original_features_with_importance = sorted(original_feature_max_importance.items(), key=lambda item: item[1], reverse=True)
    sorted_original_feature_names = [item[0] for item in sorted_original_features_with_importance]
else:
    # Fallback order if model is not loaded (should not happen normally)
    sorted_original_feature_names = ['age', 'workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']

def user_input_features():
    user_inputs = {}

    # Dictionary of lambda functions to create each widget, mapping original feature name to its widget creator
    widget_definitions = {
        'age': lambda: st.slider('Age', 17, 90, 30),
        'workclass': lambda: st.selectbox('Workclass', ['Private', 'Self-emp-not-inc', 'Local-gov', 'Unknown', 'State-gov', 'Self-emp-inc', 'Federal-gov', 'Without-pay', 'Never-worked']),
        'education': lambda: st.selectbox('Education', ['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Assoc-voc', '11th', 'Assoc-acdm', '10th', '7th-8th', '9th', 'Prof-school', '12th', 'Doctorate', '5th-6th', '1st-4th', 'Preschool']),
        'marital_status': lambda: st.selectbox('Marital Status', ['not married', 'married']),
        'occupation': lambda: st.selectbox('Occupation', ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Unknown', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces', 'No-occupation']),
        'relationship': lambda: st.selectbox('Relationship', ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative']),
        'race': lambda: st.selectbox('Race', ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']),
        'sex': lambda: st.selectbox('Sex', ['Male', 'Female']),
        'capital_gain': lambda: st.number_input('Capital Gain', 0, 99999, 0),
        'capital_loss': lambda: st.number_input('Capital Loss', 0, 4356, 0),
        'hours_per_week': lambda: st.slider('Hours per Week', 1, 99, 40),
        'native_country': lambda: st.selectbox('Native Country', ['United-States', 'Other'])
    }

    # Create widgets in the sorted order of importance, arranging them in columns
    num_cols_per_row = 2 # Number of input columns per row for better layout
    cols = st.columns(num_cols_per_row) # Create initial columns for the first row

    col_index = 0
    for feature_name in sorted_original_feature_names:
        if feature_name in widget_definitions: # Ensure the feature has a defined widget
            with cols[col_index]:
                user_inputs[feature_name] = widget_definitions[feature_name]()
            col_index = (col_index + 1) % num_cols_per_row
            if col_index == 0: # If the current row is full, create a new set of columns
                cols = st.columns(num_cols_per_row)

    features = pd.DataFrame(user_inputs, index=[0])
    return features

input_df_raw = user_input_features()

# Preprocessing user input to match model's training data
def preprocess_input_for_model(df_raw):
    # Create a DataFrame with all expected model columns, initialized to 0
    processed_input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    # Populate numerical features from raw input
    for col in numerical_cols:
        if col in df_raw.columns:
            processed_input_df[col] = df_raw[col].iloc[0]

    # Process categorical features to generate dummy variables
    for cat_col in categorical_cols:
        if cat_col in df_raw.columns:
            value = df_raw[cat_col].iloc[0]

            # Specific handling for marital_status and native_country which were pre-transformed
            # and then implicitly one-hot encoded (with drop_first=True logic)
            if cat_col == 'marital_status':
                if value == 'not married' and 'marital_status_not married' in processed_input_df.columns:
                    processed_input_df['marital_status_not married'] = 1
            elif cat_col == 'native_country':
                if value == 'United-States' and 'native_country_United-States' in processed_input_df.columns:
                    processed_input_df['native_country_United-States'] = 1
            elif cat_col == 'sex': # Handle sex_Male specifically since it's a direct dummy if 'sex_Female' was dropped
                if value == 'Male' and 'sex_Male' in processed_input_df.columns:
                    processed_input_df['sex_Male'] = 1
            else:
                # For other categorical columns that were one-hot encoded with drop_first=True (or specific drops)
                # We construct the dummy column name and set to 1 if it exists in model_columns
                dummy_col_name = f"{cat_col}_{value}"
                if dummy_col_name in processed_input_df.columns:
                    processed_input_df[dummy_col_name] = 1

    # Apply scaling to numerical features
    processed_input_df[numerical_cols] = scaler.transform(processed_input_df[numerical_cols])

    return processed_input_df

# --- Custom CSS for button styling ---
st.markdown("""
<style>
div.stButton > button {
    background-color: #4CAF50; /* Green background */
    color: white;
    padding: 15px 30px; /* Bigger padding */
    font-size: 20px; /* Larger font size */
    font-weight: bold; /* Bolder text */
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
div.stButton > button:hover {
    background-color: #45a049; /* Darker green on hover */
}
</style>
""", unsafe_allow_html=True)

# --- Centered 'Predict Income' button ---
col1_btn, col2_btn, col3_btn = st.columns([3, 2, 2]) # Adjust ratio for centering
with col2_btn:
    predict_clicked = st.button('Predict Income', type="primary", help="Click to get the income prediction based on the entered characteristics")

if predict_clicked:
    processed_input = preprocess_input_for_model(input_df_raw)
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)

    st.subheader('🚀 Prediction Results')
    st.markdown('---') 

    col1, col2 = st.columns(2) 

    with col1:
        st.write("### Predicted Income Level:")
        if prediction[0] == 1:
            st.success("**Income >50K (High Income) 🎉**")
        else:
            st.info("**Income <=50K (Low Income) 📉**")

    with col2:
        st.write("### Prediction Confidence:")
        st.metric(label="Probability of <=50K", value=f"{prediction_proba[0][0]:.2%}")
        st.metric(label="Probability of >50K", value=f"{prediction_proba[0][1]:.2%}")

    st.markdown('---') 
else:
    st.info(" Adjust the features above and click 'Predict Income' to see the results! 👆 ")



# --- Feature Importance Plot ---
st.subheader('📊 Feature Importance from Random Forest Classifier')
st.markdown("""
**Random Forest Classifier with Hyperparameter Tuning (Random Over Sampler & Randomized Search CV Method)**
""")
st.markdown("""
This interactive chart reveals which features had the most significant impact on the model's income predictions.
A higher 'Importance' score indicates a greater influence on the outcome.
""")

if model is not None:
    # Get feature importances from the loaded model
    feature_importances = model.feature_importances_

    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': model_columns,
        'Importance': feature_importances
    })

    # Sort by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Create interactive bar chart using Plotly Express
    fig = px.bar(importance_df.head(20), # Display top 20 features for clarity
                 x='Importance',
                 y='Feature',
                 orientation='h',
                 title='Top 20 Most Important Features',
                 labels={'Importance': 'Feature Importance Score', 'Feature': 'Feature Name'},
                 height=600,
                 color='Importance',
                 color_continuous_scale=px.colors.sequential.Viridis) # Use a nice color scale

    fig.update_layout(yaxis={'categoryorder':'total ascending'}) # Order features from most to least important

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Model not loaded. Cannot display feature importance.")

st.markdown('---')
st.caption("Developed with Streamlit and Scikit-learn by Dataverse Team DS55. Data Source: https://www.kaggle.com/code/tawfikelmetwally/census-income-analysis-and-modeling/input") # Footer