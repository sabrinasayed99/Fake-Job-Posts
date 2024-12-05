import streamlit as st
import pandas as pd
from joblib import load
import plotly.express as px
import numpy as np
import re
import nltk
import shap
import scipy.sparse
import lime
import lime.lime_tabular
from joblib import load
import unicodedata
import streamlit as st
import joblib
from text_processor import TextProcessor  
import os
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(layout="wide",
                   page_title="Fake Job Post Detector",
                   page_icon=":rotating_light:",
                   )

# Load the processor
@st.cache_resource
def load_processor():
    try:
        return joblib.load('text_processor.joblib')
    except Exception as e:
        st.error(f"Error loading text processor: {str(e)}")
        return None

@st.cache_resource
def load_model_and_metadata():
    try:
        MODEL_DIR = 'Models'  # relative to your app.py location
        pipeline = load(os.path.join(MODEL_DIR, 'best_model_pipeline.joblib'))
        metadata = load(os.path.join(MODEL_DIR, 'best_model_metadata.joblib'))
        shap_data = load(os.path.join(MODEL_DIR, 'shap_values.joblib'))
        lime_data = load(os.path.join(MODEL_DIR, 'lime_examples.joblib'))
        feature_importances = pd.read_csv(os.path.join(MODEL_DIR, 'feature_importances.csv'))
        training_data = load(os.path.join(MODEL_DIR, 'X_train.joblib'))
        
        # Verify data source
        if metadata['data_source'] != 'Final_Cleaned_Data.csv':
            st.warning("Warning: Model might not be using the latest data!")
            
        return pipeline, metadata, shap_data, lime_data, feature_importances, training_data
        
    except Exception as e:
        st.error(f"Error loading model artifacts: {str(e)}")
        return None, None, None, None, None, None
  
def get_lime_explanation(input_data, pipeline, feature_names, X_train):
    # Create LIME explainer
    try:
        # Ensure input_data is a DataFrame with correct columns
        if not isinstance(input_data, pd.DataFrame):
            input_data = pd.DataFrame(input_data, columns=feature_names)
        else:
            # Ensure columns are in the correct order
            input_data = input_data[feature_names]

        # Ensure X_train is a DataFrame
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=feature_names)
        else:
            # Ensure columns are in the correct order
            X_train = X_train[feature_names]

        print(f"Debug - Pre-transform shapes:")
        print(f"Input data: {input_data.shape}")
        print(f"Training data: {X_train.shape}")

        # Transform input data
        input_transformed = pipeline.named_steps['preprocessor'].transform(input_data)
        training_data_transformed = pipeline.named_steps['preprocessor'].transform(X_train)

        # Convert to dense arrays if sparse
        if scipy.sparse.issparse(input_transformed):
            input_transformed = input_transformed.toarray()
        if scipy.sparse.issparse(training_data_transformed):
            training_data_transformed = training_data_transformed.toarray()

        print(f"Debug - Post-transform shapes:")
        print(f"Transformed input: {input_transformed.shape}")
        print(f"Transformed training: {training_data_transformed.shape}")

        # Get feature names after transformation
        if hasattr(pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
            transformed_feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        else:
            transformed_feature_names = [f"feature_{i}" for i in range(input_transformed.shape[1])]

        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data_transformed,
            feature_names=transformed_feature_names,
            class_names=['Legitimate', 'Fraudulent'],
            mode='classification'
        )
        
        # Get explanation for the first instance
        exp = explainer.explain_instance(
            input_transformed[0],
            pipeline.named_steps['classifier'].predict_proba,
            num_features=10
        )

        # Create DataFrame for visualization
        importance_df = pd.DataFrame(
            exp.as_list(), 
            columns=['feature', 'importance']
        ).sort_values('importance', ascending=True)
        
        # Create visualization
        fig = go.Figure(data=[
            go.Bar(
                x=importance_df['importance'],
                y=importance_df['feature'],
                orientation='h'
            )
        ])
    
        fig.update_layout(
            title='Feature Importance for This Prediction',
            height=400,
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10)
        )

        return exp, fig, importance_df
    
    except Exception as e:
        st.error(f"Error generating LIME explanation: {str(e)}")
        print(f"Detailed error: {str(e)}")
        print(f"Debug - Input data shape: {input_data.shape}")
        print(f"Debug - Input data columns: {input_data.columns.tolist()}")
        print(f"Debug - X_train shape: {X_train.shape}")
        return None, None, None
    

# Function to get predictions from pipeline
def make_prediction(input_data):
    """Make predictions using the loaded model"""
    try:
        # Define field types
        text_fields = {'title', 'requirements','company_profile',  
                       'description', 'benefits'}
                      
        categorical_fields = {'department', 'employment_type', 'industry','function', 
                              'required_education', 'required_experience'}
        
        numeric_fields = {'has_company_logo', 'has_questions', 'telecommuting'}
        

         # Set appropriate defaults for missing fields
        for field in text_fields:
            if field not in input_data:
                input_data[field] = ''  # Empty string for text processing
                
        for field in categorical_fields:
            if field not in input_data:
                input_data[field] = 'unknown'  # Empty string for categorical features
                
        for field in numeric_fields:
            if field not in input_data:
                input_data[field] = 0  # No/False for binary features
        

        # Process the input using the text processor
        processed_features = processor.process_job_posting(input_data)
        
        # Ensure processed_features is a DataFrame with the correct columns
        expected_columns = (
            list(text_fields) +  # Text fields
            list(numeric_fields) +  # Numerical features
            list(categorical_fields) +  # Categorical features
            ['description_length'] +  # Description length
            # Fraud indicators
            [f'{field}_{indicator}' for field in text_fields 
             for indicator in ['urgency_score', 'guarantee_score', 'pressure_score',
                             'excessive_punctuation', 'all_caps_words']]
        )
        
        # Verify we have all expected columns
        missing_columns = set(expected_columns) - set(processed_features.columns)
        if missing_columns:
            raise ValueError(f"Missing expected columns: {missing_columns}")
        
        # Ensure columns are in the correct order
        processed_features = processed_features[expected_columns]
        
        # Debug print
        print("Processed features columns:", processed_features.columns)
        print("Number of features:", len(processed_features.columns))


        # Make prediction
        prediction = pipeline.predict(processed_features)
        probability = pipeline.predict_proba(processed_features)[0]
        
        # Get LIME explanation
        lime_exp, lime_fig, lime_df = get_lime_explanation(
            processed_features,  # Already a DataFrame with correct columns
            pipeline, 
            expected_columns,  # Pass the ordered column names
            training_data
        )
        
        return {
            "Prediction": "Fraudulent" if prediction[0] == 1 else "Legitimate",
            "Probability": probability,
            "LIME_explanation": lime_exp,
            "LIME_visualization": lime_fig,
            "LIME_data": lime_df
        }
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        print(f"Detailed error: {str(e)}")
        return None

# Initialize processor and model
processor = load_processor()
pipeline, metadata, shap_data, lime_data, feature_importances, training_data = load_model_and_metadata()


# Streamlit app layout
st.title(":rotating_light: Fake Job Post Detector :rotating_light:")
st.markdown("**Real or Fake?** This tool helps you determine the authenticity of job postings.")

if pipeline is not None:
    with st.form("job_post_form"):
        st.subheader("Insert Job Post Details")
        col1,col2 = st.columns(2)

        with col1:
            title = st.text_input("Title",placeholder="e.g., Senior Software Engineer")
            department = st.text_input("Department",placeholder="e.g., Engineering")
            employment_type= st.text_input("Employment Type", placeholder="e.g., Full-Time")
            industry = st.text_input("Industry",placeholder="e.g., Technology")
            function = st.text_input("Job Function",placeholder="e.g., Software Development")
        
        with col2:
            required_experience = st.text_input("Required Experience",
                                              placeholder="e.g., 5+ years")
            required_education = st.text_input("Required Education",
                                             placeholder="e.g., Bachelor's Degree")
            has_company_logo = st.checkbox("Has Company Logo")
            has_questions = st.checkbox("Has Screening Questions")
            telecommuting = st.checkbox("Telecommuting")

        # Detailed Descriptions
        st.subheader("Detailed Descriptions")
        
        company_profile = st.text_area("Company Description",
                                     placeholder="Tell us about the company...",
                                     height=150)
        
        description = st.text_area("Job Description*",
                                 placeholder="Details about the role and responsibilities...",
                                 height=200)
        
        requirements = st.text_area("Requirements",
                                  placeholder="Required skills and qualifications...",
                                  height=150)
        
        benefits = st.text_area("Benefits",
                              placeholder="Compensation and benefits offered...",
                              height=150)
        
        st.markdown("*Required fields")
        submit_button = st.form_submit_button("Check Job Post")

        if submit_button:
            # Validate required fields
            if not title.strip() or not description.strip():
                st.error("Please fill in all required fields (Job Title and Job Description)")
            else:
                # Create input dictionary
                input_data = {
                    "title": title,
                    "department": department,
                    "company_profile": company_profile,
                    "description": description,
                    "requirements": requirements,
                    "benefits": benefits,
                    "telecommuting": int(telecommuting),
                    "employment_type": employment_type,
                    "required_experience": required_experience,
                    "required_education": required_education,
                    "industry": industry,
                    "function": function,
                    "has_company_logo": int(has_company_logo),
                    "has_questions": int(has_questions)
                }

                # Process the input using the processor
                with st.spinner("Analyzing post..."):
                    result = make_prediction(input_data)
                    
                # Display results
                if result:
                    st.success("Analysis Complete!")

                    col1, col2 =st.columns(2)
                    with col1:
                        st.metric(
                            "Prediction", 
                            result['Prediction'],
                            help="Model's prediction (Real/Fake)"
                        )
                    with col2:
                        confidence = result['Probability'][1] if result['Prediction'] == "Fake" else result['Probability'][0]
                        st.metric(
                            "Confidence", 
                            f"{confidence:.1%}",
                            help="Model's confidence level"
                        )
                    
                    # Display LIME visualization
                    if result['LIME_visualization'] is not None:
                        try:
                            st.subheader("Feature Importance Analysis")
                            st.plotly_chart(result['LIME_visualization'], use_container_width=True)
                        except Exception as viz_error:
                            st.error(f"Error displaying visualization: {str(viz_error)}")
                
                    # Show detailed table as fallback
                    if result['LIME_data'] is not None:
                        with st.expander("Detailed Feature Importance"):
                            st.dataframe(
                                result['LIME_data'].style.format({
                                'importance': '{:.4f}'
                            })
                        )


else:
     st.error("Error: Model or text processor failed to load. Please check the logs.")


