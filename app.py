import streamlit as st
import pandas as pd
from joblib import load
import plotly.express as px
import numpy as np
import re
import nltk

import unicodedata
import streamlit as st
import joblib
from text_processor import TextProcessor  # Still need the class definition

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
        # Load the model that uses Cleaned_Processed_Data_2
        model_path = '/Users/sabrinasayed/Documents/GitHub/Fake-Job-Posts/Models/best_model_pipeline.joblib'
        metadata_path = '/Users/sabrinasayed/Documents/GitHub/Fake-Job-Posts/Models/best_model_metadata.joblib'
        feature_importances_path = '/Users/sabrinasayed/Documents/GitHub/Fake-Job-Posts/Models/feature_importances.csv'
        
        pipeline = load(model_path)
        metadata = load(metadata_path)
        feature_importances = pd.read_csv(feature_importances_path)
        
        # Verify data source
        if metadata['data_source'] != 'Cleaned_Processed_Data_2.csv':
            st.warning("Warning: Model might not be using the latest data!")
            
        return pipeline, metadata, feature_importances
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None
    
def get_feature_importances(processed_features):
    """Get feature importances for the processed input"""
    try:
        vectorizer = pipeline.named_steps['tfidfvectorizer']
        classifier = pipeline.named_steps['logisticregression']
        
        # Get feature names and their coefficients
        feature_names = vectorizer.get_feature_names_out()
        coefficients = classifier.coef_[0]
        
        # Get non-zero features
        non_zero_features = processed_features.nonzero()[1]
        
        # Create importance dictionary
        importances = [{
            'feature': feature_names[idx],
            'importance': abs(coefficients[idx]),
            'coefficient': coefficients[idx]
        } for idx in non_zero_features]
        
        # Sort by absolute importance
        return sorted(importances, key=lambda x: abs(x['importance']), reverse=True)[:10]
    except Exception as e:
        st.error(f"Error calculating feature importances: {str(e)}")
        return []

# Function to make predictions
def make_prediction(input_data):
    """Make predictions using the loaded model"""
    try:
        # Input validation
        if not input_data.get('title', '').strip() or not input_data.get('description', '').strip():
            raise ValueError("Job title and description are required")
    
     # Process the input using the text processor
        processed_features = processor.process_job_posting(input_data)
        
        # Make prediction
        prediction = pipeline.predict(processed_features)
        probability = pipeline.predict_proba(processed_features)[0]
        importances = get_feature_importances(processed_features)
        
        return {
            "Prediction": "Real" if prediction[0] else "Fake",
            "Probability": probability,
            "Important Features": importances
        }
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

# Initialize processor and model
processor = load_processor()
pipeline, metadata, feature_importances = load_model_and_metadata()



# Streamlit app layout
st.title(":rotating_light: Fake Job Post Detector :rotating_light:")
st.markdown("**Real or Fake?** This tool helps you determine the authenticity of job postings.")

# Load model at startup
pipeline, metadata, feature_importances = load_model_and_metadata()

if processor is not None and pipeline is not None:
    with st.form("job_post_form"):
        st.subheader("Insert Job Post Details")
        col1,col2 = st.columns(2)

        with col1:
            title = st.text_input("Title",placeholder="e.g., Senior Software Engineer")
            department = st.text_input("Department",placeholder="e.g., Engineering")
            employment_type= st.selectbox("Employment Type",
                                        options=["Full-Time","Part-Time","Contract","Temporary","Internship"])
            industry = st.text_input("Industry",placeholder="e.g., Technology")
            function = st.text_input("Job Function",placeholder="e.g., Software Development")
        
        with col2:
            required_experience = st.text_input("Required Experience",
                                              placeholder="e.g., 5+ years")
            required_education = st.text_input("Required Education",
                                             placeholder="e.g., Bachelor's Degree")
            has_company_logo = st.checkbox("Has Company Logo")
            has_questions = st.checkbox("Has Screening Questions")

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
                    st.metric(
                        "Confidence", 
                        f"{result['Probability'][1]:.1%}",
                        help="Model's confidence level"
                    )
                

        # Model Insights Section
            st.header("Model Insights")
            with st.expander("Understanding the Results"):
                    st.markdown("""
                        - The chart shows the top features influencing the model's decision
                        - Longer bars indicate stronger influence
                        - Features include words, phrases, and patterns from the job posting
                        - Positive coefficients suggest legitimate posts, negative suggest potential fraud
                    """)
            # Display important features
            importance_df = pd.DataFrame(result['Important Features'])
        
        # Create a horizontal bar chart of important features
            fig2 = px.bar(
                importance_df,
                x='importance',
                y='feature',
                orientation='h',
                title='Top Contributing Features for This Post'
                )
            fig2.update_layout(
                yaxis={'categoryorder':'total ascending'},
                height=400
                )
            st.plotly_chart(fig2, use_container_width=True)

            # Show detailed table
            with st.expander("Detailed Feature Importance"):
                st.dataframe(
                    importance_df.style.format({
                        'importance': '{:.4f}',
                        'coefficient': '{:.4f}' })
                        )

else:
    st.warning("Please enter a job description.")


