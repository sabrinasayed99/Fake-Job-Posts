import streamlit as st
import pandas as pd
from joblib import load
import plotly.express as px

# Set page configuration
st.set_page_config(layout="wide",
                   page_title="Fake Job Post Detector",
                   page_icon=":rotating_light:",
                   )


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
    

def get_feature_importances(text_input):
    """Get feature importances for the input text"""
    # Get the vectorizer and classifier from the pipeline
    vectorizer = pipeline.named_steps['tfidfvectorizer']
    classifier = pipeline.named_steps['logisticregression']
    
    # Transform the input text
    features = vectorizer.transform([text_input])
    
    # Get feature names and their coefficients
    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]
    
    # Get non-zero features for this specific input
    non_zero_features = features.nonzero()[1]
    
    # Create importance dictionary for non-zero features
    importances = []
    for idx in non_zero_features:
        importances.append({
            'feature': feature_names[idx],
            'importance': abs(coefficients[idx]),
            'coefficient': coefficients[idx]
        })
    
    # Sort by absolute importance
    importances.sort(key=lambda x: abs(x['importance']), reverse=True)
    return importances[:10]  # Return top 10 features


# Function to make predictions
def make_prediction(description):
    prediction = pipeline.predict([description])
    probability = pipeline.predict_proba([description])[0][1]
    importances = get_feature_importances(description)
    return {"Prediction": "Real" if prediction[0] else "Fake", 
            "Probability": probability[0],
            "Important Features": importances}

# Streamlit app layout
st.image("logo.jpg", width=200)
st.title(":rotating_light: Fake Job Post Detector :rotating_light:")
st.markdown("**Real or Fake?** This tool helps you determine the authenticity of job postings.")
st.write("Enter a job description to check if it's real or fake.")

# Load model at startup
pipeline, metadata, feature_importances = load_model_and_metadata()

if pipeline is not None:
    # Prediction Section
    st.header("Job Post Analysis")
    st.write("Enter a job description to check if it's real or fake.")
# Input field for job description
user_input = st.text_area("Enter a job description", placeholder="Type or paste job details here...")

# Prediction button
if st.button("Check Job Post"):
    if user_input.strip():
        with st.spinner("Processing..."):
            result = make_prediction(user_input)
        st.success("Prediction Complete!")
        st.write(f"Prediction: {result['Prediction']}")
        st.write(f"Probability: {result['Probability']:.2f}")
        # Model Insights Section (moved inside the button click)
        st.header("Model Insights")
            
        # Display important features
        st.subheader("Top Contributing Features for This Post:")
        importance_df = pd.DataFrame(result['Important Features'])
        
        # Create a horizontal bar chart
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
        st.write("\nDetailed Feature Importance:")
        st.dataframe(importance_df.style.format({
            'importance': '{:.4f}',
            'coefficient': '{:.4f}'
        }))
    else:
        st.warning("Please enter a job description.")


