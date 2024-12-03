import streamlit as st
import pandas as pd
from joblib import load

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
st.title("Fake Job Post Detector")
st.write("Enter a job description to check if it's real or fake.")

# Load model at startup
pipeline, metadata, feature_importances = load_model_and_metadata()

if pipeline is not None:
    st.write(f"Model loaded successfully! Using data source: {metadata['data_source']}")
    
    # Model Insights Section
    st.header("Model Insights")
    
    # Display Important features
    st.subheader("Important Features")
    st.bar_chart(feature_importances.set_index('feature')['importance'].head(20))
    
    # Display performance metrics
    st.subheader("Model Performance")
    metrics = metadata['performance_metrics']
    st.write(f"Accuracy: {metrics['accuracy']:.2f}")
    st.write(f"F1 Score: {metrics['f1_score']:.2f}")
    
    # Prediction Section
    st.header("Job Post Analysis")
    st.write("Enter a job description to check if it's real or fake.")
# Input field for job description
user_input = st.text_area("Job Description:", height=200)

# Prediction button
if st.button("Check Job Post"):
    if user_input.strip():
        result = make_prediction(user_input)
        st.write(f"Prediction: {result['Prediction']}")
        st.write(f"Probability: {result['Probability']:.2f}")

        # Display important features
        st.write("Top Contributing Features:")
        importance_df = pd.DataFrame(result['Important Features'])
        
        # Create a bar chart
        st.bar_chart(
            importance_df.set_index('feature')['importance']
        )
        # Show detailed table
        st.write("\nDetailed Feature Importance:")
        st.dataframe(importance_df.style.format({
            'importance': '{:.4f}',
            'coefficient': '{:.4f}'
        }))
    else:
        st.warning("Please enter a job description.")


