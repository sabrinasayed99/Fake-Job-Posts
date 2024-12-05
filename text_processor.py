import pandas as pd
import numpy as np
import re
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List
import logging


class TextProcessor:
    def __init__(self):
         # Define fields to process
        self.text_fields = [
            'title',  'company_profile', 'description',
            'requirements', 'benefits']
        
        # Define numerical features from CSV
        self.numerical_features = [
            'telecommuting', 'has_company_logo', 'has_questions']
        
        # Define categorical features that need encoding
        self.categorical_features = [
            'employment_type', 'department', 'required_experience', 
            'required_education', 'industry', 'function']
        
        #Initialize fraud detection word lists
        self.urgency_words = ['urgent', 'immediate', 'limited time', 'act now']
        self.guarantee_words = ['guarantee', 'guaranteed', 'promise', 'risk-free']
        self.pressure_words = ['only today', 'last chance', 'exclusive offer']
        
        self.feature_prefixes = {
            'fraud': ['urgency_score', 'guarantee_score', 'pressure_score', 'excessive_punctuation', 'all_caps_words']
        }
        
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Add missing patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Download required NLTK data
        # Add try/except for each download separately
        nltk_resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
        for resource in nltk_resources:
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                print(f"Warning: Failed to download {resource}: {str(e)}")


    def preprocess_text(self, text):
        """
        Preprocesses text using the same steps as in the notebook.
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        MAX_TEXT_LENGTH = 1000000  # 1MB
        if not isinstance(text, str):
            return ''
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Input text exceeds maximum length of {MAX_TEXT_LENGTH} characters")
        
        # Convert to lowercase
        text = text.lower()
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        # Remove non-ASCII characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Combine regex operations for better efficiency
        text = re.sub(r'<.*?>|[^\w\s]|\d+', ' ', text)
    
        # Remove extra whitespace
        text = ' '.join(text.split())
    
        # NLP processing with error handling
        try:
            # Tokenization
            tokens = word_tokenize(text)
            # Remove stop words
            tokens = [token for token in tokens if token not in self.stop_words]
            # Lemmatize
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            # Join tokens back into a string
            return ' '.join(tokens)
    
        except Exception as e:
            self.logger.error(f"NLP processing failed: {str(e)}")
            return text  # Return cleaned text even if NLP processing fails
    
    def extract_fraud_indicators(self, text:str) -> dict:
        """
        Extracts indicators that might suggest fraudulent content.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary of fraud indicator features
        """
        if not isinstance(text, str):
            return {
                'urgency_score': 0,
                'guarantee_score': 0,
                'pressure_score': 0,
                'excessive_punctuation': 0,
                'all_caps_words': 0
            }
        
        text_lower = text.lower()
        
        return {
            'urgency_score': sum(1 for word in self.urgency_words if word in text_lower),
            'guarantee_score': sum(1 for word in self.guarantee_words if word in text_lower),
            'pressure_score': sum(1 for word in self.pressure_words if word in text_lower),
            'excessive_punctuation': len(re.findall(r'[!?]{2,}', text)),
            'all_caps_words': len(re.findall(r'\b[A-Z]{2,}\b', text))
        }
    def extract_text_patterns(self, text):
        """
        Extracts basic text patterns
        """
        # Validate input
        if not isinstance(text, str):
            return {
                'avg_word_length': 0,
                'caps_ratio': 0,
                'url_count': 0,
                'email_pattern': 0,
                'money_pattern': 0
            }
        # Calculate features 
        words = text.split()
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        text_length = len(text)
        caps_ratio = sum(1 for c in text if c.isupper()) / text_length if text_length > 0 else 0
        
        # Return features as a dictionary
        return {
            'avg_word_length': avg_word_length,
            'caps_ratio': caps_ratio,
            'url_count': len(re.findall(self.url_pattern, text)),
            'email_pattern':  len(re.findall(r'[\w\.-]+@[\w\.-]+', text)),
            'money_pattern': len(re.findall(r'[\$£€]\d+', text))
        }
    def combine_text_fields(self, text_dict):
        """
        Combines multiple text fields into a single string.
        
        Args:
            text_dict (dict): Dictionary containing text fields with keys:
                - title
                - company_profile
                - description
                - requirements
                - benefits
        
        Returns:
            str: Combined text string
        """
        # Handle missing fields gracefully by using get() with empty string default
        combined = ' '.join([
            text_dict.get(field, '') for field in self.text_fields
        ])
        return combined.strip()
      
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a dataframe ensuring exact feature order matching training data.
        """
        df = df.copy()
    
        # Add description length
        df['description_length'] = df['description'].str.len().fillna(0)
    
        # Define the exact order of features as they were during training
        expected_columns = [
            # Text fields
            *self.text_fields,
            # Numerical features
            *self.numerical_features,
            # Categorical features
            *self.categorical_features,
            # Description length
            'description_length',
            # Fraud features grouped by type
            *[f'{field}_urgency_score' for field in self.text_fields],
            *[f'{field}_guarantee_score' for field in self.text_fields],
            *[f'{field}_pressure_score' for field in self.text_fields],
            *[f'{field}_excessive_punctuation' for field in self.text_fields],
            *[f'{field}_all_caps_words' for field in self.text_fields]
        ]
    
        # Process text fields
        for column in self.text_fields:
            if column in df.columns:
                original_text = df[column].copy()
                
                # Replace original text with preprocessed text
                df[column] = df[column].apply(self.preprocess_text)
                
                # Extract and add text patterns
                pattern_features = original_text.apply(self.extract_text_patterns)
                pattern_df = pd.DataFrame(pattern_features.tolist())
                for feature in pattern_df.columns:
                    df[f'{column}_{feature}'] = pattern_df[feature]
                
                # Extract and add fraud indicators
                fraud_features = original_text.apply(self.extract_fraud_indicators)
                fraud_df = pd.DataFrame(fraud_features.tolist())
                for feature in fraud_df.columns:
                    df[f'{column}_{feature}'] = fraud_df[feature]
            else:
                # Add empty columns for missing fields
                df[column] = ''
                df[f'{column}_processed'] = ''
                
                # Add default pattern features
                for feature in self.feature_prefixes['pattern']:
                    df[f'{column}_{feature}'] = 0
                
                # Add default fraud features
                for feature in self.feature_prefixes['fraud']:
                    df[f'{column}_{feature}'] = 0
    
        # Ensure categorical features exist
        for feature in self.categorical_features:
            if feature not in df.columns:
                df[feature] = ''
    
        # Ensure numerical features exist
        for feature in self.numerical_features:
            if feature not in df.columns:
                df[feature] = 0
    
        # Fill any missing columns with appropriate defaults
        for col in expected_columns:
            if col not in df.columns:
                if any(num_feat in col for num_feat in self.numerical_features):
                    df[col] = 0
                else:
                    df[col] = ''
    
        # Return DataFrame with exact column order
        return df[expected_columns]

    def process_job_posting(self, input_dict):
        """
        Process a complete job posting and return a dataframe.
        
        Args:
            input_dict (dict): Dictionary containing all features matching Final_Cleaned_Data.csv columns
            
        Returns:
            pandas.DataFrame: DataFrame containing all extracted features
        """
        
        if not isinstance(input_dict, dict):
            self.logger.error("Input must be a dictionary")
            return pd.DataFrame()  # Return empty DataFrame if input is invalid
        
        # Convert single posting to DataFrame and process it
        df = pd.DataFrame([input_dict])
        return self.process_dataframe(df)
    
    def _process_text_field(self, input_dict, features, field):
        """
        Helper method to process a single text field with error handling.
        """
        text_content = input_dict.get(field, '')
        if not text_content:
            self.logger.info(f"Field '{field}' is empty or missing")

        # Add original and processed text
        features[field] = text_content
        features[f'{field}_processed'] = self.preprocess_text(text_content)  # Add processed text

         # Extract patterns and indicators with error handling
        try: 
            patterns = self.extract_text_patterns(text_content)
            for key, value in patterns.items():
                features[f'{field}_{key}'] = value
        except Exception as e:
            self.logger.error(f"Error extracting patterns for '{field}': {str(e)}")
            self._set_default_pattern_features(features, field)
            
        try:
            indicators = self.extract_fraud_indicators(text_content)
            for key, value in indicators.items():
                features[f'{field}_{key}'] = value
        except Exception as e:
            self.logger.error(f"Error extracting fraud indicators for '{field}': {str(e)}")
            self._set_default_fraud_features(features, field)

    ## SET DEFAULT FEATURES WHEN VALUES ARE MISSING ##
    def _set_default_text_features(self, features, field):
        """Set default values for all features related to a text field."""
        features[field] = ''
        features[f'{field}_processed'] = ''
        self._set_default_pattern_features(features, field)
        self._set_default_fraud_features(features, field)

    def _set_default_pattern_features(self, features, field):
        """Set default values for pattern-related features. """
        for prefix in self.feature_prefixes['pattern']:
            features[f'{field}_{prefix}'] = 0

    def _set_default_fraud_features(self, features, field):
        """Set default values for fraud-related features."""
        for prefix in self.feature_prefixes['fraud']:
            features[f'{field}_{prefix}'] = 0

    def _add_numerical_categorical_features(self, input_dict, features):
        """Add numerical and categorical features with type validation."""
        # Handle numerical features
        for feature in self.numerical_features:
            try:
                value = input_dict.get(feature, 0)
                features[feature] = float(value) if value is not None else 0
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Invalid numerical value for '{feature}', using default 0: {str(e)}")
                features[feature] = 0

        # Handle categorical features
        for feature in self.categorical_features:
            try:
                value = input_dict.get(feature, '')
                features[feature] = str(value) if value is not None else ''
            except Exception as e:
                self.logger.warning(f"Invalid categorical value for '{feature}', using empty string: {str(e)}")
                features[feature] = ''



# Save processor to joblib
import joblib

if __name__ == "__main__":
    # Create processor instance
    processor = TextProcessor()
    
    # Save processor to joblib file
    joblib.dump(processor, 'text_processor.joblib')
    print("TextProcessor saved successfully!")

    # Create a sample DataFrame with one row and all required columns
    sample_data = {
        # Text fields
        'title': [''],
        'company_profile': [''],
        'description': [''],
        'requirements': [''],
        'benefits': [''],
        
        # Numerical features
        'telecommuting': [0],
        'has_company_logo': [0],
        'has_questions': [0],
        
        # Categorical features
        'employment_type': [''],
        'department': [''],
        'required_experience': [''],
        'required_education': [''],
        'industry': [''],
        'function': ['']
    }

    sample_df = pd.DataFrame(sample_data)

    # Print all expected columns in order
    processed_df = processor.process_dataframe(sample_df)
    print("Total features:", len(processed_df.columns))
    print("\nFeature names:")
    for col in processed_df.columns:
        print(f"- {col}")