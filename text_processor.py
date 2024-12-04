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
        self.text_fields = ['title',  'company_profile', 'description', 
                           'requirements', 'benefits']
        
        # Define numerical features from CSV
        self.numerical_features = ['telecommuting', 'has_company_logo', 'has_questions']
        
        # Define categorical features that need encoding
        self.categorical_features = ['employment_type', 'department', 'required_experience', 
                                   'required_education', 'industry', 'function']
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

    # Initialize fraud detection word lists
        self.urgency_words = ['urgent', 'immediate', 'limited time', 'act now']
        self.guarantee_words = ['guarantee', 'guaranteed', 'promise', 'risk-free']
        self.pressure_words = ['only today', 'last chance', 'exclusive offer']
        
        # Add compiled regex patterns as class attributes
        self.email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+')
        self.money_pattern = re.compile(r'[\$£€]\d+')
        self.punctuation_pattern = re.compile(r'[!?]{2,}')
        self.caps_pattern = re.compile(r'\b[A-Z]{2,}\b')
        self.feature_prefixes = {
            'pattern': ['avg_word_length', 'caps_ratio', 'url_count', 'email_pattern', 'money_pattern'],
            'fraud': ['urgency_score', 'guarantee_score', 'pressure_score', 'excessive_punctuation', 'all_caps_words']
        }
        
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)


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
            tokens = word_tokenize(text)
            tokens = [token for token in tokens if token not in self.stop_words]
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            return ' '.join(tokens)
    
        except Exception as e:
            print(f"Warning: NLP processing failed: {str(e)}")
            return text  # Return cleaned text even if NLP processing fails
    
    
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
            'url_count': text.count('http'),
            'email_pattern':  len(re.findall(r'[\w\.-]+@[\w\.-]+', text)),
            'money_pattern': len(re.findall(r'[\$£€]\d+', text))
        }


    def extract_fraud_indicators(self, text):
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
    def process_dataframe(self, df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
        """
        Process a dataframe by adding text features for specified columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_columns (list): List of column names to process
            
        Returns:
            pd.DataFrame: DataFrame with added feature columns
        """
        df = df.copy()
        
        for column in text_columns:
            # Extract and add fraud indicators
            fraud_features = df[column].apply(self.extract_fraud_indicators)
            fraud_df = pd.DataFrame(fraud_features.tolist())
            fraud_df.columns = [f'{column}_{col}' for col in fraud_df.columns]
            
            # Extract and add text patterns
            pattern_features = df[column].apply(self.extract_text_patterns)
            pattern_df = pd.DataFrame(pattern_features.tolist())
            pattern_df.columns = [f'{column}_{col}' for col in pattern_df.columns]
            
            # Combine with main dataframe
            df = pd.concat([df, fraud_df, pattern_df], axis=1)
            
        return df

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
        
        # Process all features
        features = {}

        # Check for missing fields and log warnings
        required_fields = self.text_fields + self.numerical_features + self.categorical_features
        missing_fields = [field for field in required_fields if field not in input_dict]
        if missing_fields:
            self.logger.warning(f"Missing fields will be filled with default values: {missing_fields}")
    
        # Calculate description length
        features['description_length'] = len(input_dict.get('description', ''))

        # Process text fields with default handling
        for field in self.text_fields:
            try:
                self._process_text_field(input_dict, features, field)
            except Exception as e:
                self.logger.error(f"Error processing text field '{field}': {str(e)}")
                self._set_default_text_features(features, field)
        
        # Add numerical and categorical features with type checking
        self._add_numerical_categorical_features(input_dict, features)
            
        return pd.DataFrame([features])
    
    def _process_text_field(self, input_dict, features, field):
        """
        Helper method to process a single text field with error handling.
        """
        text_content = input_dict.get(field, '')
        if not text_content:
            self.logger.info(f"Field '{field}' is empty or missing")

        features[field] = input_dict.get(field, '')  # Add original text
        features[f'{field}_processed'] = self.preprocess_text(input_dict.get(field, ''))  # Add processed text

         # Extract patterns and indicators with error handling
        try: 
            patterns = self.extract_text_patterns(input_dict.get(field, ''))
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
        """
        Set default values for all features related to a text field.
        """
        features[field] = ''
        features[f'{field}_processed'] = ''
        self._set_default_pattern_features(features, field)
        self._set_default_fraud_features(features, field)

    def _set_default_pattern_features(self, features, field):
        """
        Set default values for pattern-related features.
        """
        for prefix in self.feature_prefixes['pattern']:
            features[f'{field}_{prefix}'] = 0

    def _set_default_fraud_features(self, features, field):
        """
        Set default values for fraud-related features.
        """
        for prefix in self.feature_prefixes['fraud']:
            features[f'{field}_{prefix}'] = 0

    def _add_numerical_categorical_features(self, input_dict, features):
        """
        Add numerical and categorical features with type validation.
        """
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