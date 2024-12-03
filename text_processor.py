from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import re
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TextProcessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            min_df=0.01,
            max_df=0.95,
            ngram_range=(1,3),
            stop_words='english',
            max_features=1000
        )
        
        # Store the vectorizer as instance variable so it can be reused
        self.is_fitted = False
        
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

    # Update text fields list
        self.text_fields=[
            'title',
            'department',
            'company_profile',
            'description',
            'requirements',
            'benefits',
            'employment_type',
            'required_experience',
            'required_education',
            'industry',
            'function'
        ]
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
        
    # Add numerical feature names
        self.numerical_features = [
            'telecommuting',
            'has_company_logo', 
            'has_questions'
        ]

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
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Join tokens back into text
        return ' '.join(tokens)
    
    
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
      
    
    def extract_text_patterns(self, input_dict):
        """
        Extracts all features from the input fields
        """
        # Validate input
        if not isinstance(input_dict, dict):
            raise ValueError("Input must be a dictionary")
        # Update required fields
        required_fields = ['title', 'description']  # minimum required fields
        missing_fields = [field for field in required_fields if not input_dict.get(field)]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # First get text features
        combined_text = self.combine_text_fields(input_dict)
        
        # Ensure vectorizer is fitted before transform
        if not self.is_fitted:
            raise ValueError("TextProcessor must be fitted before extracting features")
            
        tfidf_features = self.transform([combined_text])
        
        features = {}
        
        # Process all text fields
        for field in self.text_fields:
            text = input_dict.get(field, '')
            preprocessed_text = self.preprocess_text(text)
            
            # Add text patterns
            patterns = self.extract_text_patterns(text)
            for key, value in patterns.items():
                features[f'{field}_{key}'] = value
            
            # Add fraud indicators
            indicators = self.extract_fraud_indicators(text)
            for key, value in indicators.items():
                features[f'{field}_{key}'] = value
            
            # Add length features
            features[f'{field}_length'] = len(preprocessed_text.split())
        
        # Add numerical features
        for feature in self.numerical_features:
            features[feature] = input_dict.get(feature, 0)
        
        # Combine all features into a single DataFrame
        features_df = pd.DataFrame([features])
        return pd.concat([features_df, tfidf_features], axis=1)
    

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

    def extract_all_features(self, input_dict):
        """
        Extracts all features from the input fields, including both text and numerical.
        
        Args:
            input_dict (dict): Dictionary containing:
                Text fields:
                    - title
                    - company_profile
                    - description
                    - requirements
                    - benefits
                Numerical fields:
                    - telecommuting (0/1)
                    - has_company_logo (0/1)
                    - has_questions (0/1)
            
        Returns:
            pandas.DataFrame: DataFrame containing all extracted features
        """
        # Validate input
        if not isinstance(input_dict, dict):
            raise ValueError("Input must be a dictionary")
        # Add validation for required fields
        required_fields = ['title', 'description']  # minimum required fields
        missing_fields = [field for field in required_fields if not input_dict.get(field)]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # First get text features
        combined_text = self.combine_text_fields(input_dict)
        
        # Ensure vectorizer is fitted before transform
        if not self.is_fitted:
            raise ValueError("TextProcessor must be fitted before extracting features")
            
        tfidf_features = self.transform([combined_text])
        
        features = {}
        
        # Process text fields
        text_fields = ['title', 'company_profile', 'description', 'requirements', 'benefits']
        for field in text_fields:
            text = input_dict.get(field, '')
            preprocessed_text = self.preprocess_text(text)
            
            # Add text patterns
            patterns = self.extract_text_patterns(text)
            for key, value in patterns.items():
                features[f'{field}_{key}'] = value
            
            # Add fraud indicators
            indicators = self.extract_fraud_indicators(text)
            for key, value in indicators.items():
                features[f'{field}_{key}'] = value
            
            # Add length features
            features[f'{field}_length'] = len(preprocessed_text.split())
        
        # Add numerical features
        for feature in self.numerical_features:
            features[feature] = input_dict.get(feature, 0)
        
        # Combine all features into a single DataFrame
        features_df = pd.DataFrame([features])
        return pd.concat([features_df, tfidf_features], axis=1)

    def process_job_posting(self, input_dict):
        """
        Process a complete job posting and return all features.
        
        Args:
            input_dict (dict): Dictionary containing:
                Text fields:
                    - title
                    - company_profile
                    - description
                    - requirements
                    - benefits
                Numerical fields:
                    - telecommuting (0/1)
                    - has_company_logo (0/1)
                    - has_questions (0/1)
            
        Returns:
            pandas.DataFrame: DataFrame containing all extracted features
        """
        if not self.is_fitted:
            raise ValueError("TextProcessor must be fitted before processing job postings")
            
        return self.extract_all_features(input_dict)
    
    def fit_transform(self, text_data):
        """
        Fits the vectorizer and transforms the input text data.
        
        Args:
            text_data (list): List of text strings to process
        
        Returns:
            pandas.DataFrame: TF-IDF features as a dataframe
        """
        
        # Convert input to list if it's a single string
        if isinstance(text_data, str):
            text_data = [text_data]
        try:
            tfidf_matrix = self.vectorizer.fit_transform(text_data)
            self.is_fitted = True
        except Exception as e:
            self.is_fitted = False
            raise ValueError(f"Failed to fit vectorizer: {str(e)}")
        
        # Convert to dataframe with feature names
        return pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=self.vectorizer.get_feature_names_out()
        )

    def transform(self, text_data):
        """
        Transforms new text data using the fitted vectorizer.
        
        Args:
            text_data (list): List of text strings to process
        
        Returns:
            pandas.DataFrame: TF-IDF features as a dataframe
        """
        if not self.is_fitted:
            raise ValueError("TextProcessor must be fitted before transform can be called")
            
        # Convert input to list if it's a single string
        if isinstance(text_data, str):
            text_data = [text_data]
            
        # Transform the text data
        tfidf_matrix = self.vectorizer.transform(text_data)
        
        # Convert to dataframe with feature names
        return pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=self.vectorizer.get_feature_names_out()
        )

# Save processor to joblib
import joblib

if __name__ == "__main__":
    # Create processor instance
    processor = TextProcessor()
    
    # Save processor to joblib file
    joblib.dump(processor, 'text_processor.joblib')
    print("TextProcessor saved successfully!")