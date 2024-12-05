## Detecting Fraudulent Job Posts
![2023-JulyAug-R1-Job-Scams-edited](https://github.com/user-attachments/assets/18e82a1d-e4d7-4a35-8998-b971b50c2be4)

#### Author : [Sabrina Sayed](https://github.com/sabrinasayed99)

## Overview:
Job scams and fake job listings are on the rise. In 2023 alone, there was a 118% increase in reported job scams compared to the previous year, according to the Identity Theft Resource Center's Trends in Identity report. AI generated job postings are making it easier for identity thieves to sounds legitimate through job postings. Last year, California had the highest number of recorded job scams. Bad actors create professional-looking LinkedIn profiles or job site profiles with live websites for phony businesses or even impersonate legitimate companies. 

## Goal:
We want to identify the characteristics and patterns of fraudulent job posts in order to protect vulnerable job seekers from identity theft and other scams, improve employers' job posting practices, and provide security recommendations to job site and hiring platforms.

## Data:
Using the Real or Fake Job Postings Dataset from [The University of the Aegean Laboratory of Information & Communication Systems Security]([url](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction/data)), we will predict if a job posting is legitimate or fraudulent. The dataset has 18 thousand data points and a total of 18 features. It includes a mix of categorical, numerical, and textual data. The data is labeled in 2 classes with a 0 for Legitimate and 1 for Fraudulent. Only 800 data points are labeled as Fraudulent, introducing heavy class imbalance. 

## Features:
- Job ID
- Title
- Description
- Company Profile
- Location
- Department
- Salary Range
- Requirements
- Required Education
- Required Experience
- Benefits
- Telecommuting
- Has Questions
- Has Company Logo
- Employment Type
- Industry
- Function
- Fraudulent


## Tech Stack:
- Languages: Python
- Libraries: scikit-learn, nltk, lime, shap, pandas, sci-py, numpy, matplotlib, seaborn
- Visualization: Streamlit

## Methods:
I built a baseline Naive Bayes Classifier, an XGBoost Classifier, and an XGBoost Classifier hypertuned with Grid Search Cross Validation and SMOTE. I utilized nltk to preprocess the textual data with lemmatization, removing stop words, and handling punctuation. I also created a modeling pipeline with a preprocessor implemented TargetEncoder, StandardScaler and TF-IDF vectorization. I used hyperparameter tuning technique like Synthetic Minority Oversampling to address class imbalance in the dataset. I prioritized the recall score  and accuracy as my main evaluating metrics.


## Results:
The last model I built, an XGBoost Classifier hypertuned with Grid Search Cross Validation, proved to be the best predictor of fraudulent posts with a recall score of 83% and an accuracy score of 98%. The classifier performed best with the default XGB parameters and SMOTE.

![Best_Smote_CFM](https://github.com/user-attachments/assets/6e03298f-8009-4dff-89f7-9d08478cffb0)

![ROC_AUC_Curve](https://github.com/user-attachments/assets/102f78bc-05e5-45f0-8171-b56fa674d6ea)

![precision-recall](https://github.com/user-attachments/assets/7bf55cc1-6d58-4c57-83fb-bb99c19d4878)


Using a SHAP Summary Plot to interpret the model, we learned that the presence of a company logo was the strongest indicator of legitimacy. We also learned legitimate posts tend to discuss growth and development, have strong grammar and clear professional language, and use descriptive terms related to work environment and team dynamics.


![Updated_SHAP](https://github.com/user-attachments/assets/2c704911-1ec2-432a-a024-52756d037d42)


The LIME Feature Importance chart showed us the red flags that come with fraudulent postings. These include mention of offshore work, use of aggressive and conversational language, overuse of external links, impersonation of well-known companies, and the targeting of entry level workers with entry level job posts.

![lime_explanation_matplotlib](https://github.com/user-attachments/assets/68c50595-8d5e-4a70-990d-cbcecd98cae6)

Our data exploration validated some of these findings. The word clouds below show the difference in the type and frequency of words used in legitimate posts versus fraudulent ones:

Real Postings Word Cloud:
![Real_Descr_WordCloud](https://github.com/user-attachments/assets/ad5b3de3-23c0-4c62-8455-c41ef71b0911)

Fake Postings Word Cloud:
![Fake_Descr_WordCloud](https://github.com/user-attachments/assets/0f1928b1-f5a9-4a72-99b8-de4c65579480)

The difference is stark with real postings having loads of industry-specific and descriptive language. Meanwhile the fake postings use very vague, dry, and generalized language that speaks very little to the nuances of work environments, dynamics, and job responsibilities.


## Recommendations:
### Job Seekers: 
Verify company claims with multiple steps of verfication (website, social media, logo, etc)
Be wary of overly casual and aggressive language in job posts.
Double check external links are legitimate.
Avoid applying to jobs that have vague job descriptions and don't provide detailed insight into company culture, expectations, goals, and responsibilities. 

### Employers:
Use industry-specific terms and descriptive language to give as much insight into what it's like to work at your company.
Provide verifiable company information with official websites, verified social media accounts, and other means of verifying your company's legitimacy.
Job posts should maintain professional language and proper grammar, but also include personality and distinctive traits.

### Job/Hiring Platforms:
Prioritize company logo verification and add multiple levels of verification for employer profiles.
Monitor links and external URL usage in job posts.
Flag posts with unusual language, lack of grammar/clear language, and mispellings for review.
Extra verification for posts mentioning offshore work.

## Directory:
[Presentation]([url](https://www.canva.com/design/DAGYKZgqBwc/_YKVCh6kJHvIwHZVKqbjFg/view?utm_content=DAGYKZgqBwc&utm_campaign=designshare&utm_medium=link&utm_source=editor))
[Dataset]([url](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction/data))

## Repository Files:
### Data 
Contains the raw data file, alternative filtered dataframes, preliminary versions of the cleaned data frame, saved versions of the lime explainer and feature importances, and the 'Final_Cleaned_Data.csv' which was used to train the model

### Notebooks
Contains Cleaning_NLP, where cleaning, EDA, and initial text preprocessing with nltk takes place; NB_Model where baseline modeling and tuning is stored; XGBoost_Model where XGBoost model and tuning is stored.

### __pycache__
Streamlits 

### Images
Seaborn and matplotlib visuals from EDA and model interpretability plots from SHAP and LIME

### Deliverables
Contains final project materials submitted for a grade including modeling notebooks, cleaning and eda notebook, and github repo snapshot.

### Models
Saved model pipeline, model metadata, training data, feature importances, shap and lime explainers used in the streamlit app.py

### Floating Files
The streamlit elements include requirements.txt which holds the app dependencies, text_prcocessor, which holds the text processor script, and app.py which holds the app's script.



