# In a separate script (save_processor.py)
from text_processor import TextProcessor
import pandas as pd
import joblib

# Create and fit processor
processor = TextProcessor()
df = pd.read_csv('Data/fake_job_postings.csv')

# Combine text fields
text_data = []
for _, row in df.iterrows():
    combined = ' '.join([
        str(row.get('title', '')),
        str(row.get('company_profile', '')),
        str(row.get('description', '')),
        str(row.get('requirements', '')),
        str(row.get('benefits', ''))
    ])
    text_data.append(combined)

# Fit processor
processor.fit_transform(text_data)

# Save fitted processor
joblib.dump(processor, 'text_processor.joblib')