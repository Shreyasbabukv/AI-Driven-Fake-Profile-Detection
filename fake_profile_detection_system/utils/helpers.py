import json
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_data(filepath):
    """
    Load dataset from a JSON file and return as a pandas DataFrame.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Assuming the JSON is a list of dicts representing accounts
        df = pd.json_normalize(data)
        return df
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}", exc_info=True)
        raise

def load_combined_fake_real_data(fake_path, real_path):
    """
    Load and combine fake and real account datasets from JSON files.
    Assign label 1 for fake, 0 for real.
    Return combined DataFrame.
    """
    try:
        with open(fake_path, 'r', encoding='utf-8') as f:
            fake_data = json.load(f)
        with open(real_path, 'r', encoding='utf-8') as f:
            real_data = json.load(f)

        fake_df = pd.json_normalize(fake_data)
        real_df = pd.json_normalize(real_data)

        # Ensure label column 'isFake' exists and is correct
        if 'isFake' not in fake_df.columns:
            fake_df['isFake'] = 1
        if 'isFake' not in real_df.columns:
            real_df['isFake'] = 0

        combined_df = pd.concat([fake_df, real_df], ignore_index=True)
        return combined_df
    except Exception as e:
        logger.error(f"Error loading combined fake and real data: {e}", exc_info=True)
        raise
