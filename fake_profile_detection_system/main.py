"""
Main entry point for training and evaluating the Fake Profile Detection System models.
"""

import yaml
from models.train_models import train_and_evaluate_models
from utils.helpers import load_combined_fake_real_data
from data.preprocessing import preprocess_data

import logging
from utils.logger import setup_logger

logger = setup_logger()

def main():
    try:
        # Load config
        with open("fake_profile_detection_system/config/config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Load and combine fake and real datasets
        fake_path = "data/fake-v1.0/fakeAccountData.json"
        real_path = "data/fake-v1.0/realAccountData.json"
        combined_data = load_combined_fake_real_data(fake_path, real_path)
        logger.info(f"Combined data columns: {combined_data.columns.tolist()}")

        # Preprocess data
        processed_data, features, labels = preprocess_data(combined_data)

        if labels is None:
            raise ValueError("Labels are None after preprocessing. Check the label column in the dataset.")

        # Train and evaluate models
        results = train_and_evaluate_models(processed_data, features, labels, config)

        logger.info(f"Training results: {results}")

    except Exception as e:
        logger.error(f"Error in main training pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
