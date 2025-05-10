import pandas as pd
import logging

logger = logging.getLogger(__name__)

def preprocess_data(df):
    """
    Preprocess the raw data:
    - Extract numeric features for model training
    - Return processed data, features, and labels
    """
    try:
        df = df.copy()
        df.fillna(0, inplace=True)

        label_col = 'label'
        labels = df[label_col] if label_col in df.columns else None

        if labels is None:
            raise ValueError(f"Label column '{label_col}' not found in dataframe.")

        logger.info(f"Label distribution:\n{labels.value_counts()}")

        feature_cols = [
            'userFollowerCount',
            'userFollowingCount',
            'userBiographyLength',
            'userMediaCount',
            'userHasProfilPic',
            'userIsPrivate',
            'usernameDigitCount',
            'usernameLength'
        ]

        for col in feature_cols:
            if col not in df.columns:
                raise ValueError(f"Feature column '{col}' not found in dataframe.")

        features = df[feature_cols]

        logger.info(f"Feature sample:\n{features.head()}")

        return df, features, labels
    except Exception as e:
        logger.error(f"Error in preprocess_data: {e}", exc_info=True)
        raise
