import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataframe by handling infinite values and outliers.
    """
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # For each numeric column, replace outliers with column median
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        # Calculate IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Replace outliers with median
        median = df[col].median()
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        df[col] = df[col].fillna(median)
    
    return df

def load_and_preprocess(path):
    try:
        # Load data
        logger.info(f"Loading data from {path}")
        df = pd.read_csv(path)
        
        # Basic data validation
        if df.empty:
            raise ValueError("Empty dataset")
            
        # Handle missing values
        missing_before = df.isnull().sum().sum()
        df = df.dropna()
        missing_after = df.isnull().sum().sum()
        if missing_before > 0:
            logger.warning(f"Removed {missing_before - missing_after} rows with missing values")
            
        # Select numeric features
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric features found in the dataset")
            
        df = df[numeric_cols]
        
        # Clean data
        logger.info("Cleaning data...")
        df = clean_data(df)
        
        # Verify no infinite values remain
        if np.isinf(df.values).any():
            logger.warning("Infinite values detected after cleaning, replacing with column medians")
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(df.median())
        
        # Scale the data
        logger.info("Scaling data...")
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)
        
        logger.info(f"Preprocessing complete. Shape: {scaled.shape}")
        return scaled, scaler
        
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise
