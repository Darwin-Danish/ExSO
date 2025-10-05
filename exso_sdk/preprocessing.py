"""
Data Preprocessing for Exoplanet Classification - V2 Compatible
==============================================================

This module provides preprocessing functions compatible with the V2 model,
which uses different strategies for critical vs auxiliary features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from .config import CRITICAL_FEATURES, AUXILIARY_FEATURES, REQUIRED_COLUMNS


def validate_features(df: pd.DataFrame) -> None:
    """
    Validate that DataFrame contains all required features for V2 model.
    
    Args:
        df: Input DataFrame
        
    Raises:
        ValueError: If required features are missing
    """
    missing_features = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")


def clean_missing_v2(df: pd.DataFrame, strategy: str = 'model_handling') -> pd.DataFrame:
    """
    Handle missing values using V2 model's approach.
    
    The V2 model handles missing values differently for critical vs auxiliary features:
    - Critical features: Filled with -999 (constant strategy)
    - Auxiliary features: Filled with median
    
    Args:
        df: Input DataFrame
        strategy: 'model_handling' (use V2 approach) or 'drop' or 'fill'
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    if strategy == 'model_handling':
        # Use the same approach as V2 model
        # Critical features: fill with -999
        for col in CRITICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(-999)
        
        # Auxiliary features: fill with median
        for col in AUXILIARY_FEATURES:
            if col in df.columns:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                
    elif strategy == 'drop':
        return df.dropna()
        
    elif strategy == 'fill':
        # Fill numeric columns with median
        for col in df.select_dtypes(include='number').columns:
            median = df[col].median()
            df[col] = df[col].fillna(median)
        # For categorical, fill with mode (if any)
        for col in df.select_dtypes(include='object').columns:
            mode = df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown'
            df[col] = df[col].fillna(mode)
    else:
        raise ValueError("strategy must be 'model_handling', 'drop', or 'fill'")
    
    return df


def normalize_scale(df: pd.DataFrame, cols: List[str], method: str = 'standard') -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numeric features using StandardScaler (compatible with V2 model).
    
    Args:
        df: Input DataFrame
        cols: List of columns to scale
        method: Scaling method ('standard' only for V2 compatibility)
        
    Returns:
        Tuple of (scaled DataFrame, scaler object)
    """
    if method != 'standard':
        raise ValueError("V2 model requires standard scaling")
    
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[cols] = scaler.fit_transform(df[cols])
    return df_scaled, scaler


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with categorical columns encoded
    """
    cat_cols = df.select_dtypes(include='object').columns
    if len(cat_cols) == 0:
        return df
    return pd.get_dummies(df, columns=cat_cols)


def preprocess_for_v2(df: pd.DataFrame, 
                     handle_missing: str = 'model_handling',
                     validate_input: bool = True) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for V2 model.
    
    Args:
        df: Input DataFrame
        handle_missing: Strategy for handling missing values
        validate_input: Whether to validate required features
        
    Returns:
        Preprocessed DataFrame ready for V2 model
    """
    if validate_input:
        validate_features(df)
    
    # Handle missing values
    df_processed = clean_missing_v2(df, strategy=handle_missing)
    
    # Note: V2 model includes its own preprocessing pipeline,
    # so we don't need to scale here - the model handles it
    
    return df_processed


def create_sample_data() -> pd.DataFrame:
    """
    Create sample data for testing V2 model.
    
    Returns:
        DataFrame with sample exoplanet data
    """
    sample_data = {
        'koi_period': [10.5, 25.3, 5.2],
        'koi_depth': [1200.5, 850.2, 2100.8],
        'koi_prad': [2.1, 1.8, 3.2],
        'koi_sma': [0.8, 1.2, 0.6],
        'koi_teq': [450.2, 380.5, 520.1],
        'koi_insol': [1.2, 0.8, 1.8],
        'koi_model_snr': [15.2, 12.8, 18.5],
        'koi_time0bk': [2454833.0, 2454834.0, 2454832.0],
        'koi_duration': [0.5, 0.8, 0.3],
        'koi_incl': [89.2, 88.5, 89.8],
        'koi_srho': [1.5, 1.8, 1.2],
        'koi_srad': [1.1, 0.9, 1.3],
        'koi_smass': [1.2, 0.8, 1.5],
        'koi_steff': [5800.0, 5200.0, 6200.0],
        'koi_slogg': [4.5, 4.2, 4.8],
        'koi_smet': [0.1, -0.2, 0.3]
    }
    
    return pd.DataFrame(sample_data)


def compute_period_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute period harmonics and folded stats (legacy function, kept for compatibility).
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with additional period features
    """
    df = df.copy()
    
    if 'koi_period' in df.columns:
        df['period_harmonic_2'] = df['koi_period'] * 2
        df['period_harmonic_0.5'] = df['koi_period'] * 0.5
        
        if 'koi_depth' in df.columns and 'koi_duration' in df.columns:
            df['folded_mean'] = df['koi_depth'] / df['koi_duration']
    
    return df


def compute_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean, std, skew, kurtosis of numeric columns (legacy function).
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with statistical features
    """
    try:
        from scipy.stats import skew, kurtosis
    except ImportError:
        print("Warning: scipy not available, skipping statistical features")
        return df
    
    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    for col in numeric_cols:
        try:
            df[f'{col}_skew'] = skew(df[col])
            df[f'{col}_kurtosis'] = kurtosis(df[col])
        except:
            # Skip if calculation fails
            pass
    
    return df


def compute_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute domain-specific features like SNR, vetting flags (legacy function).
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with domain features
    """
    df = df.copy()
    
    if 'koi_depth' in df.columns and 'koi_duration' in df.columns:
        df['transit_snr'] = df['koi_depth'] / (df['koi_duration'] + 1e-6)
    
    if 'koi_depth' in df.columns:
        df['vet_flag'] = (df['koi_depth'] > 500).astype(int)
    
    return df


def preprocess_lightcurve(lc: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for lightcurve preprocessing: detrend + resample (legacy function).
    
    Args:
        lc: pandas DataFrame with time and flux columns
        
    Returns:
        Preprocessed lightcurve DataFrame
    """
    lc = lc.copy()
    
    if 'flux' in lc.columns:
        # Simple rolling median detrend
        lc['flux_detrended'] = lc['flux'] / lc['flux'].rolling(window=101, center=True, min_periods=1).median()
    
    if 'time' in lc.columns:
        # Resample to fixed cadence (e.g., 30 min)
        lc_resampled = lc.set_index('time').resample('30T').mean().interpolate()
        return lc_resampled.reset_index()
    
    return lc


# Legacy function names for backward compatibility
def clean_missing(df, strategy='drop'):
    """Legacy function - redirects to clean_missing_v2."""
    return clean_missing_v2(df, strategy)