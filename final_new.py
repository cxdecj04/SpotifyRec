import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import logging
from typing import Dict, List, Tuple, Any

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text):
    """Clean and normalize text"""
    if pd.isna(text) or text == '':
        return 'unknown'
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s,&/-]', '', text)
    return text

def process_genres(genre_str):
    """Process genre string into list of genres"""
    if pd.isna(genre_str) or not str(genre_str).strip('[]'):
        return ['unknown']
    try:
        genres = str(genre_str).strip('[]').split(',')
        return [g.strip().strip("'\"").lower() for g in genres if g.strip()]
    except Exception as e:
        logger.warning(f"Error processing genre string: {e}")
        return ['unknown']

def prepare_recommender(df1, df2, feature_weights=None):
    """Prepare the recommender model with the given dataframes"""
    # Default feature configuration
    numeric_features = ['popularity']
    categorical_features = ['album_name', 'artists']  # Both treated as text features
    boolean_features = ['explicit']
    
    # Multi-label features configuration (only track_genre)
    multi_label_config = {
        'track_genre': {
            'processor': process_genres,
            'weight_key': 'genre'
        }
    }

    # Default weights
    weights = feature_weights or {
        'numeric': 10.0,
        'categorical': 10.0,  # Applies to TF-IDF features (album_name and artists)
        'boolean': 10.0,
        'genre': 10.0  # Applies to multi-label genre features
    }
    
    # Prepare numeric features
    scaler = None
    if numeric_features:
        combined_numeric = pd.concat([
            df1[numeric_features].fillna(0),
            df2[numeric_features].fillna(0)
        ])
        scaler = StandardScaler().fit(combined_numeric)
    
    # Prepare TF-IDF vectorizer for categorical features (album_name and artists)
    vectorizers = {}
    for col in categorical_features:
        combined_text = pd.concat([df1[col].fillna(""), df2[col].fillna("")]).astype(str)
        vectorizer = TfidfVectorizer().fit(combined_text)
        vectorizers[col] = vectorizer
    
    # Prepare MultiLabelBinarizer for multi-label genre feature
    mlbs = {}
    for feature, config in multi_label_config.items():
        processor = config['processor']
        
        # Process data from both dataframes
        processed1 = df1[feature].apply(processor)
        processed2 = df2[feature].apply(processor)
        all_processed = pd.concat([processed1, processed2], axis=0)
        
        # Collect all unique items
        all_items = set()
        for items in all_processed:
            all_items.update(items)
        unique_items = sorted(all_items)
        if not unique_items:
            unique_items = ['unknown']
        
        # Fit MultiLabelBinarizer
        mlb = MultiLabelBinarizer(classes=unique_items)
        mlb.fit([unique_items])  # Dummy fit to initialize
        mlbs[feature] = mlb

    # Return all prepared components
    return {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'boolean_features': boolean_features,
        'multi_label_config': multi_label_config,
        'weights': weights,
        'scaler': scaler,
        'mlbs': mlbs,
        'vectorizers': vectorizers
    }

def transform_features(df, model):
    """Transform all features for the input dataframe"""
    features = pd.DataFrame(index=df.index)
    
    # Transform numeric features
    if model['numeric_features'] and model['scaler']:
        numeric_data = df[model['numeric_features']].fillna(df[model['numeric_features']].mean())
        numeric_scaled = model['scaler'].transform(numeric_data)
        numeric_df = pd.DataFrame(numeric_scaled, 
                                 columns=model['numeric_features'],
                                 index=df.index)
        features = pd.concat([features, numeric_df], axis=1)
    
    # Transform categorical features using TF-IDF (album_name and artists)
    for col in model['categorical_features']:
        if col in model['vectorizers']:
            vectorizer = model['vectorizers'][col]
            tfidf_matrix = vectorizer.transform(df[col].fillna("").astype(str)).toarray()
            tfidf_df = pd.DataFrame(tfidf_matrix, 
                                    columns=[f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
                                    index=df.index)
            features = pd.concat([features, tfidf_df], axis=1)
    
    # Transform boolean features
    for col in model['boolean_features']:
        features[col] = df[col].fillna(False).astype(int)
    
    # Transform multi-label genre feature
    for feature, config in model['multi_label_config'].items():
        if feature not in df.columns:
            continue
        processor = config['processor']
        mlb = model['mlbs'][feature]
        
        processed_items = df[feature].apply(processor)
        mlb_matrix = mlb.transform(processed_items)
        columns = [f"{feature}_{item}" for item in mlb.classes_]
        mlb_df = pd.DataFrame(mlb_matrix, columns=columns, index=df.index)
        features = pd.concat([features, mlb_df], axis=1)
    
    return features

def get_recommendations(top_songs_df, dataset_df, model, k=10):
    """Get song recommendations based on top songs"""
    # Transform features
    top_songs_features = transform_features(top_songs_df, model)
    dataset_features = transform_features(dataset_df, model)
    
    # Apply weights
    # Numeric features
    if model['numeric_features']:
        top_songs_features[model['numeric_features']] *= model['weights']['numeric']
        dataset_features[model['numeric_features']] *= model['weights']['numeric']
    
    # Categorical TF-IDF features (album_name and artists)
    tfidf_cols = [col for col in top_songs_features.columns if 'tfidf' in col]
    if tfidf_cols:
        top_songs_features[tfidf_cols] *= model['weights']['categorical']
        dataset_features[tfidf_cols] *= model['weights']['categorical']
    
    # Boolean features
    if model['boolean_features']:
        top_songs_features[model['boolean_features']] *= model['weights']['boolean']
        dataset_features[model['boolean_features']] *= model['weights']['boolean']
    
    # Multi-label genre features
    for feature, config in model['multi_label_config'].items():
        weight_key = config['weight_key']
        weight = model['weights'].get(weight_key, 1.0)
        prefix = f"{feature}_"
        feature_cols = [col for col in top_songs_features.columns if col.startswith(prefix)]
        if feature_cols:
            top_songs_features[feature_cols] *= weight
            dataset_features[feature_cols] *= weight
    
    # Calculate similarities
    top_songs_vector = top_songs_features.mean(axis=0).values.reshape(1, -1)
    similarities = cosine_similarity(top_songs_vector, dataset_features)
    
    # Get recommendations
    top_indices = similarities.flatten().argsort()[-k:][::-1]
    recommendations = dataset_df.iloc[top_indices].copy()
    recommendations['similarity_score'] = similarities.flatten()[top_indices]
    
    # Prepare output
    output_cols = ['similarity_score', 'track_name', 'artists', 'album_name']  # Include key columns
    for feature_list in [model['categorical_features'], model['numeric_features']]:
        output_cols.extend([col for col in feature_list if col in recommendations.columns])
    
    existing_cols = [col for col in output_cols if col in recommendations.columns]
    final_recommendations = recommendations[existing_cols].copy()
    final_recommendations['similarity_score'] = final_recommendations['similarity_score'].round(4)
    
    return final_recommendations

if __name__ == "__main__":
    try:
        # Load datasets
        top_songs_df = pd.read_csv('playlist_tracks.csv')
        dataset_df = pd.read_csv('top_100_tracks_with_genres.csv')
        
        # Configure recommender with custom weights
        weights = {
            'numeric': 100000000000000,
            'categorical': 10000000.0,  # Higher weight for album_name and artists
            'boolean': 10.0,
            'genre': 1000.0
        }
        
        # Prepare the recommender model
        model = prepare_recommender(top_songs_df, dataset_df, feature_weights=weights)
        
        # Get recommendations
        recommendations = get_recommendations(top_songs_df, dataset_df, model, k=15)
        
        # Save results
        recommendations.to_csv('song_recommendations.csv', index=False)
        logger.info(f"Recommendations saved to song_recommendations.csv")
        logger.info(recommendations.head())
            
    except Exception as e:
        logger.error(f"Error in execution: {e}", exc_info=True)