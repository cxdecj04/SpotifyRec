import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from difflib import SequenceMatcher
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

def text_similarity(text1, text2):
    """Calculate similarity between two text values"""
    if text1 == text2:
        return 1.0
    
    # Handle multi-value fields
    set1 = set(clean_text(text1).split(','))
    set2 = set(clean_text(text2).split(','))
    
    # Use Jaccard similarity for multi-value fields
    if len(set1) > 1 or len(set2) > 1:
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    # Use sequence matcher for single values
    return SequenceMatcher(None, clean_text(text1), clean_text(text2)).ratio()

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
    categorical_features = ['artists', 'album_name']
    boolean_features = ['explicit']
    genre_features = ['track_genre']
    
    # Default weights
    weights = feature_weights or {
        'numeric': 10.0,
        'categorical': 10.0,
        'boolean': 10,
        'genre': 10.0
    }
    
    # Keep only common features
    numeric_features = [col for col in numeric_features if col in df1.columns and col in df2.columns]
    categorical_features = [col for col in categorical_features if col in df1.columns and col in df2.columns]
    boolean_features = [col for col in boolean_features if col in df1.columns and col in df2.columns]
    genre_features = [col for col in genre_features if col in df1.columns and col in df2.columns]
    
    # Prepare numeric features
    scaler = None
    if numeric_features:
        combined_numeric = pd.concat([
            df1[numeric_features].fillna(0),
            df2[numeric_features].fillna(0)
        ])
        scaler = StandardScaler().fit(combined_numeric)
    
    # Prepare categorical features
    categorical_values = {}
    categorical_similarities = {}
    for col in categorical_features:
        values1 = df1[col].apply(clean_text).unique()
        values2 = df2[col].apply(clean_text).unique()
        all_values = np.unique(np.concatenate([values1, values2]))
        categorical_values[col] = all_values
        
        # Create similarity matrix
        sim_matrix = np.zeros((len(all_values), len(all_values)))
        for i, val1 in enumerate(all_values):
            for j, val2 in enumerate(all_values):
                sim_matrix[i, j] = text_similarity(val1, val2)
        categorical_similarities[col] = sim_matrix
    
    # Prepare genre processor
    mlb = None
    if genre_features:
        genre_col = genre_features[0]
        genres1 = df1[genre_col].apply(process_genres)
        genres2 = df2[genre_col].apply(process_genres)
        
        all_genres = []
        for genre_list in genres1: all_genres.extend(genre_list)
        for genre_list in genres2: all_genres.extend(genre_list)
        
        unique_genres = sorted(set(g for g in all_genres if g))
        if not unique_genres:
            unique_genres = ['unknown']
        
        mlb = MultiLabelBinarizer(classes=list(unique_genres))
        mlb.fit([unique_genres])
    
    # Return all prepared components
    return {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'boolean_features': boolean_features,
        'genre_features': genre_features,
        'weights': weights,
        'scaler': scaler,
        'mlb': mlb,
        'categorical_values': categorical_values,
        'categorical_similarities': categorical_similarities
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
    
    # Transform categorical features
    for col in model['categorical_features']:
        if col not in model['categorical_values']:
            continue
            
        values = df[col].apply(clean_text)
        value_indices = {val: i for i, val in enumerate(model['categorical_values'][col])}
        
        embeddings = np.zeros((len(df), len(model['categorical_values'][col])))
        for i, value in enumerate(values):
            if value in value_indices:
                embeddings[i, value_indices[value]] = 1.0
        
        embed_cols = [f"{col}_embed_{i}" for i in range(embeddings.shape[1])]
        embed_df = pd.DataFrame(embeddings, columns=embed_cols, index=df.index)
        features = pd.concat([features, embed_df], axis=1)
    
    # Transform boolean features
    for col in model['boolean_features']:
        features[col] = df[col].fillna(False).astype(int)
    
    # Transform genre features
    if model['genre_features'] and model['mlb'] and hasattr(model['mlb'], 'classes_'):
        genre_col = model['genre_features'][0]
        genres = df[genre_col].apply(process_genres)
        genre_matrix = model['mlb'].transform(genres)
        genre_df = pd.DataFrame(genre_matrix, 
                               columns=model['mlb'].classes_,
                               index=df.index)
        features = pd.concat([features, genre_df], axis=1)
    
    return features

def get_recommendations(top_songs_df, dataset_df, model, k=10):
    """Get song recommendations based on top songs"""
    # Transform features
    top_songs_features = transform_features(top_songs_df, model)
    dataset_features = transform_features(dataset_df, model)
    
    # Apply weights
    # For numeric features
    if model['numeric_features']:
        top_songs_features[model['numeric_features']] *= model['weights']['numeric']
        dataset_features[model['numeric_features']] *= model['weights']['numeric']
    
    # For categorical features
    cat_embed_cols = [col for col in top_songs_features.columns if 'embed' in col]
    if cat_embed_cols:
        top_songs_features[cat_embed_cols] *= model['weights']['categorical']
        dataset_features[cat_embed_cols] *= model['weights']['categorical']
    
    # For boolean features
    if model['boolean_features']:
        top_songs_features[model['boolean_features']] *= model['weights']['boolean']
        dataset_features[model['boolean_features']] *= model['weights']['boolean']
    
    # For genre features
    if model['mlb'] and hasattr(model['mlb'], 'classes_'):
        genre_cols = list(model['mlb'].classes_)
        if genre_cols:
            top_songs_features[genre_cols] *= model['weights']['genre']
            dataset_features[genre_cols] *= model['weights']['genre']
    
    # Calculate similarities
    top_songs_vector = top_songs_features.mean(axis=0).values.reshape(1, -1)
    similarities = cosine_similarity(top_songs_vector, dataset_features)
    
    # Get recommendations
    top_indices = similarities.flatten().argsort()[-k:][::-1]
    recommendations = dataset_df.iloc[top_indices].copy()
    recommendations['similarity_score'] = similarities.flatten()[top_indices]
    
    # Prepare output
    output_cols = ['similarity_score', 'track_name']
    for feature_list in [model['categorical_features'], model['numeric_features'], model['genre_features']]:
        output_cols.extend([col for col in feature_list if col in recommendations.columns])
    
    existing_cols = [col for col in output_cols if col in recommendations.columns]
    final_recommendations = recommendations[existing_cols].copy()
    final_recommendations['similarity_score'] = final_recommendations['similarity_score'].round(4)
    
    return final_recommendations

if __name__ == "__main__":
    try:
        # Load datasets
        top_songs_df = pd.read_csv('top_100_tracks_with_genres.csv')
        dataset_df = pd.read_csv('general_huge_dataset.csv')
        
        # Configure recommender
        weights = {
            'numeric': 10.0,
            'categorical': 10.0,
            'boolean': 10.0,
            'genre': 10.0
        }
        
        # Prepare the recommender model
        model = prepare_recommender(top_songs_df, dataset_df, feature_weights=weights)
        
        # Get recommendations
        recommendations = get_recommendations(top_songs_df, dataset_df, model, k=15)
        
        # Save results
        output_filename = 'song_recommendations.csv'
        recommendations.to_csv(output_filename, index=False)
        logger.info(f"Recommendations saved to {output_filename}")
        logger.info("\nFirst few recommendations:")
        logger.info(recommendations.head())
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)