import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, LabelEncoder
import numpy as np
from datetime import datetime

class FeatureProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.mlb = None
        self.scaler = StandardScaler()
        self.all_genres = set()
        
    def clean_categorical_value(self, value):
        """Clean and handle null values in categorical data"""
        if pd.isna(value) or value == '':
            return 'unknown'
        return str(value).strip()
    
    def fit_categorical_encoders(self, df1, df2, categorical_cols):
        """Fit label encoders on combined unique values from both datasets"""
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            # Clean and combine values from both datasets
            values1 = df1[col].apply(self.clean_categorical_value)
            values2 = df2[col].apply(self.clean_categorical_value)
            combined_values = pd.concat([values1, values2]).unique()
            # Add 'unknown' category explicitly
            combined_values = np.append(combined_values, 'unknown')
            self.label_encoders[col].fit(combined_values)
    
    def clean_genre_string(self, genre_str):
        """Clean and handle null values in genre strings"""
        if pd.isna(genre_str) or not str(genre_str).strip('[]'):
            return []
        try:
            # Clean genre string and split into list
            genres = str(genre_str).strip('[]').split(',')
            return [g.strip().strip("'").strip('"') for g in genres if g.strip()]
        except:
            return []
    
    def fit_genre_encoder(self, df1, df2, genre_col):
        """Fit MultiLabelBinarizer on genres from both datasets"""
        # Process genres from both datasets
        genres1 = df1[genre_col].apply(self.clean_genre_string)
        genres2 = df2[genre_col].apply(self.clean_genre_string)
        
        # Combine all genres
        all_genres = []
        for genre_list in genres1:
            all_genres.extend(genre_list)
        for genre_list in genres2:
            all_genres.extend(genre_list)
            
        # Clean genres and remove empty strings
        all_genres = [g for g in all_genres if g]
        self.all_genres = sorted(set(all_genres))
        
        if not self.all_genres:
            # If no valid genres found, create a dummy genre to prevent errors
            self.all_genres = ['unknown_genre']
        
        # Fit MLBinarizer
        self.mlb = MultiLabelBinarizer(classes=list(self.all_genres))
        self.mlb.fit([self.all_genres])
    
    def transform_features(self, df, feature_cols):
        """Transform features using fitted encoders"""
        processed_features = pd.DataFrame()
        
        # Process numeric features
        if feature_cols['numeric']:
            # Fill NaN values with mean
            numeric_data = df[feature_cols['numeric']].fillna(df[feature_cols['numeric']].mean())
            numeric_data = self.scaler.transform(numeric_data)
            processed_features = pd.DataFrame(numeric_data, columns=feature_cols['numeric'])
        
        # Process categorical features
        if feature_cols['categorical']:
            for col in feature_cols['categorical']:
                values = df[col].apply(self.clean_categorical_value)
                encoded_col = self.label_encoders[col].transform(values)
                processed_features[f"{col}_encoded"] = encoded_col
        
        # Process date features
        if feature_cols['date']:
            current_year = datetime.now().year
            for col in feature_cols['date']:
                dates = pd.to_datetime(df[col], errors='coerce')
                years = dates.dt.year.fillna(current_year)  # Fill NaN with current year
                processed_features[f"{col}_years"] = current_year - years
        
        # Process boolean features
        if feature_cols['boolean']:
            for col in feature_cols['boolean']:
                processed_features[col] = df[col].fillna(False).astype(int)
        
        # Process genre features
        if feature_cols['genre'] and self.mlb is not None:
            for col in feature_cols['genre']:
                genres = df[col].apply(self.clean_genre_string)
                genre_matrix = self.mlb.transform(genres)
                genre_df = pd.DataFrame(genre_matrix, columns=self.mlb.classes_)
                processed_features = pd.concat([processed_features, genre_df], axis=1)
        
        return processed_features

def get_common_features(top_songs_df, dataset_df):
    """Identify common features between top songs and dataset"""
    numeric_features = ['Track Popularity', 'Track Duration', 'duration_ms', 'popularity', 
                       'danceability', 'energy', 'key', 'loudness', 'mode', 
                       'speechiness', 'acousticness', 'instrumentalness', 
                       'liveness', 'valence', 'tempo']
    
    categorical_features = ['Artist Name', 'artists', 'Album Name', 'album_name', 
                          'Track Type', 'track_type', 'Album Type', 'album_type']
    
    date_features = ['Album Release Date', 'Album Release', 'release_date']
    
    boolean_features = ['Explicit', 'explicit', 'Is Local', 'is_local']
    
    genre_features = ['Artist Genres', 'Artist Genre', 'track_genre', 'genres']
    
    top_cols = set(top_songs_df.columns)
    dataset_cols = set(dataset_df.columns)
    
    return {
        'numeric': [col for col in numeric_features if col in top_cols and col in dataset_cols],
        'categorical': [col for col in categorical_features if col in top_cols and col in dataset_cols],
        'date': [col for col in date_features if col in top_cols and col in dataset_cols],
        'boolean': [col for col in boolean_features if col in top_cols and col in dataset_cols],
        'genre': [col for col in genre_features if col in top_cols and col in dataset_cols]
    }

def flexible_song_recommender(top_songs_df, dataset_df, k=10, weights=None):
    """Recommend songs from any dataset based on common features with top songs"""
    # Get common features
    common_features = get_common_features(top_songs_df, dataset_df)
    
    # Validate features
    total_features = sum(len(features) for features in common_features.values())
    if total_features == 0:
        raise ValueError("No common features found between the datasets")
    
    # Default weights
    if weights is None:
        weights = {
            'numeric': 1.0,
            'categorical': 0.8,
            'date': 0.6,
            'boolean': 0.4,
            'genre': 0.9
        }
    
    # Initialize and fit feature processor
    processor = FeatureProcessor()
    
    # Fit all encoders
    if common_features['categorical']:
        processor.fit_categorical_encoders(top_songs_df, dataset_df, common_features['categorical'])
    if common_features['genre']:
        processor.fit_genre_encoder(top_songs_df, dataset_df, common_features['genre'][0])
    if common_features['numeric']:
        combined_numeric = pd.concat([
            top_songs_df[common_features['numeric']].fillna(0),
            dataset_df[common_features['numeric']].fillna(0)
        ])
        processor.scaler.fit(combined_numeric)
    
    # Transform features
    top_songs_features = processor.transform_features(top_songs_df, common_features)
    dataset_features = processor.transform_features(dataset_df, common_features)
    
    # Apply weights
    for feature_type, weight in weights.items():
        if feature_type == 'genre' and processor.mlb is not None:
            genre_columns = processor.mlb.classes_
            top_songs_features[genre_columns] *= weight
            dataset_features[genre_columns] *= weight
        else:
            feature_columns = [col for col in top_songs_features.columns 
                             if any(feat in col for feat in common_features[feature_type])]
            top_songs_features[feature_columns] *= weight
            dataset_features[feature_columns] *= weight
    
    # Calculate similarities
    top_songs_vector = top_songs_features.mean(axis=0).values.reshape(1, -1)
    similarities = cosine_similarity(top_songs_vector, dataset_features)
    
    # Get recommendations
    top_indices = similarities.flatten().argsort()[-k:][::-1]
    
    # Create comprehensive recommendations DataFrame
    recommended_songs = dataset_df.iloc[top_indices].copy()
    recommended_songs['similarity_score'] = similarities.flatten()[top_indices]
    
    # Get all common features
    all_common_columns = []
    for feature_list in common_features.values():
        all_common_columns.extend(feature_list)
    
    # Add similarity score to the beginning
    columns_to_show = ['similarity_score']
    
    # Add common features
    columns_to_show.extend([col for col in all_common_columns if col in recommended_songs.columns])
    
    # Reorder columns and select only common features plus similarity score
    final_recommendations = recommended_songs[columns_to_show].copy()
    
    # Round similarity scores for better readability
    final_recommendations['similarity_score'] = final_recommendations['similarity_score'].round(4)
    
    return final_recommendations

if __name__ == "__main__":
    try:
        # Load datasets
        top_songs_df = pd.read_csv('top_100_tracks_with_genres.csv')
        dataset_df = pd.read_csv('dataset.csv')
        
        # Get recommendations
        recommendations = flexible_song_recommender(top_songs_df, dataset_df, k=10)
        
        # Save to CSV
        output_filename = 'song_recommendations.csv'
        recommendations.to_csv(output_filename, index=False)
        print(f"\nRecommendations saved to {output_filename}")
        print("\nFirst few recommendations:")
        print(recommendations.head())
            
    except Exception as e:
        print(f"Error: {e}")