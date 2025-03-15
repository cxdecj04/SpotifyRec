import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Union, Optional, Set, Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MusicGenreClassifier:
    """
    A class to analyze and classify music tracks by genre and region (Indian/Western).
    """
    # Define keywords for Indian music identification
    INDIAN_KEYWORDS = {
        'indian', 'desi', 'punjabi', 'hindi', 'bollywood', 'bhangra',
        'carnatic', 'hindustani', 'tamil', 'telugu', 'malayalam',
        'kannada', 'marathi', 'gujarati', 'bengali', 'urdu', 'pakistani', 'sufi', 'filmi'
    }
    
    def __init__(self, data: Union[pd.DataFrame, str, Path]):
        """Initialize the classifier with data."""
        self.data = self._load_data(data)
        self._validate_data()
    
    def _load_data(self, data: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
        """Load data from various input types."""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if path.suffix == '.csv':
            return pd.read_csv(path)
        elif path.suffix in ['.xlsx', '.xls']:
            return pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _validate_data(self):
        """Validate the data structure."""
        required_columns = {'track_id', 'artists', 'track_genre'}
        missing_columns = required_columns - set(self.data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def classify_music_origin(self,
                            genre: Optional[str] = None,
                            additional_indian_keywords: Optional[Set[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Classify music as Indian or Western based on genre and artist information.
        """
        # Combine default and additional Indian keywords
        indian_keywords = self.INDIAN_KEYWORDS.copy()
        if additional_indian_keywords:
            indian_keywords.update(additional_indian_keywords)
        
        # Create working copy of data
        df = self.data.copy()
        
        # Apply genre filter if specified
        if genre:
            genre_mask = df['track_genre'].str.contains(genre, case=False, na=False)
            df = df[genre_mask]
        
        # Create Indian music mask using combined fields
        def is_indian(row):
            genre_str = str(row['track_genre']).lower()
            artist_str = str(row['artists']).lower()
            return any(keyword in genre_str or keyword in artist_str
                      for keyword in indian_keywords)
        
        # Apply classification
        indian_mask = df.apply(is_indian, axis=1)
        
        # Split into Indian and Western music
        indian_music = df[indian_mask].copy()
        western_music = df[~indian_mask].copy()
        
        # Reset indices
        indian_music.reset_index(drop=True, inplace=True)
        western_music.reset_index(drop=True, inplace=True)
        
        # Log results
        logger.info(f"Total tracks processed: {len(df)}")
        logger.info(f"Indian music tracks: {len(indian_music)}")
        logger.info(f"Western music tracks: {len(western_music)}")
        
        return {
            'indian': indian_music,
            'western': western_music
        }

def analyze_music_csv(csv_path: str):
    """
    Analyze a music CSV file and print comprehensive statistics.
    """
    try:
        # Initialize the classifier
        classifier = MusicGenreClassifier(csv_path)
        
        # Get all music classifications
        classified_music = classifier.classify_music_origin()
        
        # Generate basic statistics
        total_tracks = len(classifier.data)
        indian_tracks = len(classified_music['indian'])
        western_tracks = len(classified_music['western'])
        
        # Print overall statistics
        print("\n=== MUSIC CLASSIFICATION RESULTS ===")
        print(f"Total tracks analyzed: {total_tracks:,}")
        print(f"Indian tracks: {indian_tracks:,} ({(indian_tracks/total_tracks*100):.1f}%)")
        print(f"Western tracks: {western_tracks:,} ({(western_tracks/total_tracks*100):.1f}%)")
        
        # Analyze genres
        print("\n=== GENRE BREAKDOWN ===")
        for category in ['indian', 'western']:
            df = classified_music[category]
            print(f"\nTop 5 {category.title()} Genres:")
            genre_counts = df['track_genre'].value_counts()
            for genre, count in genre_counts.head().items():
                print(f"  {genre}: {count:,} tracks")
        
        # Analyze artists
        print("\n=== ARTIST BREAKDOWN ===")
        for category in ['indian', 'western']:
            df = classified_music[category]
            print(f"\nTop 5 {category.title()} Artists:")
            artist_counts = df['artists'].value_counts()
            for artist, count in artist_counts.head().items():
                print(f"  {artist}: {count:,} tracks")
        
        # Calculate unique statistics
        print("\n=== DIVERSITY METRICS ===")
        for category in ['indian', 'western']:
            df = classified_music[category]
            unique_artists = df['artists'].nunique()
            unique_genres = df['track_genre'].nunique()
            print(f"\n{category.title()} Music:")
            print(f"  Unique Artists: {unique_artists:,}")
            print(f"  Unique Genres: {unique_genres:,}")
            print(f"  Artist-to-Track Ratio: {(unique_artists/len(df)*100):.1f}%")
            print(f"  Genre-to-Track Ratio: {(unique_genres/len(df)*100):.1f}%")
            
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
        print(f"Error: Could not find the file {csv_path}")
    except Exception as e:
        logger.error(f"Error analyzing music data: {str(e)}")
        print(f"Error: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Replace with your CSV file path
    csv_file = "top_100_tracks_with_genres.csv"
    analyze_music_csv(csv_file)