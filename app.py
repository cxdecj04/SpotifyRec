import os
import requests
from flask import Flask, request, session, redirect, url_for, send_file, jsonify
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import FlaskSessionCacheHandler
import pandas as pd
from datetime import timedelta
import time as pytime
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(64)
app.permanent_session_lifetime = timedelta(hours=1)

# Define Spotify OAuth scope
scope = 'user-top-read playlist-read-private user-library-read'

# Cache Handler using Flask session
cache_handler = FlaskSessionCacheHandler(session)

# Spotify OAuth setup
sp_oauth = SpotifyOAuth(
    client_id='04ea92fc87454078bbe7fd772b20edf7',
    client_secret='ab4538e941284c92aa4b4bfdbf6fc608',
    redirect_uri='http://localhost:5000/callback',
    scope=scope,
    cache_handler=cache_handler,
    show_dialog=True
)

def validate_token():
    """Validate and refresh token if necessary."""
    try:
        token_info = session.get('token_info')
        
        if not token_info:
            return None
            
        # Check if token is expired
        now = int(pytime.time())
        is_expired = token_info.get('expires_at', now) - now < 60
        
        if is_expired and token_info.get('refresh_token'):
            try:
                token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
                session['token_info'] = token_info
            except Exception as e:
                print(f"Error refreshing token: {e}")
                return None
                
        return token_info
    except Exception as e:
        print(f"Error validating token: {e}")
        return None

@app.route('/')
def home():
    token_info = validate_token()
    if not token_info:
        return redirect(url_for('login'))
    return redirect(url_for('get_user_data', n_tracks=100))  # Default fetch of 1000 tracks

@app.route('/login')
def login():
    session.clear()
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/callback')
def callback():
    try:
        if 'code' not in request.args:
            return jsonify({"error": "No authorization code provided"}), 400
            
        code = request.args['code']
        token_info = sp_oauth.get_access_token(code)
        
        if not token_info.get('refresh_token'):
            return jsonify({
                "error": "No refresh token found. Please ensure you've authorized the application."
            }), 400
            
        session['token_info'] = token_info
        return redirect(url_for('get_user_data', n_tracks=100))  # Default fetch of 1000 tracks
        
    except Exception as e:
        return jsonify({"error": f"Authorization failed: {str(e)}"}), 400

@app.route('/get_user_data/<int:n_tracks>', methods=['GET'])
def get_user_data(n_tracks):
    start_time = pytime.time()  # Record the start time

    token_info = validate_token()
    if not token_info:
        return redirect(url_for('login'))

    try:
        sp = Spotify(auth=token_info['access_token'], requests_timeout=30)  # Timeout set to 30 seconds

        track_data = []
        limit = 50  # Spotify API limit per request
        offset = 0
        total = 0

        while len(track_data) < n_tracks:
            top_tracks = sp.current_user_top_tracks(limit=limit, offset=offset, time_range='short_term')
            if not top_tracks['items']:
                break

            for track in top_tracks['items']:
                artist_name = track['artists'][0]['name']
                total += 1

                # Fetch artist genres from Spotify API
                artist_info = sp.artist(track['artists'][0]['uri'])
                artist_genres = ', '.join(artist_info['genres'])  # Joining genres for the artist

                track_info = {
                    'Track Name': track['name'],
                    'Artist Name': artist_name,
                    'Artist Genres': artist_genres,  # List of artist genres as a string
                    'Album Name': track['album']['name'],
                    'Album Type': track['album']['album_type'],
                    'Total Tracks in Album': track['album']['total_tracks'],
                    'Available Markets': ', '.join(track['album']['available_markets']),
                    'Album URL': track['album']['external_urls']['spotify'],
                    'Album Release Date': track['album']['release_date'],
                    'Track Duration (ms)': track['duration_ms'],
                    'Track Popularity': track['popularity'],
                    'Preview URL': track.get('preview_url', 'N/A'),
                    'Explicit': track['explicit'],
                    'External ID (ISRC)': track['external_ids'].get('isrc', 'N/A'),
                    'Track URL': track['external_urls']['spotify'],
                    'Track Number': track['track_number'],
                    'Track Type': track['type'],
                    'Track URI': track['uri'],
                    'Is Local': track['is_local']
                }
                track_data.append(track_info)

            offset += limit

        df_tracks = pd.DataFrame(track_data)
        
        # Further processing and recommendations can be done after this
        # Creating the CSV file and generating recommendations

        return "Your top tracks and their genres are successfully fetched!"  # Modify this as necessary

    except Exception as e:
        return jsonify({"error": f"Failed to fetch user data: {str(e)}"}), 500

def generate_recommendations_from_top_tracks(top_tracks_df):
    """ 
    Generate song recommendations based on the user's top tracks.

    Parameters: 
        top_tracks_df (pandas DataFrame): DataFrame of the top tracks (e.g., top 1000 tracks).

    Returns:
        non_playlist_df_top_40: Top 40 recommendations based on cosine similarity.
    """
    # Assuming `df` contains all available songs
    df = get_all_songs_data()

    # Extract features from the top tracks (ignoring 'id' column for the calculation)
    top_tracks_feature_set = df[df['id'].isin(top_tracks_df['id'].values)].copy()
    top_tracks_feature_set = top_tracks_feature_set.merge(top_tracks_df[['id', 'date_added']], on='id', how='inner')

    # Sort by the date_added to apply recency weighting
    top_tracks_feature_set = top_tracks_feature_set.sort_values('date_added', ascending=False)
    most_recent_date = top_tracks_feature_set.iloc[0]['date_added']
    
    # Calculate months from the most recent song and apply recency weighting
    top_tracks_feature_set['months_from_recent'] = top_tracks_feature_set['date_added'].apply(
        lambda x: (most_recent_date - x).days // 30
    )
    top_tracks_feature_set['weight'] = top_tracks_feature_set['months_from_recent'].apply(
        lambda x: 1.09 ** (-x)  # Apply weight factor of 1.09 for recency
    )

    # Multiply features by their weight
    top_tracks_feature_set_weighted = top_tracks_feature_set.copy()
    top_tracks_feature_set_weighted.update(top_tracks_feature_set_weighted.iloc[:, :-4].mul(top_tracks_feature_set_weighted['weight'], axis=0))

    # Create the final weighted feature vector for the user's top 1000 songs
    top_tracks_feature_vector = top_tracks_feature_set_weighted.iloc[:, :-4].sum(axis=0)

    # Now, calculate the similarity of each song in the non-playlist dataset
    non_playlist_df = df[~df['id'].isin(top_tracks_df['id'].values)]  # Exclude top tracks from recommendations
    non_playlist_df['sim'] = cosine_similarity(
        non_playlist_df.drop('id', axis=1).values, top_tracks_feature_vector.values.reshape(1, -1)
    )[:, 0]

    # Sort the recommendations by similarity and select the top 40
    non_playlist_df_top_40 = non_playlist_df.sort_values('sim', ascending=False).head(40)

    return non_playlist_df_top_40

def get_all_songs_data():
    """ 
    Dummy function to get all available songs in the database.
    In reality, you should retrieve this from Spotify or your local dataset.
    """
    # Replace this with actual code to fetch data from Spotify API or your database
    return pd.DataFrame()  # Placeholder

if __name__ == '__main__':
    app.run(debug=True)
