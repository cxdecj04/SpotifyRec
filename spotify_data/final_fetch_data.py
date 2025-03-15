import os
import time as pytime
import pandas as pd
import io
from datetime import timedelta
from collections import Counter
from flask import Flask, request, session, redirect, url_for, jsonify, Response, make_response
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import FlaskSessionCacheHandler

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(64)
app.permanent_session_lifetime = timedelta(hours=1)

# Define Spotify OAuth scope
scope = 'user-top-read playlist-read-private user-library-read'
cache_handler = FlaskSessionCacheHandler(session)
sp_oauth = SpotifyOAuth(
    client_id='04ea92fc87454078bbe7fd772b20edf7',
    client_secret='ab4538e941284c92aa4b4bfdbf6fc608',
    redirect_uri='http://localhost:5000/callback',
    scope=scope,
    cache_handler=cache_handler,
    show_dialog=True
)

# Global variable to store CSV data
# This avoids session storage issues but means data is shared across all users
# For production, you'd want to use a proper database or user-specific storage
csv_data_store = None

def validate_token():
    """Validate and refresh the Spotify token if necessary."""
    try:
        token_info = session.get('token_info')
        if not token_info:
            return None
        
        now = int(pytime.time())
        is_expired = token_info.get('expires_at', now) - now < 60
        
        if is_expired and token_info.get('refresh_token'):
            token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
            session['token_info'] = token_info

        return token_info
    except Exception as e:
        print(f"Error validating token: {e}")
        return None

@app.route('/')
def home():
    session.pop('token_info', None)  # Clear any old session data
    return redirect(url_for('login'))

@app.route('/login')
def login():
    session.clear()  # Ensure a fresh session
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/callback')
def callback():
    """Handle the Spotify authentication callback."""
    if 'code' not in request.args:
        return jsonify({"error": "No authorization code provided"}), 400
    
    try:
        code = request.args['code']
        token_info = sp_oauth.get_access_token(code)
        if not token_info.get('refresh_token'):
            return jsonify({"error": "No refresh token found. Please reauthorize the app."}), 400
        
        session['token_info'] = token_info
        return redirect(url_for('get_user_data', n_tracks=100))
    
    except Exception as e:
        return jsonify({"error": f"Authorization failed: {str(e)}"}), 400

@app.route('/get_user_data/<int:n_tracks>', methods=['GET'])
def get_user_data(n_tracks):
    """Fetch user's top tracks from Spotify and generate CSV in memory."""
    global csv_data_store
    
    start_time = pytime.time()
    
    token_info = validate_token()
    if not token_info:
        return redirect(url_for('login'))

    try:
        sp = Spotify(auth=token_info['access_token'], requests_timeout=30)
        track_data = []
        limit = 50  # Max tracks per request
        offset = 0

        while len(track_data) < n_tracks:
            top_tracks = sp.current_user_top_tracks(limit=limit, offset=offset, time_range='long_term')
            if not top_tracks['items']:
                break

            for track in top_tracks['items']:
                # Fetch artist details
                artist_name = track['artists'][0]['name']
                artist_info = sp.artist(track['artists'][0]['uri'])
                artist_genres = ', '.join(artist_info['genres'])

                # Fetch album details
                album_info = sp.album(track['album']['uri'])
                album_type = album_info['type']
                total_tracks_in_album = album_info['total_tracks']
                available_markets = ', '.join(album_info['available_markets'])
                album_url = album_info['external_urls']['spotify']
                album_release_date = album_info['release_date']

                # Track details
                track_name = track['name']
                track_duration_ms = track['duration_ms']
                track_popularity = track['popularity']
                preview_url = track['preview_url']
                explicit = track['explicit']
                external_id_isrc = track['external_ids'].get('isrc', 'N/A')
                track_url = track['external_urls']['spotify']
                track_number = track['track_number']
                track_type = track['type']
                track_uri = track['uri']
                is_local = track['is_local']

                # Append all details to track_data
                track_info = {
                    'track_name': track_name,
                    'artists': artist_name,
                    'track_genre': artist_genres,
                    'album_name': track['album']['name'],
                    'Album Type': album_type,
                    'Total Tracks in Album': total_tracks_in_album,
                    'Available Markets': available_markets,
                    'Album URL': album_url,
                    'Album Release Date': album_release_date,
                    'Track Duration (ms)': track_duration_ms,
                    'popularity': track_popularity,
                    'Preview URL': preview_url,
                    'explicit': explicit,
                    'External ID (ISRC)': external_id_isrc,
                    'Track URL': track_url,
                    'Track Number': track_number,
                    'Track Type': track_type,
                    'Track URI': track_uri,
                    'Is Local': is_local
                }
                track_data.append(track_info)

                if len(track_data) >= n_tracks:
                    break

            offset += limit

        # Convert to DataFrame
        df_tracks = pd.DataFrame(track_data)

        # Save CSV in memory (not on disk)
        csv_buffer = io.StringIO()
        df_tracks.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_buffer.seek(0)  # Reset buffer position

        # Store the CSV data in both session and global variable
        csv_data_store = csv_buffer.getvalue()
        session['csv_data'] = csv_data_store
        
        # Force session to be saved
        session.modified = True

        # Compute statistics
        avg_duration_minutes = df_tracks['Track Duration (ms)'].mean() / 60000
        avg_popularity = df_tracks['popularity'].mean()
        most_popular_song = df_tracks.loc[df_tracks['popularity'].idxmax()]
        least_popular_song = df_tracks.loc[df_tracks['popularity'].idxmin()]
        
        # Top 5 genres
        genres = ', '.join(df_tracks['track_genre']).split(', ')
        top_5_genres = Counter(genres).most_common(5)

        # Oldest & newest songs
        oldest_song = df_tracks.loc[df_tracks['Album Release Date'].idxmin()]
        newest_song = df_tracks.loc[df_tracks['Album Release Date'].idxmax()]

        end_time = pytime.time()
        response_time = end_time - start_time

        # Generate HTML response
        stats_html = f"""
        <h3>Statistics:</h3>
        <p>Average Track Duration: {avg_duration_minutes:.2f} minutes</p>
        <p>Average Popularity: {avg_popularity:.2f}</p>
        <p>Response Time: {response_time:.2f} seconds</p>
        <p>Most Popular Song: {most_popular_song['track_name']} by {most_popular_song['artists']} (Popularity: {most_popular_song['popularity']})</p>
        <p>Least Popular Song: {least_popular_song['track_name']} by {least_popular_song['artists']} (Popularity: {least_popular_song['popularity']})</p>
        <p>Oldest Song: {oldest_song['track_name']} by {oldest_song['artists']} (Release Date: {oldest_song['Album Release Date']})</p>
        <p>Newest Song: {newest_song['track_name']} by {newest_song['artists']} (Release Date: {newest_song['Album Release Date']})</p>
        <h4>Top 5 Genres:</h4>
        <ul>{''.join([f'<li>{genre[0]}: {genre[1]} occurrences</li>' for genre in top_5_genres])}</ul>
        <p><a href="/download_csv" target="_blank">Download CSV</a></p>
        """

        # Create response with the HTML content
        response = make_response(stats_html + df_tracks.to_html(classes='table table-striped', index=False))
        
        # Set cookie to indicate CSV data is available
        response.set_cookie('has_csv_data', 'true')
        
        return response

    except Exception as e:
        return jsonify({"error": f"Failed to fetch user data: {str(e)}"}), 500
@app.route('/download_csv')
def download_csv():
    """Serve the CSV file stored in memory."""
    global csv_data_store
    
    token_info = validate_token()
    if not token_info:
        return redirect(url_for('login'))

    # Try to get CSV data from session first, fall back to global variable
    csv_data = session.get('csv_data') or csv_data_store
    
    if not csv_data:
        return jsonify({"error": "No CSV data available. Please generate the data first."}), 400

    response = Response(csv_data, content_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=top_tracks.csv"
    return response

if __name__ == '__main__':
    app.run(debug=True)