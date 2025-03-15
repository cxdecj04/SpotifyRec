import os
import io
from flask import Flask, request, session, redirect, url_for, send_file, jsonify, Response, make_response
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import FlaskSessionCacheHandler
import pandas as pd

# Flask Setup
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(64)

# Define Spotify OAuth scope
scope = 'playlist-read-private user-library-read playlist-read-collaborative'

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

# Global variable to store CSV data
csv_data_store = None

@app.route('/')
def home():
    if not sp_oauth.validate_token(cache_handler.get_cached_token()):
        return redirect(url_for('login'))
    return redirect(url_for('get_user_playlists'))

@app.route('/login')
def login():
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/callback')
def callback():
    code = request.args['code']
    token_info = sp_oauth.get_access_token(code)
    session['token_info'] = token_info
    return redirect(url_for('get_user_playlists'))

@app.route('/get_user_playlists')
def get_user_playlists():
    global csv_data_store

    token_info = session.get('token_info', {})
    if not token_info:
        return redirect(url_for('login'))

    access_token = token_info['access_token']
    sp = Spotify(auth=access_token)

    # Get the playlists
    playlists = sp.current_user_playlists()
    
    # If no playlists are found
    if not playlists['items']:
        return "You have no playlists."

    # Select the first playlist (you can modify this to select a specific playlist)
    playlist = playlists['items'][10]
    playlist_name = playlist['name']
    playlist_id = playlist['id']
    
    playlist_data = []
    
    # Paginate through playlist tracks (as playlists can have more than 100 tracks)
    offset = 0
    limit = 100

    while True:
        tracks = sp.playlist_tracks(playlist_id, offset=offset, limit=limit)
        if not tracks['items']:
            break

        for track in tracks['items']:
            # Fetch track details
            track_name = track['track']['name']
            track_url = track['track']['external_urls']['spotify']
            track_uri = track['track']['uri']
            track_number = track['track']['track_number']
            is_local = track['track']['is_local']
            track_duration_ms = track['track']['duration_ms']
            track_popularity = track['track']['popularity']
            preview_url = track['track']['preview_url']
            explicit = track['track']['explicit']
            external_id_isrc = track['track']['external_ids'].get('isrc', 'N/A')
            track_type = track['track']['type']

            # Fetch album details
            album = track['track']['album']
            album_name = album['name']
            album_type = album['type']
            total_tracks_in_album = album['total_tracks']
            available_markets = ', '.join(album['available_markets'])
            album_url = album['external_urls']['spotify']
            album_release_date = album['release_date']

            # Fetch artist details
            artists = track['track']['artists']
            artist_names = ', '.join([artist['name'] for artist in artists])
            artist_genres = []
            
            for artist in artists:
                artist_info = sp.artist(artist['id'])
                artist_genres.append(', '.join(artist_info['genres']))
            
            genres = ', '.join(artist_genres)

            # Append all details to playlist_data
            track_info = {
                'track_name': track_name,
                'artists': artist_names,
                'track_genre': genres,
                'album_name': album_name,
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
            playlist_data.append(track_info)

        offset += limit
    
    # Convert the playlist data into a DataFrame
    df_playlists = pd.DataFrame(playlist_data)

    # Save CSV in memory (not on disk)
    csv_buffer = io.StringIO()
    df_playlists.to_csv(csv_buffer, index=False, encoding='utf-8')
    csv_buffer.seek(0)  # Reset buffer position

    # Store the CSV data in both session and global variable
    csv_data_store = csv_buffer.getvalue()
    session['csv_data'] = csv_data_store
    
    # Force session to be saved
    session.modified = True

    # Summary Statistics
    total_tracks = len(df_playlists)
    playlist_name = playlist_name  # Name of the selected playlist

    # HTML Output
    playlist_tracks_html = df_playlists.to_html(classes='table table-striped', index=False)

    stats_html = f"""
    <h3>Statistics for Playlist: {playlist_name}</h3>
    <p>Total Tracks: {total_tracks}</p>
    """

    # Add a link to download the CSV
    download_link = f"""
    <p><a href="/download_playlist_csv" target="_blank">Download Playlist Tracks as CSV</a></p>
    """

    return stats_html + download_link + playlist_tracks_html

@app.route('/download_playlist_csv')
def download_playlist_csv():
    global csv_data_store

    token_info = session.get('token_info', {})
    if not token_info:
        return redirect(url_for('login'))

    # Try to get CSV data from session first, fall back to global variable
    csv_data = session.get('csv_data') or csv_data_store
    
    if not csv_data:
        return jsonify({"error": "No CSV data available. Please generate the data first."}), 400

    response = Response(csv_data, content_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=playlist_tracks.csv"
    return response

@app.route('/refresh_token')
def refresh_token():
    token_info = session.get('token_info', {})
    if sp_oauth.is_token_expired(token_info):
        token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
        session['token_info'] = token_info
    return redirect(url_for('get_user_playlists'))

if __name__ == '__main__':
    app.run(debug=True)