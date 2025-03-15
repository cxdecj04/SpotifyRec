import os
from flask import Flask, request, session, redirect, url_for, send_file
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import FlaskSessionCacheHandler
import pandas as pd

# Flask Setup
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(64)

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

@app.route('/')
def home():
    if not sp_oauth.validate_token(cache_handler.get_cached_token()):
        return redirect(url_for('login'))
    return redirect(url_for('get_user_data'))

@app.route('/login')
def login():
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/callback')
def callback():
    code = request.args['code']
    token_info = sp_oauth.get_access_token(code)
    session['token_info'] = token_info
    return redirect(url_for('get_user_data'))

@app.route('/get_user_data')
def get_user_data():
    token_info = session.get('token_info', {})
    if not token_info:
        return redirect(url_for('login'))

    # Get the custom number of tracks from the request, default to 50 if not provided
    n_tracks = request.args.get('n', default=50, type=int)

    access_token = token_info['access_token']
    sp = Spotify(auth=access_token)

    # Paginate to fetch more than 50 top tracks
    track_data = []
    limit = 50
    offset = 0

    while len(track_data) < n_tracks:
        top_tracks = sp.current_user_top_tracks(limit=limit, offset=offset, time_range='short_term')
        if not top_tracks['items']:
            break  # Exit if no more items

        for track in top_tracks['items']:
            # album_url=track['album']['external_urls']['spotify']
            # album_id = album_url.split('/')[-1].split('?')[0]
            # genres = get_spotify_album_genre(album_url, client_id, client_secret)
            # print("Genres:", genres)
            track_info = {
                'Track Name': track['name'],
                'Artist Name': track['artists'][0]['name'],
                'Album Name': track['album']['name'],
                'Album Type': track['album']['album_type'],
                'Total Tracks in Album': track['album']['total_tracks'],
                'Available Markets': ', '.join(track['album']['available_markets']),
                'Album URL': track['album']['external_urls']['spotify'],
                'Album Release Date': track['album']['release_date'],
                'Album Release Precision': track['album']['release_date_precision'],
                'Album URI': track['album']['uri'],
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

            if len(track_data) >= n_tracks:  # Stop if we've reached the requested number of tracks
                break

        offset += limit

    # Convert to DataFrame
    df_tracks = pd.DataFrame(track_data)

    # Save DataFrame to CSV
    csv_path = 'top_tracks.csv'
    df_tracks.to_csv(csv_path, index=False)

    # Summary Statistics
    # avg_duration_minutes = df_tracks['Track Duration (ms)'].mean() / 60000
    # avg_popularity = df_tracks['Track Popularity'].mean()

    # HTML Output
    top_tracks_html = df_tracks.to_html(classes='table table-striped', index=False)

    stats_html = f"""
    <h3>Statistics:</h3>

    """

    # Add a link to download the CSV
    download_link = f"""
    <p><a href="/download_csv" target="_blank">Download Top Tracks as CSV</a></p>
    """

    return stats_html + download_link + top_tracks_html

@app.route('/get_user_data_input')
def get_user_data_input():
    # Render a simple form for the user to input the desired number of tracks
    form_html = """
    <form action="/get_user_data" method="get">
        <label for="n">Enter the number of tracks to fetch:</label>
        <input type="number" id="n" name="n" min="1" value="50">
        <button type="submit">Submit</button>
    </form>
    """
    return form_html

@app.route('/download_csv')
def download_csv():
    csv_path = 'top_tracks.csv'
    return send_file(csv_path, as_attachment=True)


@app.route('/refresh_token')
def refresh_token():
    token_info = session.get('token_info', {})
    if sp_oauth.is_token_expired(token_info):
        token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
        session['token_info'] = token_info
    return redirect(url_for('get_user_data'))

if __name__ == '__main__':
    app.run(debug=True)






    # <p>Average Track Duration: {avg_duration_minutes:.2f} minutes</p>
    # <p>Average Popularity: {avg_popularity:.2f}</p>