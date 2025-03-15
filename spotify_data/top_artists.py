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
scope = 'playlist-read-private user-library-read playlist-read-collaborative user-top-read'


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
    return redirect(url_for('get_user_top_artists'))

@app.route('/login')
def login():
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/callback')
def callback():
    code = request.args['code']
    token_info = sp_oauth.get_access_token(code)
    session['token_info'] = token_info
    return redirect(url_for('get_user_top_artists'))

@app.route('/get_user_top_artists')
def get_user_top_artists():
    token_info = session.get('token_info', {})
    if not token_info:
        return redirect(url_for('login'))

    sp = Spotify(auth=token_info['access_token'])

    # Get the custom number of top artists from the request, default to 1000 if not provided
    n_artists = request.args.get('n', default=1000, type=int)
    
    # Paginate to fetch more than 50 top artists
    artist_data = []
    limit = 50  # Spotify API limit per request
    offset = 0
    genres = []
    total_popularity = 0
    max_popularity = 0
    min_popularity = 101
    total_followers = 0
    mostp=[]
    minp=[]
    count=0
    while len(artist_data) < n_artists:
        top_artists = sp.current_user_top_artists(limit=limit, offset=offset)
        if not top_artists['items']:
            break  # Exit if no more items

        for artist in top_artists['items']:
            count=count+1
            artist_info = {
                'Artist Name': artist['name'],
                'Artist Genres': ', '.join(artist['genres']),
                'Popularity': artist['popularity'],
                'Artist URL': artist['external_urls']['spotify'],
                'Artist ID': artist['id'], 
            }
            # artists=sp.artist(artist['id'])
            # print(artists)

            genres.extend(artist['genres'])
            total_popularity += artist['popularity']
            if(max_popularity<artist['popularity']):
                max_popularity =artist['popularity']
                mostp=artist['name']
            if(min_popularity>artist['popularity']):
                min_popularity =artist['popularity']
                minp=artist['name']
            total_followers += artist['followers']['total']

            # Calculate insights
            artist_data.append(artist_info)
            print(artist_info)
            if len(artist_data) >= n_artists:  # Stop if we've reached the requested number of artists
                break

        offset += limit  # Update offset for pagination

    # Convert to DataFrame
    artist_count =count
    df_artists = pd.DataFrame(artist_data)
    most_common_genres = pd.Series(genres).value_counts().head(5).to_dict()
    avg_popularity = total_popularity / artist_count if artist_count > 0 else 0
    # Save DataFrame to CSV
    csv_path = 'top_artists.csv'
    df_artists.to_csv(csv_path, index=False)

    # HTML Output
    top_artists_html = df_artists.to_html(classes='table table-striped', index=False)

    # Statistics
    total_artists = len(df_artists)



    stats_html = f"""
    <h3>Statistics:</h3>
    <p>Total Top Artists: {total_artists}</p>
    """
    user_insights_html = f"""
    <h3>Your Insights</h3>
    <p><strong>Total Top Artists:</strong> {artist_count}</p>
    <p><strong>Max popularity:</strong> {max_popularity, mostp}</p>
    <p><strong>Min popularity:</strong> {min_popularity, minp}</p>
    <p><strong>Average Artist Popularity:</strong> {avg_popularity:.2f}</p>
    <p><strong>Most Common Genres:</strong> {', '.join([f"{genre} ({count})" for genre, count in most_common_genres.items()])}</p>
    """

    # Add a link to download the CSV
    download_link = f"""
    <p><a href="/download_top_artists_csv" target="_blank">Download Top Artists as CSV</a></p>
    """

    return stats_html + user_insights_html + download_link + top_artists_html

@app.route('/user_features')
def user_features():
    token_info = session.get('token_info', {})
    if not token_info:
        return redirect(url_for('login'))

    sp = Spotify(auth=token_info['access_token'])

    # Fetch user profile information
    # user_profile = sp.current_user()
    # user_name = user_profile['display_name']
    # user_country = user_profile['country']
    # user_image = user_profile['images'][0]['url'] if user_profile['images'] else None
    # user_profile_url = user_profile['external_urls']['spotify']

    # Fetch user's top artists
    top_artists = sp.current_user_top_artists(limit=1000)
    genres = []
    total_popularity = 0
    total_followers = 0
    artist_count = len(top_artists['items'])

    for artist in top_artists['items']:
        genres.extend(artist['genres'])
        total_popularity += artist['popularity']
        total_followers += artist['followers']['total']

    # Calculate insights
    most_common_genres = pd.Series(genres).value_counts().head(5).to_dict()
    avg_popularity = total_popularity / artist_count if artist_count > 0 else 0

    # # Display user information and insights
    # user_info_html = f"""
    # <h3>User Profile</h3>
    # <p><strong>Name:</strong> {user_name}</p>
    # <p><strong>Country:</strong> {user_country}</p>
    # <p><strong>Spotify Profile:</strong> <a href="{user_profile_url}" target="_blank">{user_profile_url}</a></p>
    # """
    # if user_image:
    #     user_info_html += f'<img src="{user_image}" alt="Profile Picture" width="200"><br>'

    user_insights_html = f"""
    <h3>Your Insights</h3>
    <p><strong>Total Top Artists:</strong> {artist_count}</p>
    <p><strong>Average Artist Popularity:</strong> {avg_popularity:.2f}</p>
    <p><strong>Total Followers of Top Artists:</strong> {total_followers}</p>
    <p><strong>Most Common Genres:</strong> {', '.join([f"{genre} ({count})" for genre, count in most_common_genres.items()])}</p>
    """

    return user_insights_html

@app.route('/download_top_artists_csv')
def download_top_artists_csv():
    csv_path = 'top_artists.csv'
    return send_file(csv_path, as_attachment=True)

@app.route('/refresh_token')
def refresh_token():
    token_info = session.get('token_info', {})
    if sp_oauth.is_token_expired(token_info):
        token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
        session['token_info'] = token_info
    return redirect(url_for('get_user_top_artists'))

if __name__ == '__main__':
    app.run(debug=True)



