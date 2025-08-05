# SpotifyRecommender

A recommendation system for Spotify that suggests personalized song recommendations.

## Overview

The Spotify Music Recommendation System is a Python-based application that provides song recommendations based on user preferences. The system uses a hybrid approach that combines content-based filtering and collaborative filtering to generate relevant and diverse suggestions.

## Features

* **Personalized Recommendations:** Get song recommendations tailored to your musical taste based on your user profile and listening history
* **Multiple Recommendation Algorithms:** The system uses a combination of recommendation algorithms to provide the best possible suggestions.
* **Web-based Interface:** The application features a user-friendly web-based interface built with Streamlit.

## Requirements

* Python 3.6+
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Spotipy

## Installation

1.  Clone the repository:
    ```
    git clone [https://github.com/your-username/SpotifyRecommender.git](https://github.com/your-username/SpotifyRecommender.git)
    ```
2.  Install the required packages:
    ```
    pip install -r requirements.txt
    ```
3.  Set up your Spotify API credentials:
    * Create a new application on the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/).
    * Add your `Client ID` and `Client Secret` as environment variables.

## Usage

1.  Run the Streamlit application:
    ```
    streamlit run app.py
    ```
2.  Open the application in your web browser at `http://localhost:8501`.
3.  Enter a song or artist in the search bar to get recommendations.
