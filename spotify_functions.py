from email import message
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os

from dotenv import load_dotenv

load_dotenv()

# Set up Spotify with Spotipy
SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')
SPOTIPY_REDIRECT_URI = os.getenv('SPOTIPY_REDIRECT_URI')

SCOPE = 'user-modify-playback-state user-read-playback-state'

def validate_spotify_credentials(client_id, client_secret, redirect_uri):
    """
    Validate Spotify credentials without establishing a full connection.
    
    Args:
        client_id (str): Spotify Client ID
        client_secret (str): Spotify Client Secret
        redirect_uri (str): Spotify Redirect URI
    
    Returns:
        tuple: (is_valid, error_message)
    """
    # Basic input validation
    if not all([client_id, client_secret, redirect_uri]):
        return False, "All credentials must be provided"
    
    # Validate format of credentials
    if len(client_id) < 10 or len(client_secret) < 10:
        return False, "Invalid Client ID or Client Secret"
    
    # Validate redirect URI
    if not redirect_uri.startswith(('http://', 'https://')):
        return False, "Invalid Redirect URI format"
    
    try:
        # Attempt to create an auth manager (this will validate basic credentials)
        auth_manager = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=SCOPE
        )
        
        # Attempt to get the authorization URL (this checks basic credential validity)
        auth_manager.get_authorize_url()
        
        return True, "Credentials are valid"
    except Exception as e:
        return False, f"Credential validation failed: {str(e)}"

def connect_spotify(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI,
                    SCOPE):
    sp = None
    try:
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=SPOTIPY_CLIENT_ID,
            client_secret=SPOTIPY_CLIENT_SECRET,
            redirect_uri=SPOTIPY_REDIRECT_URI,
            scope=SCOPE
        ))
        
        # Verify Spotify connection by attempting to get current playback
        current_playback = sp.current_playback()
        
        print('Spotify Connected')
        if current_playback:
            print("Currently playing:", current_playback['item']['name'] if 'item' in current_playback else "No track")
        else:
            print("No active playback")
    except Exception as e:
        print(f"Spotify Authentication Error: {e}")
        print(f"Client ID: {SPOTIPY_CLIENT_ID}")
        print(f"Redirect URI: {SPOTIPY_REDIRECT_URI}")
    
    return sp

def now_playing(sp):
    if sp is None:
        raise ValueError("Spotify connection not established")
    
    current_playback = sp.current_playback()
    if current_playback is None or 'item' not in current_playback:
        raise ValueError("No active playback")
    
    song_dict = current_playback['item']
    is_playing = current_playback['is_playing']
    album_art = song_dict['album']['images'][0]['url']
    track_name = str(song_dict['name'])
    artists = song_dict['artists']
    artist_names = list()
    for artist in artists:
        artist_names.append(artist['name'])
    artist_names = ', '.join(artist_names)
    progress_ms = current_playback['progress_ms']
    duration_ms = current_playback['item']['duration_ms']

    return is_playing, track_name, artist_names, album_art, progress_ms, duration_ms

def change_playback(sp):
    if sp is None:
        raise ValueError("Spotify connection not established")
    
    current_playback = sp.current_playback()
    if current_playback is None:
        raise ValueError("No active playback")

    if current_playback['is_playing']:
        sp.pause_playback()
        return 'paused'
    else:
        sp.start_playback()
        return 'playing'

def change_volume(sp, direction):
    if sp is None:
        raise ValueError("Spotify connection not established")
    current_playback = sp.current_playback()
    if current_playback is None:
        raise ValueError("No active playback")

    if current_playback['is_playing']:
        current_volume = current_playback['device']['volume_percent']
        new_volume = current_volume + 15 if (direction) else current_volume - 15
        
        # Don't allow values < 0 or > 100 (system complains)
        if new_volume > 100:
            sp.volume(100)
            return "Volume already at 100%"
        elif new_volume < 0:
            sp.volume(0)
            return "Volume already at 0%"
        
        sp.volume(new_volume)
        return "Volume Increased" if direction else "Volume Decreased"
    else:
        return "Currently not playing music..."

def main():
    credentials = {
        'client_id': SPOTIPY_CLIENT_ID,
        'client_secret': SPOTIPY_CLIENT_SECRET,
        'redirect_uri': SPOTIPY_REDIRECT_URI
    }
    
    isValid, message = validate_spotify_credentials(credentials['client_id'], credentials['client_secret'], credentials['redirect_uri'])
    # print(isValid, message)
    
    sp = connect_spotify(credentials['client_id'], credentials['client_secret'], credentials['redirect_uri'], SCOPE)
    is_playing, track_name, artist_names, album_art, progress_ms, duration_ms = now_playing(sp)
    print(is_playing, track_name, artist_names, album_art, progress_ms, duration_ms)

if __name__ == '__main__':
    main()
