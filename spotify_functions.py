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

def connect_spotify(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI,
                    SCOPE):
    try:
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=SPOTIPY_CLIENT_ID,
            client_secret=SPOTIPY_CLIENT_SECRET,
            redirect_uri=SPOTIPY_REDIRECT_URI,
            scope=SCOPE
        ))
        print('Spotify Connected')
        # print(len(sp.current_playback()))
        # print(sp.current_playback().keys())
    except Exception as e:
        print(f"Spotify Authentication Error: {e}")

    return sp

def current_song(sp):
    song_dict = sp.current_playback()['item']
    song = song_dict['name']
    artists = song_dict['artists']
    artist_names = list()
    for artist in artists:
        artist_names.append(artist['name'])
    artist_names = ', '.join(artist_names)
    return song, artist_names

def change_playback(sp):
    current_playback = sp.current_playback()

    if current_playback['is_playing']:
        sp.pause_playback()
        return 'paused'
    else:
        sp.start_playback()
        return 'playing'

def change_volume(sp,direction):
    current_playback = sp.current_playback()
    if current_playback['is_playing']:
        current_volume = current_playback['device']['volume_percent']
        # current_volume = device['volume_percent']
        new_volume =  current_volume + 15 if (direction) else  current_volume - 15
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

    sp = connect_spotify(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI, SCOPE)

    song, artist_names = current_song(sp)

    print(f'Current Song: {song} \nBy: {artist_names}')

    # song_status = change_playback(sp)

    # print(song_status)
    device = change_volume(sp,1)
    # change_volume(sp,1)

    print(device)

if __name__ == '__main__':
    main()
