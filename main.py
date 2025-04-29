import os
from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify
from flask_socketio import SocketIO, emit
from video_feed import start_video
from spotify_functions import validate_spotify_credentials, now_playing as get_now_playing, connect_spotify

app_flask = Flask(__name__)
app_flask.secret_key = os.urandom(24)  # For flash messages
socketio = SocketIO(app_flask)

# Path to .env file
ENV_FILE = os.path.join(os.path.dirname(__file__), '.env')

# Global variable to store Spotify credentials
SPOTIFY_CREDENTIALS = {
    'client_id': '',
    'client_secret': '',
    'redirect_uri': ''
}

@app_flask.route('/save_credentials', methods=['POST'])
def save_credentials():
    client_id = request.form.get('client_id', '').strip()
    client_secret = request.form.get('client_secret', '').strip()
    redirect_uri = request.form.get('redirect_uri', '').strip()

    try:
        # # Ensure .env file exists
        # if not os.path.exists(ENV_FILE):
        #     open(ENV_FILE, 'a').close()

        # # Save credentials to .env file
        # set_key(ENV_FILE, 'SPOTIPY_CLIENT_ID', client_id)
        # set_key(ENV_FILE, 'SPOTIPY_CLIENT_SECRET', client_secret)
        # set_key(ENV_FILE, 'SPOTIPY_REDIRECT_URI', redirect_uri)

        # Update environment variables for current session
        os.environ['SPOTIPY_CLIENT_ID'] = client_id
        os.environ['SPOTIPY_CLIENT_SECRET'] = client_secret
        os.environ['SPOTIPY_REDIRECT_URI'] = redirect_uri

        flash('Spotify credentials saved successfully!', 'success')
        return redirect(url_for('home'))
    except Exception as e:
        # logger.error(f"Error saving credentials: {e}")
        flash('Failed to save credentials. Please try again.', 'error')
        return redirect(url_for('home'))

@app_flask.route('/validate_credentials', methods=['POST'])
def validate_credentials():
    try:
        credentials = request.get_json()
        
        # Validate input
        if not all(key in credentials for key in ['client_id', 'client_secret', 'redirect_uri']):
            return jsonify({'valid': False, 'error': 'Missing credentials'}), 400
        
        # Use the validation function from spotify_functions
        is_valid, message = validate_spotify_credentials(
            credentials['client_id'], 
            credentials['client_secret'], 
            credentials['redirect_uri']
        )
        
        return jsonify({
            'valid': is_valid, 
            'message': message
        }), 200
    except Exception as e:
        return jsonify({
            'valid': False, 
            'error': str(e)
        }), 500

@app_flask.route('/set_credentials', methods=['POST'])
def set_credentials():
    global SPOTIFY_CREDENTIALS
    try:
        credentials = request.get_json()
        
        # Validate input
        if not all(key in credentials for key in ['client_id', 'client_secret', 'redirect_uri']):
            return jsonify({'valid': False, 'error': 'Missing credentials'}), 400

        is_valid, message = validate_spotify_credentials(
            credentials['client_id'], 
            credentials['client_secret'], 
            credentials['redirect_uri']
        )
        
        if not is_valid:
            return jsonify({
                'valid': False, 
                'error': message
            }), 400
        
        # Set credentials for app.py
        SPOTIFY_CREDENTIALS['client_id'] = credentials['client_id']
        SPOTIFY_CREDENTIALS['client_secret'] = credentials['client_secret']
        SPOTIFY_CREDENTIALS['redirect_uri'] = credentials['redirect_uri']
        
        return jsonify({
            'valid': True, 
            'message': 'Credentials set successfully'
        }), 200
    except Exception as e:
        return jsonify({
            'valid': False, 
            'error': str(e)
        }), 500

@app_flask.route('/')
def home():
    return render_template('index.html')

@app_flask.route('/video_feed')
def video_feed():
    try:
        return Response(start_video(SPOTIFY_CREDENTIALS), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        return str(e), 500

@app_flask.route('/now_playing', methods=['GET'])
def now_playing():
    try:
        sp = connect_spotify(
            SPOTIFY_CREDENTIALS['client_id'], 
            SPOTIFY_CREDENTIALS['client_secret'], 
            SPOTIFY_CREDENTIALS['redirect_uri'], 
            'user-read-playback-state'
        )
        is_playing, track_name, artist_name, album_art, progress_ms, duration_ms = get_now_playing(sp)
        return jsonify({
            'is_playing': is_playing,
            'track_name': track_name,
            'artist_name': artist_name,
            'album_art': album_art,
            'progress_ms': progress_ms,
            'duration_ms': duration_ms
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('gesture')
def handle_gesture(data):
    print(f'Gesture received: {data}')
    
    # Map gestures to Spotify actions
    try:
        hand = data.get('hand', '')
        gesture = data.get('gesture', '')
        
        if hand == 'left':
            if gesture == 'point':
                spotify_functions.play_pause()
            elif gesture == 'open_palm':
                spotify_functions.volume_up()
            elif gesture == 'fist':
                spotify_functions.volume_down()
        
        elif hand == 'right':
            if gesture == 'index_up_left':
                spotify_functions.next_track()
            elif gesture == 'index_up_right':
                spotify_functions.previous_track()
        
    except Exception as e:
        print(f"Error processing gesture: {e}")

if __name__ == '__main__':
    app_flask.run(debug=True)