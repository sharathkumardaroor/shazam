# audio_fingerprint.py

import os
import sqlite3
import numpy as np
import librosa
from scipy.ndimage import maximum_filter
from pydub import AudioSegment
from tqdm import tqdm

DB_FILE = 'fingerprints.db'
SONGS_DIR = 'songs'
SR = 22050  # Sampling rate for audio

def init_db(db_file):
    """Initialize the database and create tables (and an index) if they don't exist."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS songs (
            song_id INTEGER PRIMARY KEY AUTOINCREMENT,
            song_name TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fingerprints (
            hash INTEGER,
            song_id INTEGER,
            offset INTEGER,
            FOREIGN KEY (song_id) REFERENCES songs(song_id)
        )
    ''')
    # Create an index on the hash column to speed up lookups later.
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_hash ON fingerprints(hash)')
    conn.commit()
    return conn

def store_song(conn, song_name):
    """Insert a song record and return its song_id."""
    cursor = conn.cursor()
    cursor.execute('INSERT INTO songs (song_name) VALUES (?)', (song_name,))
    conn.commit()
    return cursor.lastrowid

def store_fingerprints(conn, song_id, fingerprints):
    """Store a list of fingerprints for a given song_id."""
    cursor = conn.cursor()
    cursor.executemany(
        'INSERT INTO fingerprints (hash, song_id, offset) VALUES (?, ?, ?)',
        [(fp, song_id, offset) for fp, offset in fingerprints]
    )
    conn.commit()

def load_audio(file_path, sr=SR):
    """
    Load audio from a file.
    For MP3 files, use PyDub to bypass potential issues with audioread/aifc.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.mp3':
        audio = AudioSegment.from_file(file_path, format="mp3")
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels))
            samples = samples.mean(axis=1)
        y = samples / (2**15)  # Normalize 16-bit PCM data to [-1, 1]
        if audio.frame_rate != sr:
            y = librosa.resample(y, orig_sr=audio.frame_rate, target_sr=sr)
        return y, sr
    else:
        y, sr = librosa.load(file_path, sr=sr, mono=True)
        return y, sr

def compute_spectrogram(y, n_fft=4096, hop_length=512):
    """Compute the magnitude spectrogram using STFT."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    return S

def detect_peaks(S, neighborhood_size=20, amplitude_min=10):
    """
    Detect local peaks in the spectrogram.
    Returns an array of [frequency_bin, time_bin] indices.
    """
    local_max = maximum_filter(S, size=neighborhood_size) == S
    detected_peaks = np.argwhere(local_max & (S >= amplitude_min))
    return detected_peaks

def create_fingerprints(peaks, min_time_delta=1, max_time_delta=50):
    """
    Create fingerprints by pairing each peak with every subsequent peak within a target zone.
    Each fingerprint is a hash of (freq1, freq2, time_delta) along with the time offset.
    """
    print("Creating fingerprints from peaks...")
    fingerprints = []
    peaks = sorted(peaks, key=lambda x: x[1])  # Sort by time (second element)
    num_peaks = len(peaks)
    for i in range(num_peaks):
        freq1, t1 = peaks[i]
        for j in range(i + 1, num_peaks):
            freq2, t2 = peaks[j]
            time_delta = t2 - t1
            if time_delta < min_time_delta:
                continue
            if time_delta > max_time_delta:
                break
            fp = hash((int(freq1), int(freq2), int(time_delta))) & 0xffffffff
            fingerprints.append((fp, t1))
        if i % 50 == 0 and i > 0:
            print(f"Processed {i}/{num_peaks} peaks for fingerprint creation...")
    print(f"Created {len(fingerprints)} fingerprints from {num_peaks} peaks.")
    return fingerprints

def process_song(file_path):
    """Process a single audio file and return its fingerprints."""
    print(f"\nProcessing {file_path} ...")
    y, sr = load_audio(file_path)
    print("Audio loaded.")
    S = compute_spectrogram(y)
    print("Spectrogram computed.")
    peaks = detect_peaks(S)
    print(f"Detected {len(peaks)} peaks.")
    fingerprints = create_fingerprints(peaks)
    return fingerprints

def main():
    conn = init_db(DB_FILE)
    files = [f for f in os.listdir(SONGS_DIR) if f.lower().endswith(('.mp3', '.wav', '.flac', '.ogg'))]
    print(f"Found {len(files)} audio files in '{SONGS_DIR}' directory.")
    
    for filename in tqdm(files, desc="Processing songs", unit="song"):
        file_path = os.path.join(SONGS_DIR, filename)
        fingerprints = process_song(file_path)
        song_id = store_song(conn, filename)
        store_fingerprints(conn, song_id, fingerprints)
        print(f"Stored {len(fingerprints)} fingerprints for '{filename}' (song_id: {song_id}).")
    
    conn.close()
    print("All songs processed and fingerprints stored.")

if __name__ == "__main__":
    main()
