import os
import sqlite3
import numpy as np
import librosa
import sounddevice as sd
from scipy.ndimage import maximum_filter
from tqdm import tqdm

DB_FILE = 'fingerprints.db'
DURATION = 10  # Duration in seconds for the snippet
SR = 22050     # Sampling rate for recording

# Adjustable amplitude threshold for query snippet.
QUERY_AMPLITUDE_MIN = 5  

def init_db(db_file):
    """Connect to the existing SQLite database."""
    conn = sqlite3.connect(db_file)
    return conn

def record_audio(duration=DURATION, sr=SR):
    """Record audio from the default microphone."""
    print(f"Recording a {duration}-second audio snippet from the microphone...")
    audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    return audio_data.flatten(), sr

def compute_spectrogram(y, n_fft=4096, hop_length=512):
    """Compute the magnitude spectrogram using STFT."""
    print("Computing spectrogram...")
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    print("Spectrogram computed.")
    return S

def detect_peaks(S, neighborhood_size=20, amplitude_min=QUERY_AMPLITUDE_MIN):
    """
    Detect local peaks in the spectrogram.
    Returns an array of [frequency_bin, time_bin] indices.
    """
    print("Detecting peaks in spectrogram...")
    local_max = maximum_filter(S, size=neighborhood_size) == S
    detected_peaks = np.argwhere(local_max & (S >= amplitude_min))
    print(f"Detected {len(detected_peaks)} peaks.")
    return detected_peaks

def create_fingerprints(peaks, min_time_delta=1, max_time_delta=50):
    """
    Create fingerprints by pairing each peak with every subsequent peak within a target zone.
    Each fingerprint is a hash of (freq1, freq2, time_delta) along with the time offset.
    """
    print("Creating fingerprints from peaks...")
    fingerprints = []
    peaks = sorted(peaks, key=lambda x: x[1])
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

def process_query(duration=DURATION):
    """Record audio snippet from microphone and return its fingerprints."""
    y, sr = record_audio(duration=duration, sr=SR)
    S = compute_spectrogram(y)
    peaks = detect_peaks(S)
    fingerprints = create_fingerprints(peaks)
    print(f"Extracted {len(peaks)} peaks and {len(fingerprints)} fingerprints from query.")
    return fingerprints

def load_all_fingerprints(conn):
    """
    Load all stored fingerprints into a dictionary.
    Returns a dictionary mapping hash -> list of (song_id, offset).
    """
    print("Loading all stored fingerprints into memory...")
    cursor = conn.cursor()
    cursor.execute("SELECT hash, song_id, offset FROM fingerprints")
    rows = cursor.fetchall()
    fp_dict = {}
    for h, song_id, offset in rows:
        if isinstance(offset, bytes):
            offset = int.from_bytes(offset, byteorder='little')
        else:
            offset = int(offset)
        fp_dict.setdefault(h, []).append((song_id, offset))
    print(f"Loaded fingerprints for {len(fp_dict)} unique hashes.")
    return fp_dict

def match_fingerprints(conn, query_fingerprints):
    """
    Match query fingerprints against stored fingerprints loaded in memory.
    Returns a dictionary with keys (song_id, offset_difference) and vote counts.
    """
    stored_fp = load_all_fingerprints(conn)
    matches = {}
    total_matches = 0
    for fp, q_offset in tqdm(query_fingerprints, desc="Matching fingerprints", unit="fp"):
        if fp in stored_fp:
            for song_id, db_offset in stored_fp[fp]:
                diff = int(db_offset) - int(q_offset)
                key = (song_id, diff)
                matches[key] = matches.get(key, 0) + 1
                total_matches += 1
    print(f"Total matches found: {total_matches}")
    return matches

def find_best_match(matches):
    """Aggregate votes per song and return the song_id with the most votes."""
    song_votes = {}
    for (song_id, diff), count in matches.items():
        song_votes[song_id] = song_votes.get(song_id, 0) + count
    if not song_votes:
        return None, 0
    best_song = max(song_votes, key=song_votes.get)
    return best_song, song_votes[best_song]

def get_song_name(conn, song_id):
    """Retrieve the song name from the songs table."""
    cursor = conn.cursor()
    cursor.execute("SELECT song_name FROM songs WHERE song_id = ?", (song_id,))
    row = cursor.fetchone()
    return row[0] if row else None

def main():
    print("Recording query audio snippet...")
    query_fingerprints = process_query()
    conn = init_db(DB_FILE)
    matches = match_fingerprints(conn, query_fingerprints)
    best_song, votes = find_best_match(matches)
    if best_song:
        song_name = get_song_name(conn, best_song)
        print(f"Best match: {song_name} (song_id: {best_song}) with {votes} votes")
    else:
        print("No match found.")
    conn.close()

if __name__ == "__main__":
    main()
