import parselmouth
import os
import joblib
import pandas as pd
import json
import time

import sounddevice as sd
from scipy.io.wavfile import write

audio_directory = "/home/cohen/Desktop/parselmouth/audio"

def load_audio_files(directory):
    audio_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            audio_files.append(os.path.join(directory, filename))
    return audio_files

def extract_features(audio_file):
    sound = parselmouth.Sound(audio_file)
    pitch = sound.to_pitch()
    intensity = sound.to_intensity()

    pitch_mean = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")
    pitch_min = parselmouth.praat.call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    pitch_max = parselmouth.praat.call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    intensity_mean = parselmouth.praat.call(intensity, "Get mean", 0, 0, "energy")

    point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 500)
    jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    #hnr = parselmouth.praat.call(sound, "Get harmonicity (cc)", 0.01, 75, 0.1, 1.0)

    return {
        "pitch_mean": pitch_mean,
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "intensity_mean": intensity_mean,
        "jitter": jitter,
        "shimmer": shimmer,
        #"hnr": hnr
    }

def process_audio_files(directory):
    features_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            audio_file = os.path.join(directory, filename)
            features = extract_features(audio_file)
            features["file"] = filename
            features_list.append(features)

    return features_list

def main():
    # Configure recording parameters
    SAMPLE_RATE = 44100  # CD quality audio (Hz)
    DURATION = 5         # Duration in seconds
    FILENAME = f"{audio_directory}/audio{time.time()}.wav"
    print("Recording started... Speak now!")

    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=2)

    # Block execution until the recording is complete
    sd.wait() 

    print("Recording finished. Saving file...")

    # Save the recording as a WAV file
    write(FILENAME, SAMPLE_RATE, recording)
    print(f"File saved successfully as '{FILENAME}'")

    features_list = process_audio_files(audio_directory)
    for item in features_list:
        print(item["file"])
        for key, value in item.items():
            if key != "file":
                print(key, value)

    with open("/home/cohen/Desktop/parselmouth/outputs/output.json", "w") as json_file:
        json.dump(features_list, json_file, indent = 4)

if __name__ == "__main__":
    main()