import parselmouth
from parselmouth import praat
import math
import os
import joblib
import pandas as pd
import json
import time

import sounddevice as sd
from scipy.io.wavfile import write

from pathlib import Path
from record_chunks import ChunkRecorder

cwd = Path(__file__).resolve().parent
audio_directory = cwd / "audio"
datapoint_interval = 0.2

#Higher pitch results in more deception.

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

    pitch_mean = praat.call(pitch, "Get mean", 0, 0, "Hertz")
    pitch_min = praat.call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    pitch_max = praat.call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    intensity_mean = praat.call(intensity, "Get mean", 0, 0, "energy")

    point_process = praat.call(sound, "To PointProcess (periodic, cc)", 75, 500)
    jitter = praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    data_points = []

    for i in range(0, math.floor(sound.duration/datapoint_interval)):
        pitch_point = praat.call(pitch, "Get mean", i*datapoint_interval, (i+1)*datapoint_interval, "Hertz")
        intensity_point = praat.call(intensity, "Get mean", i*datapoint_interval, (i+1)*datapoint_interval, "energy")
        jitter_point = praat.call(point_process, "Get jitter (local)", i*datapoint_interval, (i+1)*datapoint_interval, 0.0001, 0.02, 1.3)
        shimmer_point = praat.call([sound, point_process], "Get shimmer (local)", i*datapoint_interval, (i+1)*datapoint_interval, 0.0001, 0.02, 1.3, 1.6)

        data_points.append(
            {
                "pitch": pitch_point,
                "intensity": intensity_point,
                "jitter": jitter_point,
                "shimmer": shimmer_point,
            }
        )

    return {
        "pitch_mean": pitch_mean,
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "intensity_mean": intensity_mean,
        "jitter": jitter,
        "shimmer": shimmer,
        "data_points": data_points
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

def analyze_chunks(path):
    data = extract_features(path)

    with open(cwd/"outputs"/"temp"/f"{Path(path).stem}.json", "w") as json_file:
        json.dump(data, json_file, indent = 4)

def record():
    for file_path in (cwd/"outputs"/"temp").iterdir():
            if file_path.is_file():
                file_path.unlink()
    
    rec = ChunkRecorder(
        output=audio_directory/"final",
        chunk_dir=audio_directory/"temp",
        chunk_seconds=datapoint_interval,
        on_chunk = analyze_chunks
    )

    rec.start()

    input("Press enter to stop")

    rec.stop()

def main():
    record()
    
    features_list = process_audio_files(audio_directory/"final")
    for item in features_list:
        print(item["file"])
        for key, value in item.items():
            if key != "file":
                print(key, value)

    with open(cwd/"outputs"/"final"/f"{time.time()}.json", "w") as json_file:
        json.dump(features_list, json_file, indent = 4)

if __name__ == "__main__":
    main()