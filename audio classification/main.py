'''
import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
import seaborn as sns

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')

sns.set(style='darkgrid', palette='magma')
plot_settings = {
    'font.family': 'calibri',
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'figure.dpi': 140,
    'axes.titlepad': 15,
    'axes.labelpad': 15,
    'figure.titlesize': 24,
    'figure.titleweight': 'bold',
}
# Use the dictionary variable to update the settings using matplotlib
plt.rcParams.update(plot_settings)

tf.config.list_physical_devices('GPU')


import os
import pandas as pd

# Put the cremad directory in a list
cremad = os.listdir(r'C:\waudio\\Yarchive\\Crema')

# Make a list for emotion labels and a list for path to audio files
emotions = []
paths = []

# Loop through all the files and extract the emotion label and path
for file in cremad:
    # Extract the emotion label from the file name
    emotion = file.split('_')[2]
    if emotion == 'SAD':
        emotion = 'sadness'
    elif emotion == 'ANG':
        emotion = 'anger'
    elif emotion == 'DIS':
        emotion = 'disgust'
    elif emotion == 'FEA':
        emotion = 'fear'
    elif emotion == 'HAP':
        emotion = 'happiness'
    elif emotion == 'NEU':
        emotion = 'neutral'
    elif emotion == 'SUR':
        emotion = 'surprise'
    else:
        emotion = 'Unknown'
    
    # Construct the full path to the audio file
    path = os.path.join(r'C:\waudio\\Yarchive\\Crema\AudioWAV', file)
    
    # Append the emotion and path to their lists
    emotions.append(emotion)
    paths.append(path)

# Create a dataframe from the lists
cremad_df = pd.DataFrame(emotions, columns=['Emotion'])
cremad_df['Path'] = paths

# Inspect the dataframe
cremad_df.head()





# Put the ravdess directory in a list
ravdess = os.listdir(r"C:\waudio\\Yarchive\\Ravdess\\audio_speech_actors_01-24")

# Make a list for emotion labels and a list for path to audio files
emotions = []
paths = []

# Loop through all the actor directories in audio_speech_actors_01-24
for dir in ravdess:
    # Construct the full path to the directory
    actor_dir = os.path.join(r"C:\waudio\\Yarchive\\Ravdess\\audio_speech_actors_01-24", dir)
    
    # Check if it's a directory
    if not os.path.isdir(actor_dir):
        continue
    
    # Loop through all the files in each actor directory
    for file in os.listdir(actor_dir):
        # Extract the emotion label from the file name
        emotion = file.split('-')[2]
        if emotion == '01':
            emotion = 'neutral'
        elif emotion == '02':
            emotion = 'calm'
        elif emotion == '03':
            emotion = 'happiness'
        elif emotion == '04':
            emotion = 'sadness'
        elif emotion == '05':
            emotion = 'anger'
        elif emotion == '06':
            emotion = 'fear'
        elif emotion == '07':
            emotion = 'disgust'
        elif emotion == '08':
            emotion = 'surprise'
        else:
            emotion = 'Unknown'
        
        # Construct the full path to the audio file
        path = os.path.join(actor_dir, file)
        
        # Append the emotion and path to their lists
        emotions.append(emotion)
        paths.append(path)

# Create a dataframe from the lists
ravdess_df = pd.DataFrame(emotions, columns=['Emotion'])
ravdess_df['Path'] = paths

# Inspect the dataframe
ravdess_df.head()





# Put the tess directory in a list
tess = os.listdir(r'C:\waudio\\Yarchive\\Tess')
# Make a list for emotion labels and a list for path to audio files
emotions = []
paths = []

# Loop through all the audio file directories
for dir in tess:
    # Construct the full path to the subdirectory
    sub_dir = os.path.join(r'C:\waudio\\Yarchive\\Tess', dir)
    # Ensure it's a directory (skip files, if any)
    if not os.path.isdir(sub_dir):
        continue
    
    # Loop through all the files in each directory
    for file in os.listdir(sub_dir):
        # Extract the emotion label from the file name
        emotion = file.split('.')[0]
        emotion = emotion.split('_')[2]
        if emotion == 'ps':
            emotion = 'surprise'
        elif emotion == 'sad':
            emotion = 'sadness'
        elif emotion == 'disgust':
            emotion = 'disgust'
        elif emotion == 'angry':
            emotion = 'anger'
        elif emotion == 'happy':
            emotion = 'happiness'
        elif emotion == 'neutral':
            emotion = 'neutral'
        elif emotion == 'fear':
            emotion = 'fear'
        else:
            emotion = 'Unknown'
        
        # Construct the full path to the audio file
        path = os.path.join(sub_dir, file)
        
        # Append the emotion and path to their lists
        emotions.append(emotion)
        paths.append(path)

# Create a dataframe from the lists
tess_df = pd.DataFrame(emotions, columns=['Emotion'])
tess_df['Path'] = paths

# Inspect the dataframe
tess_df.head()



# Put the savee directory in a list
savee = os.listdir(r'C:\waudio\\Yarchive\Savee')  # Use raw string to avoid path issues

# Make a list for emotion labels and a list for path to audio files
emotions = []
paths = []

# Loop through all the files in the directory
for file in savee:
    # Separate the wav file name from the emotion label
    emotion = file.split('.')[0]
    # Extract the emotion label from the file name
    emotion = emotion.split('_')[1]
    # Exclude the numbers from the emotion label
    emotion = emotion[:-2]
    if emotion == 'a':
        emotion = 'anger'
    elif emotion == 'd':
        emotion = 'disgust'
    elif emotion == 'f':
        emotion = 'fear'
    elif emotion == 'h':
        emotion = 'happiness'
    elif emotion == 'n':
        emotion = 'neutral'
    elif emotion == 'sa':
        emotion = 'sadness'
    elif emotion == 'su':
        emotion = 'surprise'
    else:
        emotion = 'Unknown'
    
    # Correctly join the directory and file path
    path = os.path.join(r'C:\waudio\\Yarchive\Savee', file)
    
    # Append the emotion and path to their lists
    emotions.append(emotion)
    paths.append(path)

# Create a dataframe from the lists
savee_df = pd.DataFrame(emotions, columns=['Emotion'])
savee_df['Path'] = paths

# Inspect the dataframe
print(savee_df.head())


# Plot the value counts for each emotion in each dataset
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
sns.countplot(cremad_df, x = cremad_df['Emotion'], palette='magma')
plt.title('CREMA-D')
plt.xlabel('')
plt.subplot(2, 2, 2)
sns.countplot(ravdess_df, x = ravdess_df['Emotion'], palette='magma')
plt.title('RAVDESS')
plt.xlabel('')
plt.subplot(2, 2, 3)
sns.countplot(tess_df, x = tess_df['Emotion'], palette='magma')
plt.title('TESS')
plt.subplot(2, 2, 4)
sns.countplot(savee_df, x = savee_df['Emotion'], palette='magma')
plt.title('SAVEE')
plt.suptitle('Emotion Counts for Each Dataset')
# Adjust the layout so there are no overlapping titles
plt.tight_layout(pad=2)
# Remove the spines
sns.despine()
plt.show()
'''




import os
import tensorflow as tf
import librosa
from sklearn.model_selection import train_test_split
import speech_recognition as sr
import pyaudio

DATASET_PATH = r"C:\waudio\\Yarchive"
MODEL_PATH = "audio_classification_model.h5"
AUTOTUNE = tf.data.AUTOTUNE

# Function to recursively find all .wav files in the dataset path
def find_wav_files_and_labels(directory):
    file_paths = []
    labels = []
    class_map = {}
    
    for root, _, files in os.walk(directory):
        label = os.path.basename(root)  # Use directory name as label
        if label not in class_map:
            class_map[label] = len(class_map)  # Map unique labels to integers
        
        for file_name in files:
            if file_name.endswith('.wav'):
                file_paths.append(os.path.join(root, file_name))
                labels.append(class_map[label])
    
    print(f"Total files found: {len(file_paths)}")
    print(f"Class mapping: {class_map}")
    return file_paths, labels, len(class_map)

# Function to load and resample WAV files using librosa
def load_wav_16k_mono(filename):
    wav, _ = librosa.load(filename.numpy().decode("utf-8"), sr=16000, mono=True)
    return tf.convert_to_tensor(wav, dtype=tf.float32)

# Wrapper to make `load_wav_16k_mono` TensorFlow-compatible
def load_wav_16k_mono_tf(filename):
    return tf.py_function(func=load_wav_16k_mono, inp=[filename], Tout=tf.float32)

# Function to preprocess the audio file
def preprocess(file_path, label):
    wav = load_wav_16k_mono_tf(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    return spectrogram, label

# Find all WAV files and labels
file_paths, labels, num_classes = find_wav_files_and_labels(DATASET_PATH)

# Split dataset into training and validation sets
file_paths_train, file_paths_val, labels_train, labels_val = train_test_split(
    file_paths, labels, test_size=0.3, random_state=40
)

# Convert to TensorFlow Datasets
dataset_train = tf.data.Dataset.from_tensor_slices((file_paths_train, labels_train))
dataset_val = tf.data.Dataset.from_tensor_slices((file_paths_val, labels_val))

# Map preprocess function
dataset_train = dataset_train.map(preprocess, num_parallel_calls=AUTOTUNE)
dataset_val = dataset_val.map(preprocess, num_parallel_calls=AUTOTUNE)

# Batch, cache, and prefetch
dataset_train = dataset_train.batch(8).cache().prefetch(buffer_size=AUTOTUNE)
dataset_val = dataset_val.batch(8).cache().prefetch(buffer_size=AUTOTUNE)

# Build the model
def build_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(None, None, 1)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    return model

# Train the model only if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Training model...")
    model = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)
        ]
    )
    print("Model trained and saved.")
else:
    print("Model already exists. Loading the saved model...")
    model = tf.keras.models.load_model(MODEL_PATH)

emotion_mapping = {
    0: 'Yarchive', 1: 'Crema', 2: 'Ravdess', 3: 'audio_speech_actors_01-24', 
    4: 'Actor_01', 5: 'Actor_02', 6: 'Actor_03', 7: 'Actor_04', 8: 'Actor_05',
    9: 'Actor_06', 10: 'Actor_07', 11: 'Actor_08', 12: 'Actor_09', 13: 'Actor_10',
    14: 'Actor_11', 15: 'Actor_12', 16: 'Actor_13', 17: 'Actor_14', 18: 'Actor_15',
    19: 'Actor_16', 20: 'Actor_17', 21: 'Actor_18', 22: 'Actor_19', 23: 'Actor_20',
    24: 'Actor_21', 25: 'Actor_22', 26: 'Actor_23', 27: 'Actor_24', 28: 'Savee', 
    29: 'Tess', 30: 'OAF_angry', 31: 'OAF_disgust', 32: 'OAF_Fear', 33: 'OAF_happy',
    34: 'OAF_neutral', 35: 'OAF_Pleasant_surprise', 36: 'OAF_Sad', 37: 'YAF_angry',
    38: 'YAF_disgust', 39: 'YAF_fear', 40: 'YAF_happy', 41: 'YAF_neutral', 
    42: 'YAF_pleasant_surprised', 43: 'YAF_sad'
}


# Test phase with speech recognition
def recognize_and_predict(audio_path=None):
    if not audio_path:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Recording audio...")
            audio = recognizer.listen(source)
            with open("temp_audio.wav", "wb") as f:
                f.write(audio.get_wav_data())
            audio_path = "temp_audio.wav"
    
    wav, _ = librosa.load(audio_path, sr=16000, mono=True)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    spectrogram = tf.expand_dims(spectrogram, axis=0)
    prediction = model.predict(spectrogram)
    predicted_label = tf.argmax(prediction, axis=1).numpy()[0]
    
    emotion = emotion_mapping.get(predicted_label, "Unknown")
    print(f"Predicted emotion: {emotion}")

   # print(f"Predicted label: {predicted_label}")

# Test the system
recognize_and_predict()
