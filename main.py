# importing the libraries
import pickle
import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile
import matplotlib.pyplot as plt

# open the pickle file
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

# defining the function for prediction


def predction(filename):
    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs = np.mean(mfccs_features.T, axis=0)
    pred = classifier.predict(mfccs.reshape(1, -1))
    fig = plt.figure(figsize=(10, 2))
    fig.set_facecolor('#d1d1e0')
    plt.title("Wave-form")
    librosa.display.waveshow(audio, sr=44100)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.spines["right"].set_visible(False)
    plt.gca().axes.spines["left"].set_visible(False)
    plt.gca().axes.spines["top"].set_visible(False)
    plt.gca().axes.spines["bottom"].set_visible(False)
    plt.gca().axes.set_facecolor('#d1d1e0')
    st.write(fig)
    return pred


# app title
st.title('HEART DISEASE PREDICTION')

# uploding audio
upload_audio = st.file_uploader(
    label="uplod heart sound", type=".wav")

# displying the audio file
if upload_audio is not None:
    st.write(upload_audio)
st.audio(upload_audio, format='audio/wav')

# wave form


# prediction

if st.button('predict'):
    result = predction(upload_audio)[0]
    if result == 'normal':
        st.success('Your heart sound is Normal')
    else:
        st.error('Your heart sound in Abnormal')
