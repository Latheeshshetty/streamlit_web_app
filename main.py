# importing the libraries
import pickle
import streamlit as st
import numpy as np
import librosa
import soundfile

# open the pickle file
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

# defining the function for prediction


def predction(filename):
    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs = np.mean(mfccs_features.T, axis=0)
    pred = classifier.predict(mfccs.reshape(1, -1))
    return pred


# app title
st.title('Heart disease prediction')

# uploding audio
upload_audio = st.sidebar.file_uploader(
    label="uplod heart sound", type=".wav")

# displying the audio file
if upload_audio is not None:
    st.write(upload_audio)
st.audio(upload_audio, format='audio/wav')

# prediction

if st.button('predict'):
    result = predction(upload_audio)[0]
    if result == 'normal':
        st.success('Your heart sound is Normal')
    else:
        st.error('Your heart sound in Abnormal')
