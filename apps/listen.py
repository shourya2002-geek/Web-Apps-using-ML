import os
import streamlit as st
import glob
from .helper import *
from .transcribe import *



def app():
    st.header("1. Record your own voice")

    filename = st.text_input("Choose a filename: ")

    if st.button(f"Click to Record"):
        if filename == "":
            st.warning("Choose a filename.")
        else:
            record_state = st.text("Recording...")
            duration = 5  # seconds
            fs = 48000
            myrecording = record(duration, fs)
            record_state.text(f"Saving sample as {filename}.mp3")

            path_myrecording = f"./{filename}.mp3"

            save_record(path_myrecording, myrecording, fs)
            record_state.text(f"Done! Saved sample as {filename}.mp3")

            st.audio(read_audio(path_myrecording))
            
            if st.button("Transcribe Data"):
                token, t_id = upload_file(path_myrecording)
                result = {}
                #polling
                sleep_duration = 1
                percent_complete = 0
                progress_bar = st.progress(percent_complete)
                st.text("Currently in queue")
                while result.get("status") != "processing":
                    percent_complete += sleep_duration
                    time.sleep(sleep_duration)
                    progress_bar.progress(percent_complete/10)
                    result = get_text(token,t_id)

                sleep_duration = 0.01

                for percent in range(percent_complete,101):
                    time.sleep(sleep_duration)
                    progress_bar.progress(percent)

                with st.spinner("Processing....."):
                    while result.get("status") != 'completed':
                        result = get_text(token,t_id)

                
                st.header("Transcribed Text")
                st.subheader(result['text'])

                st.session_state = result['text']



        

    "## 2. Choose an audio record"

    audio_folder = "samples"
    filenames = glob.glob(os.path.join(audio_folder, "*.mp3"))
    selected_filename = st.selectbox("Select a file", filenames)
