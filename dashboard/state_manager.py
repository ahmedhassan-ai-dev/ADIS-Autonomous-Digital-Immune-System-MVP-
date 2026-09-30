import streamlit as st
from datetime import datetime

def initialize_session_state():
    defaults = {
        "adis_events": [],
        "experiment_count": 0,
        "last_experiment": None,
        "last_updated": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def record_event(event):
    initialize_session_state()
    st.session_state.adis_events.append(event)
    st.session_state.experiment_count += 1
    st.session_state.last_experiment = event
    st.session_state.last_updated = datetime.utcnow().isoformat() + "Z"
