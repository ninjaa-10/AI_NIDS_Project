import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
st.set_page_config(page_title="AI-Based traffic detection", layout="wide")
st.title("AI-Based Network Intrusion Detection System")
@st.cache_data
def load_data():
    df = pd.read_csv("data/cic_ids.csv")
    df.columns = df.columns.str.strip()
    df["label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)
    selected_features = ["Flow Duration","Total Fwd Packets","Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets"
    ]
    df = df[selected_features + ["label"]]
    df = df.fillna(0)
    return df
df = load_data()
st.sidebar.header("⚙ Controls")
if st.sidebar.button("Train Model Now"):
    X = df.drop("label", axis=1)
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success("Model Trained Successfully!")
    st.info(f"Model Accuracy: {acc * 100:.2f}%")
    st.session_state["model"] = model
st.header("Finding Live Network Traffic Simulation")
flow_duration = st.number_input("Flow Duration", 0)
fwd_packets = st.number_input("Total Forward Packets", 0)
bwd_packets = st.number_input("Total Backward Packets", 0)
fwd_bytes = st.number_input("Total Length of Forward Packets", 0)
bwd_bytes = st.number_input("Total Length of Backward Packets", 0)
if st.button("Detect Intrusion"):
    if "model" not in st.session_state:
        st.warning("Please train the model first!")
    else:
        input_data = np.array([[flow_duration, fwd_packets, bwd_packets, fwd_bytes, bwd_bytes]])
        prediction = st.session_state["model"].predict(input_data)

        if prediction[0] == 1:
            st.error("Intrusion Detected!")
        else:
            st.success("Normal Network Traffic")
