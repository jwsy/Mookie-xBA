import joblib
import streamlit as st
from PIL import Image

# Load trained model
model = joblib.load("xba_rf_model.joblib")

# App title
st.title("Expected Batting Average (xBA) Predictor")
st.write(
    "Enter the Exit Velocity (mph) and Launch Angle (°) to predict the xBA of a batted ball."
)

# Sidebar inputs
ev = st.slider("Exit Velocity (mph)",
               min_value=30.0,
               max_value=120.0,
               value=95.0,
               step=0.1)
la = st.slider("Launch Angle (°)",
               min_value=-90.0,
               max_value=90.0,
               value=20.0,
               step=0.1)

# Prediction
input_data = [[ev, la]]
predicted_xba = model.predict(input_data)[0]

st.metric(label="Predicted xBA", value=f"{predicted_xba:.3f}")

# Display image
st.subheader("xBA by Exit Velocity and Launch Angle")
image = Image.open("mookie-xba.png")
st.image(image,
         caption="Statcast Heatmap of xBA by EV and LA",
         use_column_width=True)
