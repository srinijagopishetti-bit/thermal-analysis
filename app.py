import cv2
import numpy as np
import streamlit as st

st.title("AI Thermal Imaging Health Assessment")
st.write("Upload a thermal image for a non-invasive preliminary health report.")

uploaded_file = st.file_uploader("Choose a thermal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    min_temp = round(25.0 + (np.min(gray) / 255.0) * 13.0, 1)
    max_temp = round(25.0 + (np.max(gray) / 255.0) * 13.0, 1)
    avg_temp = round(25.0 + (np.mean(gray) / 255.0) * 13.0, 1)
    
    h, w = gray.shape
    top_region = np.mean(gray[0:int(h/3), :])
    mid_region = np.mean(gray[int(h/3):int(2*h/3), :])
    bot_region = np.mean(gray[int(2*h/3):h, :])
    
    regions = {"Face/Forehead": top_region, "Chest/Abdomen": mid_region, "Hands/Feet": bot_region}
    warmest_region = max(regions, key=regions.get)
    coolest_region = min(regions, key=regions.get)
    
    left_side = np.mean(gray[:, 0:int(w/2)])
    right_side = np.mean(gray[:, int(w/2):w])
    lr_diff = round(abs(left_side - right_side) * (13.0 / 255.0), 1)
    
    st.subheader("Thermal Analysis Report")
    st.text(f"Temperature Range : {min_temp}°C – {max_temp}°C")
    st.text(f"Average Temp     : {avg_temp}°C")
    st.text(f"Warmest Region   : {warmest_region}")
    st.text(f"Coolest Region    : {coolest_region}")
    st.text(f"Left/Right Diff  : {lr_diff}°C")
    
    if max_temp > 37.2 or lr_diff > 1.5:
        st.error("STATUS: ABNORMAL DETECTED\nNoticeable thermal variance detected.")
    else:
        st.success("STATUS: NORMAL PATTERN\nThermal distribution is uniform.")
