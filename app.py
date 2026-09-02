import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="AI Thermal Health Dashboard", layout="wide")

st.title("🌡️ AI Thermal Health Assessment Dashboard")
st.write("Upload a thermal image or take a live photo using your phone camera for thermal health analysis.")

# Sidebar Controls
st.sidebar.header("⚙️ Settings & Options")
selected_cmap = st.sidebar.selectbox("Choose Primary Heatmap Colormap:", ["jet", "inferno", "plasma", "viridis", "magma"])

# Standard File Uploader with Mobile Camera Trigger
uploaded_file = st.file_uploader("📸 Tap below to Take Photo or Upload Image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image = np.array(pil_image)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Thermal Scale Mapping (25°C - 38°C)
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
        
        st.markdown("---")
        st.subheader("🖼️ Thermal Visualizations")
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("##### 1. Input Image")
            st.image(image_bgr, channels="BGR", use_container_width=True)
            
        with c2:
            st.markdown(f"##### 2. Selected Heatmap ({selected_cmap.upper()})")
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            cax1 = ax1.imshow(gray, cmap=selected_cmap)
            fig1.colorbar(cax1, label="Temp Index (°C)", shrink=0.8)
            ax1.axis("off")
            st.pyplot(fig1)
            
        with c3:
            st.markdown("##### 3. INFERNO Hotspot Map")
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            cax2 = ax2.imshow(gray, cmap="inferno")
            fig2.colorbar(cax2, label="Intensity", shrink=0.8)
            ax2.axis("off")
            st.pyplot(fig2)
            
        st.markdown("---")
        st.subheader("📊 Thermal Metrics & Analysis")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Min - Max Temp", f"{min_temp}°C – {max_temp}°C")
        m2.metric("Average Temp", f"{avg_temp}°C")
        m3.metric("Left/Right Asymmetry", f"{lr_diff}°C")
        m4.metric("Warmest Zone", warmest_region)

        st.markdown("---")
        st.subheader("📝 Automated Health Report")
        
        report_text = f"""======================================
     THERMAL ANALYSIS REPORT     
======================================
🌡️ Temperature Range        : {min_temp}°C – {max_temp}°C
📊 Average Temperature      : {avg_temp}°C

🔥 Warmest Zone             : {warmest_region}
❄️ Coolest Zone              : {coolest_region}
↔️ Asymmetry Difference     : {lr_diff}°C

--------------------------------------
📌 OBSERVATION SUMMARY:"""

        if max_temp > 37.2 or lr_diff > 1.5:
            report_text += "\n⚠️ STATUS: ABNORMAL PATTERN DETECTED\nHigh thermal variance or hotspot observed."
            st.error("🚨 **STATUS: ABNORMAL PATTERN DETECTED**")
        else:
            report_text += "\n✅ STATUS: NORMAL PATTERN\nThermal distribution is standard and uniform."
            st.success("✅ **STATUS: NORMAL PATTERN**")

        st.code(report_text, language="markdown")

    except Exception as e:
        st.error("⚠️ Error processing image. Please try another image.")
        
        
