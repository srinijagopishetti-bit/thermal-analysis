import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="AI Thermal Imaging Dashboard", layout="wide")

st.title("🌡️ AI Thermal Imaging Health Assessment")
st.write("Upload a thermal image for a complete visual analysis and preliminary health report.")

# Added webp and lower/upper case extensions for Mobile compatibility
uploaded_file = st.file_uploader("📷 Choose a thermal image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        # PIL handles mobile/downloaded formats (.webp, compressed jpgs) seamlessly
        pil_image = Image.open(uploaded_file).convert("RGB")
        image = np.array(pil_image)
        # Convert RGB (PIL) to BGR (OpenCV)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Thermal Mapping Logic
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
        st.subheader("🖼️ Thermal Image Visualizations & Heatmaps")
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("##### 1. Original Thermal Image")
            st.image(image, channels="BGR", use_container_width=True)
            
        with c2:
            st.markdown("##### 2. JET Color Heatmap")
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            cax1 = ax1.imshow(gray, cmap="jet")
            fig1.colorbar(cax1, label="Temp Index (°C)", shrink=0.8)
            ax1.axis("off")
            st.pyplot(fig1)
            
        with c3:
            st.markdown("##### 3. INFERNO Hotspot Map")
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            cax2 = ax2.imshow(gray, cmap="inferno")
            fig2.colorbar(cax2, label="Intensity Scale", shrink=0.8)
            ax2.axis("off")
            st.pyplot(fig2)
            
        st.markdown("---")
        
        st.subheader("📊 Quick Metric Cards")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Min - Max Temp", f"{min_temp}°C – {max_temp}°C")
        m2.metric("Average Temp", f"{avg_temp}°C")
        m3.metric("Left/Right Diff", f"{lr_diff}°C")
        m4.metric("Warmest Region", warmest_region)

        st.markdown("---")
        
        st.subheader("📝 Generated Thermal Analysis Report")
        
        report_text = f"""======================================
     THERMAL IMAGE ANALYSIS REPORT     
======================================
🌡️ Temperature Range : {min_temp}°C – {max_temp}°C
📊 Average Temperature : {avg_temp}°C

🔥 Warmest Region     : {warmest_region}
❄️ Coolest Region      : {coolest_region}
↔️ Left/Right Diff    : {lr_diff}°C

--------------------------------------
📌 OVERALL PATTERN & OBSERVATION:"""

        if max_temp > 37.2 or lr_diff > 1.5:
            report_text += "\n⚠️ STATUS: ABNORMAL DETECTED\nNoticeable thermal variance/hotspot detected. Recommend secondary evaluation if persistent."
            st.error("🚨 **STATUS: ABNORMAL DETECTED** — Noticeable thermal variance detected!")
        else:
            report_text += "\n✅ STATUS: NORMAL PATTERN\nThermal distribution is uniform and within standard non-febrile thresholds."
            st.success("✅ **STATUS: NORMAL PATTERN** — Thermal distribution is uniform.")

        st.code(report_text, language="markdown")

    except Exception as e:
        st.error("⚠️ Unsupported image format or corrupted download. Please upload a standard JPG/PNG file.")
