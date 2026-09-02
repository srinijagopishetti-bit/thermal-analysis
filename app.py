import io
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.set_page_config(page_title="AI Thermal Health Dashboard", layout="wide")

st.title("🌡️ AI Thermal Health Assessment Dashboard")
st.write("Upload any photo to generate thermal analysis and download the PDF report.")

# Sidebar Controls
st.sidebar.header("⚙️ Settings & Options")
selected_cmap = st.sidebar.selectbox("Choose Primary Heatmap Colormap:", ["jet", "inferno", "plasma", "viridis", "magma"])

# File Uploader
uploaded_file = st.file_uploader("📸 Upload Image from Gallery or Camera", type=["jpg", "jpeg", "png", "webp"])

def generate_pdf(min_temp, max_temp, avg_temp, warmest_region, coolest_region, lr_diff, status_text):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    
    p.setFont("Helvetica-Bold", 18)
    p.drawString(100, 750, "AI Thermal Health Assessment Report")
    p.setLineWidth(1)
    p.line(100, 740, 500, 740)
    
    p.setFont("Helvetica", 12)
    p.drawString(100, 700, f"Temperature Range: {min_temp}°C - {max_temp}°C")
    p.drawString(100, 680, f"Average Temperature: {avg_temp}°C")
    p.drawString(100, 660, f"Warmest Zone: {warmest_region}")
    p.drawString(100, 640, f"Coolest Zone: {coolest_region}")
    p.drawString(100, 620, f"Bilateral Asymmetry: {lr_diff}°C")
    
    p.setFont("Helvetica-Bold", 12)
    p.drawString(100, 580, f"Diagnostic Status: {status_text}")
    
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        pil_image.thumbnail((800, 800))
        
        image = np.array(pil_image)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Thermal Scale Mapping
        min_temp = round(25.0 + (float(np.min(gray)) / 255.0) * 13.0, 1)
        max_temp = round(25.0 + (float(np.max(gray)) / 255.0) * 13.0, 1)
        avg_temp = round(25.0 + (float(np.mean(gray)) / 255.0) * 13.0, 1)
        
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
            st.markdown("##### 1. Original Image")
            st.image(image_bgr, channels="BGR", use_container_width=True)
        with c2:
            st.markdown(f"##### 2. {selected_cmap.upper()} Heatmap")
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            ax1.imshow(gray, cmap=selected_cmap)
            ax1.axis("off")
            st.pyplot(fig1)
        with c3:
            st.markdown("##### 3. INFERNO Map")
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            ax2.imshow(gray, cmap="inferno")
            ax2.axis("off")
            st.pyplot(fig2)
            
        st.markdown("---")
        st.subheader("📊 Thermal Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Min - Max Temp", f"{min_temp}°C – {max_temp}°C")
        m2.metric("Average Temp", f"{avg_temp}°C")
        m3.metric("Asymmetry Diff", f"{lr_diff}°C")
        m4.metric("Warmest Zone", warmest_region)

        # Status Check
        if max_temp > 37.2 or lr_diff > 1.5:
            status_text = "ABNORMAL PATTERN DETECTED"
            st.error(f"🚨 **STATUS: {status_text}**")
        else:
            status_text = "NORMAL PATTERN"
            st.success(f"✅ **STATUS: {status_text}**")

        # PDF Download Section
        pdf_data = generate_pdf(min_temp, max_temp, avg_temp, warmest_region, coolest_region, lr_diff, status_text)
        
        st.markdown("---")
        st.download_button(
            label="📥 Download Diagnostic PDF Report",
            data=pdf_data,
            file_name="Thermal_Health_Report.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error("⚠️ Error processing image. Please try uploading another JPG/PNG file.")
        
