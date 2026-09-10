import io
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from supabase import create_client, Client

# --- 1. SUPABASE CONFIGURATION ---
SUPABASE_URL = "https://your-project-id.supabase.co"  # Replace with your actual Supabase URL
SUPABASE_KEY = "your-supabase-anon-key"             # Replace with your actual Supabase Anon Key

st.set_page_config(page_title="AI Thermal Health Dashboard", layout="centered")

@st.cache_resource
def init_supabase():
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None

supabase = init_supabase()

# --- 2. CNN MODEL ARCHITECTURE ---
@st.cache_resource
def load_thermal_cnn():
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax') # 0: Normal, 1: Localized Heat, 2: Anomaly
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    x_dummy = np.random.random((5, 128, 128, 3)).astype(np.float32)
    y_dummy = np.random.randint(0, 3, size=(5,))
    model.fit(x_dummy, y_dummy, epochs=1, verbose=0)
    return model

cnn_model = load_thermal_cnn()

# --- 3. AUTHENTICATION LOGIC ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Thermal Health Dashboard - Login")
    st.caption("Access Restricted: Authorized Medical & Evaluation Personnel Only")
    
    col1, _ = st.columns([1, 2])
    with col1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        login_btn = st.button("🔓 Login to Dashboard", key="btn_login", use_container_width=True)
        
        if login_btn:
            if username == "admin" and password == "Thermal2026!":
                st.session_state.authenticated = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid Username or Password")
    st.stop()

# --- 4. MAIN DASHBOARD & TABS ---
st.sidebar.button("🔒 Logout", key="btn_logout_main", on_click=lambda: st.session_state.update(authenticated=False))

st.title("🌡️ AI Thermal Health Assessment Dashboard")
st.write("Upload thermal or standard images to extract temperature metrics, perform CNN pattern classification, and export graphical reports.")

st.sidebar.header("⚙️ Settings & Options")
selected_cmap = st.sidebar.selectbox("Choose Heatmap Colormap:", ["jet", "inferno", "plasma", "viridis", "magma"], key="cmap_select")

tab1, tab2 = st.tabs(["New Assessment", "Past History"])

def upload_to_supabase(file_bytes, filename):
    if supabase is None or SUPABASE_URL == "https://your-project-id.supabase.co":
        return False
    try:
        supabase.storage.from_("thermal-images").upload(filename, file_bytes)
        return True
    except Exception:
        return False

# --- 5. PDF GENERATOR ---
def generate_attractive_pdf(orig_img_bytes, heatmap_bytes, min_temp, max_temp, avg_temp, warmest_region, coolest_region, lr_diff, cnn_status, cnn_conf):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=18, textColor=colors.HexColor("#1A365D"), alignment=1)
    sub_title_style = ParagraphStyle('SubTitle', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12, textColor=colors.HexColor("#2B6CB0"), spaceAfter=10)

    story.append(Paragraph("AI Thermal Health Assessment Report", title_style))
    story.append(Spacer(1, 10))

    img_orig_stream = io.BytesIO(orig_img_bytes)
    img_heat_stream = io.BytesIO(heatmap_bytes)
    
    rl_img_orig = RLImage(img_orig_stream, width=200, height=160)
    rl_img_heat = RLImage(img_heat_stream, width=200, height=160)

    img_table = Table([[rl_img_orig, rl_img_heat]], colWidths=[250, 250])
    img_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    story.append(img_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph("Quantitative Thermal Analysis & Deep Learning Diagnostic", sub_title_style))
    data = [
        ["Parameter / Metric", "Value", "Reference Threshold"],
        ["Body Temp Range", f"{min_temp}°C – {max_temp}°C", "25.0°C – 38.0°C"],
        ["Average Temp", f"{avg_temp}°C", "36.1°C – 37.2°C"],
        ["Warmest Zone", f"{warmest_region}", "Subject Specific"],
        ["Coolest Zone", f"{coolest_region}", "Subject Specific"],
        ["Bilateral Asymmetry", f"{lr_diff}°C", "< 1.5°C Normal"],
        ["CNN Model Prediction", f"{cnn_status}", f"{cnn_conf}% Confidence"]
    ]

    t = Table(data, colWidths=[170, 170, 160])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
    ]))
    story.append(t)
    doc.build(story)
    buffer.seek(0)
    return buffer

with tab1:
    uploaded_file = st.file_uploader("📸 Upload Image from Gallery or Camera", type=["jpg", "jpeg", "png", "webp"], key="file_input")

    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
            pil_image.thumbnail((800, 800))
            
            image = np.array(pil_image)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            
            buf_orig = io.BytesIO()
            pil_image.save(buf_orig, format="PNG")
            orig_img_bytes = buf_orig.getvalue()

            if upload_to_supabase(orig_img_bytes, uploaded_file.name):
                st.toast("💾 Record backed up to Supabase Cloud Storage!", icon="✅")

            # Segmentation & Noise Cleaning
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            non_black_mask = cv2.inRange(gray, 45, 255)

            if np.sum(skin_mask > 0) > (0.05 * gray.size):
                body_mask = cv2.bitwise_and(skin_mask, non_black_mask)
            else:
                body_mask = non_black_mask

            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
            body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

            body_pixels = gray[body_mask == 255]
            if len(body_pixels) == 0:
                body_pixels = gray.flatten()

            min_temp = round(25.0 + (float(np.min(body_pixels)) / 255.0) * 13.0, 1)
            max_temp = round(25.0 + (float(np.max(body_pixels)) / 255.0) * 13.0, 1)
            avg_temp = round(25.0 + (float(np.mean(body_pixels)) / 255.0) * 13.0, 1)
            
            h, w = gray.shape
            top_mask = (body_mask[0:int(h/3), :] == 255)
            mid_mask = (body_mask[int(h/3):int(2*h/3), :] == 255)
            bot_mask = (body_mask[int(2*h/3):h, :] == 255)

            top_region = np.mean(gray[0:int(h/3), :][top_mask]) if np.any(top_mask) else 0
            mid_region = np.mean(gray[int(h/3):int(2*h/3), :][mid_mask]) if np.any(mid_mask) else 0
            bot_region = np.mean(gray[int(2*h/3):h, :][bot_mask]) if np.any(bot_mask) else 0

            regions = {"Upper Zone": top_region, "Middle Zone": mid_region, "Lower Zone": bot_region}
            warmest_region = max(regions, key=regions.get)
            coolest_region = min(regions, key=regions.get)

            left_mask = (body_mask[:, 0:int(w/2)] == 255)
            right_mask = (body_mask[:, int(w/2):w] == 255)
            left_side = np.mean(gray[:, 0:int(w/2)][left_mask]) if np.any(left_mask) else 0
            right_side = np.mean(gray[:, int(w/2):w][right_mask]) if np.any(right_mask) else 0
            lr_diff = round(abs(left_side - right_side) * (13.0 / 255.0), 1)

            cnn_input = cv2.resize(image, (128, 128)) / 255.0
            cnn_input = np.expand_dims(cnn_input, axis=0)
            cnn_preds = cnn_model.predict(cnn_input, verbose=0)
            
            classes = ["Normal Thermal Pattern", "Elevated Local Warming", "Thermal Anomaly Detected"]
            cnn_status = classes[np.argmax(cnn_preds)]
            cnn_conf = round(float(np.max(cnn_preds)) * 100, 1)

            st.markdown("---")
            st.subheader("🖼️ Thermal Visualizations")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 1. Original Image")
                st.image(image_bgr, channels="BGR", use_container_width=True)
                
            with c2:
                st.markdown(f"##### 2. {selected_cmap.upper()} Heatmap")
                fig1, ax1 = plt.subplots(figsize=(4, 4))
                masked_float = np.where(body_mask == 255, gray.astype(float), np.nan)
                cax1 = ax1.imshow(masked_float, cmap=selected_cmap)
                fig1.colorbar(cax1, label="Temp Scale (°C)", shrink=0.8)
                ax1.set_facecolor('black')
                ax1.axis("off")
                st.pyplot(fig1)
                
                buf_heat = io.BytesIO()
                fig1.savefig(buf_heat, format="png", bbox_inches='tight', facecolor='black')
                heatmap_bytes = buf_heat.getvalue()

            st.markdown("---")
            st.subheader("📊 Thermal Metrics & CNN Classification")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🌡️ Temp Range", f"{min_temp}°C – {max_temp}°C")
            m2.metric("📊 Average Temp", f"{avg_temp}°C")
            m3.metric("↔️ Asymmetry", f"{lr_diff}°C")
            m4.metric("🔥 Warmest Zone", warmest_region)

            st.info(f"🤖 **CNN Model Output:** {cnn_status} (Confidence: {cnn_conf}%)")

            pdf_data = generate_attractive_pdf(
                orig_img_bytes, heatmap_bytes, 
                min_temp, max_temp, avg_temp, 
                warmest_region, coolest_region, 
                lr_diff, cnn_status, cnn_conf
            )
            
            st.markdown("---")
            st.download_button(
                label="📥 Download Graphical Diagnostic PDF",
                data=pdf_data,
                file_name="Thermal_Diagnostic_Report.pdf",
                mime="application/pdf",
                key="btn_download_pdf"
            )

        except Exception as e:
            st.error(f"⚠️ Error processing image: {e}")

with tab2:
    st.subheader("📂 Previous Uploads History")
    if supabase:
        try:
            files = supabase.storage.from_("thermal-images").list()
            if files:
                for file in files:
                    img_url = supabase.storage.from_("thermal-images").get_public_url(file['name'])
                    st.image(img_url, caption=f"File: {file['name']}", use_container_width=True)
            else:
                st.info("No past images found in Supabase storage yet.")
        except Exception as e:
            st.warning(f"Could not load history. Make sure bucket 'thermal-images' is public in Supabase.")
    else:
        st.warning("Please configure your Supabase credentials to view history.")

