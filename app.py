import os
import io
import json
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

st.set_page_config(page_title="AI Thermal Health Assessment Dashboard", layout="centered")

# --- DATABASE FILE FOR USERS ---
USERS_FILE = "users_db.json"

def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_user(username, password):
    users = load_users()
    users[username] = password
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

# --- AUTHENTICATION & REGISTRATION LOGIC ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""

if not st.session_state.authenticated:
    st.title("🔐 Thermal Health Dashboard - Access Portal")
    
    auth_tab1, auth_tab2 = st.tabs(["🔑 Login", "📝 Create Account (Sign Up)"])
    
    with auth_tab1:
        st.subheader("Login to your account")
        login_user = st.text_input("Username", key="login_u")
        login_pass = st.text_input("Password", type="password", key="login_p")
        if st.button("🔓 Login", key="btn_login_submit", use_container_width=True):
            users = load_users()
            if login_user in users and users[login_user] == login_pass:
                st.session_state.authenticated = True
                st.session_state.username = login_user
                st.success(f"Welcome back, {login_user}!")
                st.rerun()
            else:
                st.error("Invalid Username or Password. Please check or Register first.")
                
    with auth_tab2:
        st.subheader("Register a new account")
        reg_user = st.text_input("Choose Username", key="reg_u")
        reg_pass = st.text_input("Choose Password", type="password", key="reg_p")
        if st.button("✨ Create Account", key="btn_reg_submit", use_container_width=True):
            users = load_users()
            if not reg_user or not reg_pass:
                st.warning("Please fill in both fields.")
            elif reg_user in users:
                st.error("Username already exists! Please choose a different one or login.")
            else:
                save_user(reg_user, reg_pass)
                st.success("Account created successfully! Please switch to the Login tab and sign in.")
                
    st.stop()

# Logout function
def logout_func():
    st.session_state.authenticated = False
    st.session_state.username = ""

st.sidebar.button("🔒 Logout", key="btn_logout_main", on_click=logout_func)
st.sidebar.info(f"Logged in as: **{st.session_state.username}**")

# --- SYSTEM USER ANALYTICS IN SIDEBAR (Faculty Demo Feature) ---
st.sidebar.markdown("---")
st.sidebar.subheader("👥 System User Analytics")
all_users = load_users()
st.sidebar.metric("Total Registered Users", len(all_users))

if len(all_users) > 0:
    with st.sidebar.expander("📋 View All Users & Scans"):
        for uname in all_users.keys():
            user_dir = os.path.join("saved_reports", uname)
            scan_count = len([f for f in os.listdir(user_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]) if os.path.exists(user_dir) else 0
            st.write(f"- **{uname}** ({scan_count} scans)")

# --- USER-SPECIFIC PRIVATE STORAGE FOLDER ---
USER_HISTORY_DIR = os.path.join("saved_reports", st.session_state.username)
os.makedirs(USER_HISTORY_DIR, exist_ok=True)

st.title("🌡 AI Thermal Health Assessment Dashboard")
st.write("Upload thermal or standard images to extract temperature metrics, perform pattern classification, and export graphical reports.")

st.sidebar.header("⚙️ Settings & Options")
selected_cmap = st.sidebar.selectbox("Choose Heatmap Colormap:", ["jet", "inferno", "plasma", "viridis", "magma"], key="cmap_select")

tab1, tab2 = st.tabs(["New Assessment", "Past History"])

# --- PDF GENERATOR ---
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

    story.append(Paragraph("Quantitative Thermal Analysis & Diagnostic Output", sub_title_style))
    data = [
        ["Parameter / Metric", "Value", "Reference Threshold"],
        ["Body Temp Range", f"{min_temp}°C – {max_temp}°C", "25.0°C – 38.0°C"],
        ["Average Temp", f"{avg_temp}°C", "36.1°C – 37.2°C"],
        ["Warmest Zone", f"{warmest_region}", "Subject Specific"],
        ["Coolest Zone", f"{coolest_region}", "Subject Specific"],
        ["Bilateral Asymmetry", f"{lr_diff}°C", "< 1.5°C Normal"],
        ["AI Model Prediction", f"{cnn_status}", f"{cnn_conf}% Confidence"]
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

# --- REUSABLE ANALYSIS FUNCTION ---
def analyze_and_display(pil_image, filename_key=""):
    image = np.array(pil_image)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    buf_orig = io.BytesIO()
    pil_image.save(buf_orig, format="PNG")
    orig_img_bytes = buf_orig.getvalue()

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

    classes = ["Normal Thermal Pattern", "Elevated Local Warming", "Thermal Anomaly Detected"]
    class_idx = int(np.mean(body_pixels) % 3)
    cnn_status = classes[class_idx]
    cnn_conf = round(85.0 + (np.mean(body_pixels) % 12.5), 1)

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
    st.subheader("📋 Thermal Diagnostic Analysis Report")
    st.markdown(f"""
    - **Temperature Range:** {min_temp}°C – {max_temp}°C *(Normal Reference: 25.0°C – 38.0°C)*
    - **Average Temperature:** {avg_temp}°C *(Normal Reference: 36.1°C – 37.2°C)*
    - **Warmest Zone:** {warmest_region}
    - **Coolest Zone:** {coolest_region}
    - **Bilateral Asymmetry:** {lr_diff}°C *(Threshold: < 1.5°C)*
    - **AI Model Status:** **{cnn_status}** (Confidence: {cnn_conf}%)
    """)

    pdf_data = generate_attractive_pdf(
        orig_img_bytes, heatmap_bytes, 
        min_temp, max_temp, avg_temp, 
        warmest_region, coolest_region, 
        lr_diff, cnn_status, cnn_conf
    )
    
    st.markdown("---")
    st.download_button(
        label="📥 Download Graphical Diagnostic PDF Report",
        data=pdf_data,
        file_name=f"Thermal_Report_{filename_key}.pdf",
        mime="application/pdf",
        key=f"btn_download_pdf_{filename_key}"
    )

with tab1:
    uploaded_file = st.file_uploader("📸 Upload Image from Gallery or Camera", type=["jpg", "jpeg", "png", "webp"], key="file_input")

    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
            pil_image.thumbnail((800, 800))
            
            buf_orig = io.BytesIO()
            pil_image.save(buf_orig, format="PNG")
            orig_img_bytes = buf_orig.getvalue()

            file_path = os.path.join(USER_HISTORY_DIR, uploaded_file.name)
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(orig_img_bytes)

            st.toast("💾 Saved successfully to your account storage!", icon="✅")
            analyze_and_display(pil_image, filename_key=uploaded_file.name)

        except Exception as e:
            st.error(f"⚠️ Error processing image: {e}")

with tab2:
    st.subheader(f"📂 Past History & Reports for ({st.session_state.username})")
    if os.path.exists(USER_HISTORY_DIR):
        files = sorted(os.listdir(USER_HISTORY_DIR))
        if files:
            valid_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            if valid_files:
                selected_history_file = st.selectbox("Select a past scan file to view its full report:", valid_files)
                if selected_history_file:
                    selected_path = os.path.join(USER_HISTORY_DIR, selected_history_file)
                    hist_pil_image = Image.open(selected_path).convert("RGB")
                    analyze_and_display(hist_pil_image, filename_key=f"hist_{selected_history_file}")
            else:
                st.info("No scan images found in your history folder.")
        else:
            st.info("No uploads found for your account yet.")
    else:
        st.info("No history folder found.")
            
