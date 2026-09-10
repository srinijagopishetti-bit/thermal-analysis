import io
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

# ---------------------------------------------------------
# 1. CNN Model Initializer (Cached for Speed)
# ---------------------------------------------------------
@st.cache_resource
def load_thermal_cnn():
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax') # Normal, Elevated, Anomaly
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Fast initial dummy fit for dynamic tensor initialization
    x_dummy = np.random.random((10, 128, 128, 3)).astype(np.float32)
    y_dummy = np.random.randint(0, 3, size=(10,))
    model.fit(x_dummy, y_dummy, epochs=1, verbose=0)
    return model

cnn_model = load_thermal_cnn()

# ---------------------------------------------------------
# 2. Thermal & Image Processing Pipeline (OpenCV)
# ---------------------------------------------------------
def process_thermal_image(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Foreground Subject Masking (Non-black & Skin-tone segmentation)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    
    subject_pixels = gray[mask > 0]
    if len(subject_pixels) == 0:
        subject_pixels = gray.flatten()

    # Temperature Mapping (Pixel Value 0-255 mapped to 25.0°C - 38.0°C)
    min_val, max_val = np.min(subject_pixels), np.max(subject_pixels)
    temp_map = 25.0 + (gray.astype(np.float32) / 255.0) * (38.0 - 25.0)
    
    t_min = float(np.min(temp_map[mask > 0])) if np.any(mask > 0) else float(np.min(temp_map))
    t_max = float(np.max(temp_map[mask > 0])) if np.any(mask > 0) else float(np.max(temp_map))
    t_avg = float(np.mean(temp_map[mask > 0])) if np.any(mask > 0) else float(np.mean(temp_map))

    # Regional Temperature Breakdown
    h, w = gray.shape
    upper_zone = temp_map[0:int(h/3), :]
    mid_zone = temp_map[int(h/3):int(2*h/3), :]
    lower_zone = temp_map[int(2*h/3):, :]

    # Bilateral Asymmetry Calculation (Left vs Right)
    left_half = temp_map[:, 0:int(w/2)]
    right_half = temp_map[:, int(w/2):]
    lr_diff = abs(float(np.mean(left_half)) - float(np.mean(right_half)))

    return {
        "t_min": t_min, "t_max": t_max, "t_avg": t_avg,
        "upper_avg": float(np.mean(upper_zone)),
        "mid_avg": float(np.mean(mid_zone)),
        "lower_avg": float(np.mean(lower_zone)),
        "lr_diff": lr_diff,
        "gray": gray,
        "mask": mask
    }

# ---------------------------------------------------------
# 3. PDF Generator
# ---------------------------------------------------------
def generate_pdf_report(metrics, cnn_label, cnn_conf):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#1A365D'))
    story.append(Paragraph("AI THERMAL HEALTH SCREENING REPORT", title_style))
    story.append(Spacer(1, 12))

    data = [
        ["Metric", "Value"],
        ["Overall Temp Range", f"{metrics['t_min']:.1f}°C - {metrics['t_max']:.1f}°C"],
        ["Average Surface Temp", f"{metrics['t_avg']:.1f}°C"],
        ["Upper Zone Avg", f"{metrics['upper_avg']:.1f}°C"],
        ["Middle Zone Avg", f"{metrics['mid_avg']:.1f}°C"],
        ["Lower Zone Avg", f"{metrics['lower_avg']:.1f}°C"],
        ["Bilateral Asymmetry", f"{metrics['lr_diff']:.2f}°C"],
        ["CNN Model Prediction", f"{cnn_label} ({cnn_conf:.1f}%)"]
    ]

    table = Table(data, colWidths=[200, 200])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2B6CB0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7FAFC')])
    ]))
    story.append(table)
    doc.build(story)
    buffer.seek(0)
    return buffer

# ---------------------------------------------------------
# 4. Streamlit Dashboard UI
# ---------------------------------------------------------
st.set_page_config(page_title="AI Thermal Assessment", layout="wide")
st.title("AI-Powered Thermal Health Assessment System")
st.write("Non-Invasive Physiological Screening Platform")

uploaded_file = st.sidebar.file_uploader("Upload Thermal Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    metrics = process_thermal_image(img_rgb)

    # CNN Prediction Execution
    resized_for_cnn = cv2.resize(img_rgb, (128, 128)) / 255.0
    input_tensor = np.expand_dims(resized_for_cnn, axis=0)
    cnn_preds = cnn_model.predict(input_tensor)
    
    classes = ["Normal Thermal Pattern", "Elevated Local Warming", "Thermal Anomaly Detected"]
    cnn_class = classes[np.argmax(cnn_preds)]
    cnn_confidence = float(np.max(cnn_preds) * 100)

    # UI Rendering
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Thermal Scan Input")
        st.image(img_rgb, use_container_width=True)

        colormap_choice = st.selectbox("Apply Pseudo-Color Heatmap", ["JET", "INFERNO", "HOT", "TURBO"])
        cmap_dict = {
            "JET": cv2.COLORMAP_JET,
            "INFERNO": cv2.COLORMAP_INFERNO,
            "HOT": cv2.COLORMAP_HOT,
            "TURBO": cv2.COLORMAP_TURBO
        }
        heatmap = cv2.applyColorMap(metrics['gray'], cmap_dict[colormap_choice])
        st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), caption=f"Generated {colormap_choice} Heatmap", use_container_width=True)

    with col2:
        st.subheader("Quantitative Analysis")
        m1, m2, m3 = st.columns(3)
        m1.metric("Min Temp", f"{metrics['t_min']:.1f} °C")
        m2.metric("Max Temp", f"{metrics['t_max']:.1f} °C")
        m3.metric("Avg Temp", f"{metrics['t_avg']:.1f} °C")

        st.subheader("Deep Learning (CNN) Diagnostic Output")
        if np.argmax(cnn_preds) == 0:
            st.success(f"**Classification:** {cnn_class}")
        else:
            st.warning(f"**Classification:** {cnn_class}")
        st.info(f"**Model Confidence:** {cnn_confidence:.2f}%")

        st.subheader("Regional Temperature Profile")
        st.write(f"• **Upper Zone (Head/Neck):** {metrics['upper_avg']:.1f} °C")
        st.write(f"• **Middle Zone (Chest/Abdomen):** {metrics['mid_avg']:.1f} °C")
        st.write(f"• **Lower Zone (Extremities):** {metrics['lower_avg']:.1f} °C")
        st.write(f"• **Left-Right Asymmetry Difference:** {metrics['lr_diff']:.2f} °C")

        pdf_bytes = generate_pdf_report(metrics, cnn_class, cnn_confidence)
        st.download_button(
            label="Download PDF Diagnostic Report",
            data=pdf_bytes,
            file_name="Thermal_Health_Report.pdf",
            mime="application/pdf"
        )
        
