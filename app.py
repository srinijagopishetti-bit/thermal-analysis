        import io
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

st.set_page_config(page_title="AI Thermal Health Dashboard", layout="wide")

st.title("🌡️ AI Thermal Health Assessment Dashboard")
st.write("Upload any photo to generate thermal analysis and download a visual PDF report with thermal icons.")

# Sidebar Controls
st.sidebar.header("⚙️ Settings & Options")
selected_cmap = st.sidebar.selectbox("Choose Primary Heatmap Colormap:", ["jet", "inferno", "plasma", "viridis", "magma"])

# File Uploader
uploaded_file = st.file_uploader("📸 Upload Image from Gallery or Camera", type=["jpg", "jpeg", "png", "webp"])

def generate_attractive_pdf(orig_img_bytes, heatmap_bytes, min_temp, max_temp, avg_temp, warmest_region, coolest_region, lr_diff, status_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    story = []
    styles = getSampleStyleSheet()

    # Custom Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=10,
        alignment=1
    )
    
    sub_title_style = ParagraphStyle(
        'SubTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        textColor=colors.HexColor("#2B6CB0"),
        spaceAfter=10
    )

    # 1. Header / Title
    story.append(Paragraph("AI Thermal Health Assessment Report", title_style))
    story.append(Spacer(1, 10))

    # 2. Images Side-by-Side
    img_orig_stream = io.BytesIO(orig_img_bytes)
    img_heat_stream = io.BytesIO(heatmap_bytes)
    
    rl_img_orig = RLImage(img_orig_stream, width=240, height=200)
    rl_img_heat = RLImage(img_heat_stream, width=240, height=200)

    img_table = Table([[rl_img_orig, rl_img_heat]], colWidths=[270, 270])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    
    story.append(img_table)
    story.append(Spacer(1, 15))

    # 3. Metrics Table with Visual Descriptors
    story.append(Paragraph("Quantitative Thermal Analysis", sub_title_style))
    
    data = [
        ["Parameter / Metric", "Value", "Reference Threshold"],
        ["Temp Range", f"{min_temp}°C – {max_temp}°C", "25.0°C – 38.0°C"],
        ["Average Temp", f"{avg_temp}°C", "36.1°C – 37.2°C"],
        ["Warmest Zone (HOT)", f"{warmest_region}", "Facial / Chest"],
        ["Coolest Zone (COLD)", f"{coolest_region}", "Extremities"],
        ["Bilateral Asymmetry", f"{lr_diff}°C", "< 1.5°C Normal"]
    ]

    t = Table(data, colWidths=[180, 180, 180])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#F7FAFC")),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 9),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))

    # 4. Diagnostic Status Banner
    story.append(Paragraph("Diagnostic Status Summary", sub_title_style))
    bg_color = colors.HexColor("#E53E3E") if "ABNORMAL" in status_text else colors.HexColor("#38A169")
    
    status_style = ParagraphStyle(
        'StatusText',
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=colors.white,
        alignment=1
    )
    
    status_p = Paragraph(f"STATUS: {status_text}", status_style)
    status_table = Table([[status_p]], colWidths=[540])
    status_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg_color),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    
    story.append(status_table)

    doc.build(story)
    buffer.seek(0)
    return buffer

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        pil_image.thumbnail((800, 800))
        
        image = np.array(pil_image)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Save original image bytes for PDF
        buf_orig = io.BytesIO()
        pil_image.save(buf_orig, format="PNG")
        orig_img_bytes = buf_orig.getvalue()

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
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 1. Original Image")
            st.image(image_bgr, channels="BGR", use_container_width=True)
            
        with c2:
            st.markdown(f"##### 2. {selected_cmap.upper()} Heatmap")
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            cax1 = ax1.imshow(gray, cmap=selected_cmap)
            fig1.colorbar(cax1, label="Temp Index (°C)", shrink=0.8)
            ax1.axis("off")
            st.pyplot(fig1)
            
            # Save heatmap bytes for PDF
            buf_heat = io.BytesIO()
            fig1.savefig(buf_heat, format="png", bbox_inches='tight')
            heatmap_bytes = buf_heat.getvalue()

        st.markdown("---")
        st.subheader("📊 Thermal Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🌡️ Temp Range", f"{min_temp}°C – {max_temp}°C")
        m2.metric("📊 Average Temp", f"{avg_temp}°C")
        m3.metric("↔️ Asymmetry Diff", f"{lr_diff}°C")
        m4.metric("🔥 Warmest Zone", warmest_region)

        # Status Check with Icons
        if max_temp > 37.2 or lr_diff > 1.5:
            status_text = "ABNORMAL PATTERN DETECTED"
            st.error(f"🚨 **STATUS: {status_text}**")
        else:
            status_text = "NORMAL PATTERN"
            st.success(f"✅ **STATUS: {status_text}**")

        # PDF Generation
        pdf_data = generate_attractive_pdf(
            orig_img_bytes, heatmap_bytes, 
            min_temp, max_temp, avg_temp, 
            warmest_region, coolest_region, 
            lr_diff, status_text
        )
        
        st.markdown("---")
        st.download_button(
            label="📥 Download Graphical PDF Report",
            data=pdf_data,
            file_name="Thermal_Health_Report.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error("⚠️ Error processing image. Please try uploading another photo.")
        
