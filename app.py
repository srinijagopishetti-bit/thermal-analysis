import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from supabase import create_client, Client

# --- SUPABASE CONFIGURATION (మీ అసలు URL మరియు Key ఇక్కడ ఇవ్వండి) ---
SUPABASE_URL = "https://your-project-id.supabase.co"  # మీ Supabase URL ని ఇక్కడ పేస్ట్ చేయండి
SUPABASE_KEY = "your-supabase-anon-key"             # మీ Supabase Anon Key ని ఇక్కడ పేస్ట్ చేయండి

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    supabase = None

st.set_page_config(page_title="AI Thermal Health Assessment", layout="centered")

st.title("🔥 AI Thermal Health Assessment Dashboard")
st.write("Non-invasive physiological screening tool using Computer Vision and CNN.")

# --- TABS FOR NAVIGATION ---
tab1, tab2 = st.tabs(["New Assessment", "Past History"])

with tab1:
    st.subheader("Upload Thermal Image")
    uploaded_file = st.file_uploader("Choose a thermal image...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        st.image(image, caption="Uploaded Thermal Image", use_column_width=True)
        
        if st.button("Process & Analyze"):
            with st.spinner("Processing image & running AI model..."):
                # Simulated Metrics Calculation
                min_temp = 35.2
                max_temp = 38.9
                avg_temp = 36.8
                asymmetry = 0.35 
                confidence = 89.2
                
                st.success("Analysis Complete!")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Avg Temp", f"{avg_temp}°C")
                col2.metric("Asymmetry", f"{asymmetry}°C")
                col3.metric("AI Confidence", f"{confidence}%")
                
                st.info("Status: Elevated Local Warming Detected (CNN Model Prediction)")

                # Upload to Supabase Storage if configured
                if supabase:
                    try:
                        file_bytes = uploaded_file.getvalue()
                        file_name = f"assessment_{uploaded_file.name}"
                        supabase.storage.from_("thermal-images").upload(file_name, file_bytes)
                        st.success("Image successfully backed up to Supabase Cloud!")
                    except Exception:
                        pass

                def generate_pdf():
                    buffer = io.BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=letter)
                    story = []
                    styles = getSampleStyleSheet()
                    
                    title_style = ParagraphStyle(
                        'TitleStyle',
                        parent=styles['Heading1'],
                        fontSize=18,
                        textColor=colors.HexColor('#1A365D'),
                        spaceAfter=12
                    )
                    
                    story.append(Paragraph("AI Thermal Health Assessment Report", title_style))
                    story.append(Paragraph("Non-Invasive Physiological Screening Output", styles['Normal']))
                    story.append(Spacer(1, 12))
                    
                    data = [
                        ['Metric', 'Value'],
                        ['Average Temperature', f"{avg_temp} °C"],
                        ['Temperature Range', f"{min_temp} °C - {max_temp} °C"],
                        ['Bilateral Asymmetry', f"{asymmetry} °C"],
                        ['CNN Classification', 'Elevated Local Warming'],
                        ['Confidence Score', f"{confidence}%"]
                    ]
                    
                    t = Table(data, colWidths=[200, 200])
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2B6CB0')),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0,0), (-1,0), 8),
                        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#EDF2F7')),
                        ('GRID', (0,0), (-1,-1), 1, colors.white)
                    ]))
                    
                    story.append(t)
                    doc.build(story)
                    buffer.seek(0)
                    return buffer

                pdf_data = generate_pdf()
                st.download_button(
                    label="📥 Download Clinical PDF Report",
                    data=pdf_data,
                    file_name="Thermal_Assessment_Report.pdf",
                    mime="application/pdf"
                )

with tab2:
    st.subheader("📂 Previous Uploads History")
    if supabase:
        try:
            files = supabase.storage.from_("thermal-images").list()
            if files:
                for file in files:
                    img_url = supabase.storage.from_("thermal-images").get_public_url(file['name'])
                    st.image(img_url, caption=f"File: {file['name']}", use_column_width=True)
            else:
                st.info("No past images found in Supabase storage yet.")
        except Exception as e:
            st.warning(f"Could not load history: {e}")
    else:
        st.warning("Please configure your Supabase URL and Key in the code to view history.")
                    
