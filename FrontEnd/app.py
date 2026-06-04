import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="CliniScan", layout="wide")

st.markdown("""
<style>
.hero {
    text-align:center;
    padding:20px 0 30px 0;
}
.hero-title {
    font-size:40px;
    font-weight:700;
    color:#4CAF50;
}
.hero-sub {
    color:#9aa0a6;
    font-size:16px;
}

.section-title {
    font-size:22px;
    font-weight:600;
    margin-top:20px;
    margin-bottom:10px;
}

.card {
    padding:16px;
    border-radius:14px;
    background: linear-gradient(145deg, #1c1f26, #262b34);
    color:white;
    margin-bottom:12px;
    border:1px solid #2f3542;
}

.card:hover {
    border:1px solid #4CAF50;
}

.conf {
    color:#4CAF50;
    font-weight:600;
}

.divider {
    margin:25px 0;
    border-top:1px solid #2a2f3a;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("best_final.pt")

model = load_model()

st.markdown("""
<div class="hero">
    <div class="hero-title">CliniScan</div>
    <div class="hero-sub">AI-powered chest X-ray analysis</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("Settings")
conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

uploaded_file = st.file_uploader("Upload X-ray image", type=["jpg","png","jpeg"])

label_map = {
    "No finding": "Normal",
    "Aortic enlargement": "Enlarged blood vessel",
    "Atelectasis": "Collapsed lung part",
    "Calcification": "Calcium deposits",
    "Cardiomegaly": "Enlarged heart",
    "Consolidation": "Lung infection",
    "ILD": "Chronic lung disease",
    "Infiltration": "Possible infection",
    "Lung Opacity": "Abnormal lung patch",
    "Nodule/Mass": "Suspicious lump",
    "Other lesion": "Unusual abnormal area",
    "Pleural effusion": "Fluid in lungs",
    "Pleural thickening": "Thick lung lining",
    "Pneumothorax": "Collapsed lung",
    "Pulmonary fibrosis": "Lung scarring"
}

link_map = {
    "No finding": "https://www.nhs.uk/conditions/",
    "Aortic enlargement": "https://www.michiganmedicine.org/health-lab/enlarged-aorta-risks-and-symptoms-what-know",
    "Atelectasis": "https://www.mayoclinic.org/diseases-conditions/atelectasis/symptoms-causes/syc-20369684",
    "Calcification": "https://my.clevelandclinic.org/health/diseases/23117-calcium-deposits",
    "Cardiomegaly": "https://my.clevelandclinic.org/health/diseases/21490-enlarged-heart-cardiomegaly",
    "Consolidation": "https://www.healthline.com/health/lung-consolidation",
    "ILD": "https://my.clevelandclinic.org/health/diseases/17809-interstitial-lung-disease",
    "Infiltration": "https://www.mylungcancerteam.com/resources/what-are-lung-infiltrates-causes-and-risk-for-lung-cancer",
    "Lung Opacity": "https://www.healthline.com/health/lung-opacity",
    "Nodule/Mass": "https://my.clevelandclinic.org/health/diseases/14799-pulmonary-nodules",
    "Other lesion": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11193846/",
    "Pleural effusion": "https://my.clevelandclinic.org/health/diseases/17373-pleural-effusion",
    "Pleural thickening": "https://www.asbestos.com/mesothelioma/pleural-thickening/",
    "Pneumothorax": "https://www.mayoclinic.org/diseases-conditions/pneumothorax/symptoms-causes/syc-20350367",
    "Pulmonary fibrosis": "https://www.mayoclinic.org/diseases-conditions/pulmonary-fibrosis/symptoms-causes/syc-20353690"
}

if uploaded_file:

    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file)

    with col1:
        st.markdown('<div class="section-title">Original Image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        path = tmp.name

    results = model.predict(source=path, conf=conf)

    with col2:
        st.markdown('<div class="section-title">Detection Result</div>', unsafe_allow_html=True)
        st.image(results[0].plot(), use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Detected Abnormalities</div>', unsafe_allow_html=True)

    if len(results[0].boxes) == 0:
        st.success("No abnormalities detected")
    else:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            label = model.names[cls_id]
            simple = label_map.get(label, label)
            url = link_map.get(label, "#")

            st.markdown(f"""
            <div class="card">
                <b>
                    <a href="{url}" target="_blank" style="color:#4CAF50; text-decoration:none;">
                        {simple}
                    </a>
                </b><br>
                <span style="color:#888;">{label}</span><br>
                Confidence: <span class="conf">{confidence:.2f}</span>
            </div>
            """, unsafe_allow_html=True)

    os.remove(path)

else:
    st.info("Upload an X-ray image to begin")