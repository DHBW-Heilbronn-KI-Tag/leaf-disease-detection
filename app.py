import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import requests
import os
import json
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KI-Pflanzendoktor",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }
    .accent { color: #22c55e; }
    .subtitle { color: #888; font-size: 1rem; margin-bottom: 2rem; font-weight: 300; }

    .result-box {
        background: linear-gradient(135deg, #0f2017 0%, #0a1a0f 100%);
        border: 1px solid #22c55e44;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .result-label { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em;
                    text-transform: uppercase; color: #22c55e; margin-bottom: 0.3rem; }
    .result-disease { font-family: 'Syne', sans-serif; font-size: 1.6rem;
                      font-weight: 800; color: #f0f0f0; margin-bottom: 0.2rem; }
    .result-plant { font-size: 1rem; color: #aaa; margin-bottom: 1rem; }

    .healthy-box {
        background: linear-gradient(135deg, #0f2017 0%, #0a1a0f 100%);
        border: 1px solid #22c55e;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .sick-box {
        background: linear-gradient(135deg, #1f0f0f 0%, #150808 100%);
        border: 1px solid #ef444466;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .info-chip {
        display: inline-block;
        background: rgba(34,197,94,0.1);
        border: 1px solid rgba(34,197,94,0.25);
        border-radius: 100px;
        padding: 3px 10px;
        font-size: 0.78rem;
        color: #22c55e;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    .example-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #666;
        margin-bottom: 0.5rem;
    }

    div[data-testid="stButton"] > button {
        border-radius: 12px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
    }

    .context-box {
        background: rgba(34,197,94,0.05);
        border-left: 3px solid #22c55e;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 1rem 0;
        font-size: 0.88rem;
        color: #aaa;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# ── Class names (PlantVillage 38 classes) ────────────────────────────────────
CLASS_NAMES = [
    'Apfel – Schorf',
    'Apfel – Schwarzfäule',
    'Apfel – Rostfleckenkrankheit',
    'Apfel – Gesund',
    'Heidelbeere – Gesund',
    'Kirsche – Mehltau',
    'Kirsche – Gesund',
    'Mais – Grauer Blattfleck',
    'Mais – Gemeiner Rost',
    'Mais – Nördliche Blattfleckenkrankheit',
    'Mais – Gesund',
    'Weintraube – Schwarzfäule',
    'Weintraube – Schwarzer Mehltau',
    'Weintraube – Blattfleckenkrankheit',
    'Weintraube – Gesund',
    'Orange – Huanglongbing (Zitrusgreening)',
    'Pfirsich – Bakterienbrand',
    'Pfirsich – Gesund',
    'Paprika – Bakterienbrand',
    'Paprika – Gesund',
    'Kartoffel – Frühfäule',
    'Kartoffel – Kraut- und Knollenfäule',
    'Kartoffel – Gesund',
    'Himbeere – Gesund',
    'Sojabohne – Gesund',
    'Kürbis – Mehltau',
    'Erdbeere – Blattdürre',
    'Erdbeere – Gesund',
    'Tomate – Bakterienbrand',
    'Tomate – Früh fäule',
    'Tomate – Kraut- und Knollenfäule',
    'Tomate – Schimmelfleck',
    'Tomate – Septoria-Blattfleck',
    'Tomate – Spinnmilben',
    'Tomate – Zielfleckenkrankheit',
    'Tomate – Gelbkräuselkrankheit',
    'Tomate – Mosaikvirus',
    'Tomate – Gesund',
]

HEALTHY_KEYWORDS = ['Gesund']

DISEASE_INFO = {
    'Schorf': 'Pilzkrankheit – befällt Früchte und Blätter, führt zu dunklen Flecken und Ernteverlusten.',
    'Schwarzfäule': 'Pilzinfektion – zerstört Früchte innerhalb weniger Tage, schwer zu bekämpfen.',
    'Mehltau': 'Häufige Pilzkrankheit – weißer Belag auf Blättern, hemmt die Photosynthese.',
    'Rost': 'Pilzsporenbefall – orangefarbene Pusteln, breitet sich schnell über den Wind aus.',
    'Fäule': 'Schimmelpilz – zerstört Pflanzengewebe, oft ausgelöst durch Feuchtigkeit.',
    'Bakterienbrand': 'Bakterieninfektion – dunkle Flecken, kann ganze Felder befallen.',
    'Mosaikvirus': 'Virusinfektion – verursacht mosaikartige Verfärbungen, kein Heilmittel.',
    'Spinnmilben': 'Schädlingsbefall – winzige Milben saugen Pflanzensaft, kaum sichtbar.',
    'Huanglongbing': 'Unheilbare Bakterienkrankheit – bedroht weltweit die Zitrusproduktion.',
    'Grauer Blattfleck': 'Pilzkrankheit – grau-braune Blattflecken, reduziert den Ertrag stark.',
}

# ── Model loading ────────────────────────────────────────────────────────────
MODEL_PATH = "plant_disease_model.pth"
MODEL_URL = "https://huggingface.co/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification/resolve/main/model.pth"

@st.cache_resource
def load_model():
    """Load model – download from HF Hub if not cached locally."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🌿 KI-Modell wird geladen (einmalig ~14 MB)..."):
            r = requests.get(MODEL_URL, stream=True)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 38)
    state = torch.load(MODEL_PATH, map_location="cpu")
    # Handle both raw state_dict and wrapped checkpoint
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

# ── Image transform ──────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(model, img: Image.Image):
    tensor = TRANSFORM(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        top3 = torch.topk(probs, 3)
    results = []
    for idx, prob in zip(top3.indices.tolist(), top3.values.tolist()):
        results.append((CLASS_NAMES[idx], round(prob * 100, 1)))
    return results

def get_disease_hint(class_name):
    for keyword, info in DISEASE_INFO.items():
        if keyword.lower() in class_name.lower():
            return info
    return None

# ── Sample images ─────────────────────────────────────────────────────────────
SAMPLE_IMAGES = {
    "🍎 Apfel – Schorf (krank)": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Venturia_inaequalis_symptoms.jpg/320px-Venturia_inaequalis_symptoms.jpg",
    "🍅 Tomate – Krautfäule (krank)": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Potato_late_blight.jpg/320px-Potato_late_blight.jpg",
    "🌽 Mais – Rost (krank)": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Corn_leaf_rust.jpg/320px-Corn_leaf_rust.jpg",
    "🍇 Weintraube – Schwarzfäule (krank)": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Black_rot_grapes.jpg/320px-Black_rot_grapes.jpg",
}

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🌿 KI-<span class="accent">Pflanzendoktor</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Kann die KI erkennen was du nicht siehst?</div>', unsafe_allow_html=True)

st.markdown("""
<div class="context-box">
💡 <strong>Warum das wichtig ist:</strong> 40% der weltweiten Ernte gehen durch Pflanzenkrankheiten verloren.
In Afrika und Asien gibt es oft keinen Agrarberater weit und breit. Diese KI gibt jedem Bauern
einen Experten in der Hosentasche – und erkennt Krankheiten oft <em>bevor</em> sie mit bloßem Auge sichtbar sind.
</div>
""", unsafe_allow_html=True)

# Load model
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Modell konnte nicht geladen werden: {e}")
    model_loaded = False

st.divider()

# ── Tabs: Beispielbilder / Upload ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["📋 Beispielbilder", "📤 Eigenes Bild hochladen"])

with tab1:
    st.markdown("**Wähle ein Bild aus – was denkst du, was ist damit nicht stimmt?**")
    st.caption("Tipp: Schau genau hin. Kannst du das Problem erkennen? Dann lass die KI urteilen!")

    cols = st.columns(2)
    selected_sample = None

    for i, (label, url) in enumerate(SAMPLE_IMAGES.items()):
        with cols[i % 2]:
            try:
                resp = requests.get(url, timeout=5)
                img = Image.open(__import__('io').BytesIO(resp.content))
                st.image(img, caption=label.split("(")[0].strip(), use_container_width=True)
                if st.button(f"Diese analysieren", key=f"btn_{i}"):
                    selected_sample = img
                    st.session_state["selected_label"] = label
            except:
                st.warning(f"Bild konnte nicht geladen werden: {label}")

    if selected_sample and model_loaded:
        st.divider()
        st.markdown("### 🔍 KI-Analyse läuft...")
        with st.spinner("Muster werden erkannt..."):
            results = predict(model, selected_sample)

        top_class, top_conf = results[0]
        is_healthy = any(kw in top_class for kw in HEALTHY_KEYWORDS)
        box_class = "healthy-box" if is_healthy else "sick-box"
        status_emoji = "✅" if is_healthy else "🔴"

        st.markdown(f"""
        <div class="{box_class}">
            <div class="result-label">{status_emoji} KI-Diagnose</div>
            <div class="result-disease">{top_class}</div>
            <div class="result-plant">Konfidenz: <strong>{top_conf}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

        hint = get_disease_hint(top_class)
        if hint:
            st.info(f"ℹ️ **Was ist das?** {hint}")

        if len(results) > 1:
            with st.expander("Weitere Möglichkeiten laut KI"):
                for cls, conf in results[1:]:
                    st.markdown(f"- **{cls}** – {conf}%")

with tab2:
    st.markdown("**Lade ein Bild eines Pflanzenblatts hoch:**")
    uploaded = st.file_uploader(
        "Bild auswählen (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded and model_loaded:
        img = Image.open(uploaded)
        col_img, col_result = st.columns([1, 1])

        with col_img:
            st.image(img, caption="Dein Bild", use_container_width=True)

        with col_result:
            with st.spinner("🔍 KI analysiert..."):
                results = predict(model, img)

            top_class, top_conf = results[0]
            is_healthy = any(kw in top_class for kw in HEALTHY_KEYWORDS)
            status_emoji = "✅" if is_healthy else "🔴"
            box_class = "healthy-box" if is_healthy else "sick-box"

            st.markdown(f"""
            <div class="{box_class}">
                <div class="result-label">{status_emoji} KI-Diagnose</div>
                <div class="result-disease">{top_class}</div>
                <div class="result-plant">Konfidenz: <strong>{top_conf}%</strong></div>
            </div>
            """, unsafe_allow_html=True)

            hint = get_disease_hint(top_class)
            if hint:
                st.info(f"ℹ️ {hint}")

            if len(results) > 1:
                with st.expander("Weitere Möglichkeiten"):
                    for cls, conf in results[1:]:
                        st.markdown(f"- **{cls}** – {conf}%")

st.divider()
st.caption("🌍 KI-Tag an Schulen · Station: KI verändert die Welt · Modell trainiert auf PlantVillage (54.000+ Bilder)")
