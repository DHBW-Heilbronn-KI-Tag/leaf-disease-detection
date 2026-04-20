import streamlit as st
import numpy as np
import json
import io
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="KI-Pflanzendoktor", page_icon="🍅", layout="centered")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');
  html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
  .main-title{font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;line-height:1.1;margin-bottom:.2rem;}
  .accent{color:#22c55e;}
  .subtitle{color:#888;font-size:1rem;margin-bottom:1.5rem;font-weight:300;}
  .healthy-box{background:linear-gradient(135deg,#0f2017,#0a1a0f);border:1px solid #22c55e;border-radius:16px;padding:1.4rem;margin:1rem 0;}
  .sick-box{background:linear-gradient(135deg,#1f0f0f,#150808);border:1px solid #ef444466;border-radius:16px;padding:1.4rem;margin:1rem 0;}
  .result-label{font-size:.72rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:#22c55e;margin-bottom:.25rem;}
  .result-disease{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;color:#f0f0f0;margin-bottom:.15rem;}
  .result-conf{font-size:.9rem;color:#aaa;}
  .context-box{background:rgba(34,197,94,.05);border-left:3px solid #22c55e;border-radius:0 8px 8px 0;padding:.8rem 1rem;margin:1rem 0;font-size:.88rem;color:#aaa;line-height:1.5;}
</style>
""", unsafe_allow_html=True)

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "best_model.h5"
CLASSES_PATH = Path(__file__).parent / "class_indices.json"

@st.cache_resource(show_spinner="🌿 Modell wird geladen...")
def load_model():
    import tensorflow as tf
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    model = tf.keras.models.load_model(str(MODEL_PATH))
    with open(CLASSES_PATH) as f:
        class_indices = json.load(f)
    # JSON has string keys {"0": "Tomato_...", "1": ...}
    # model outputs integer indices → map int → classname directly
    idx_to_class = {int(k): v for k, v in class_indices.items()}
    return model, idx_to_class

def get_input_size(model):
    """Auto-detect required input size from model."""
    try:
        shape = model.input_shape  # e.g. (None, 150, 150, 3)
        return shape[1], shape[2]
    except Exception:
        return 224, 224

def predict(model, idx_to_class, img: Image.Image):
    h, w = get_input_size(model)
    img_resized = img.convert("RGB").resize((w, h))
    arr = np.array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    top3_idx = np.argsort(preds)[::-1][:3]
    return [(idx_to_class.get(i, str(i)), round(float(preds[i]) * 100, 1)) for i in top3_idx]

# ── German labels ─────────────────────────────────────────────────────────────
# Maps raw class names → German display name
LABEL_DE = {
    "Tomato_Bacterial_spot":                        ("Tomate", "Bakterienbrand"),
    "Tomato_Early_blight":                          ("Tomate", "Frühfäule"),
    "Tomato_Late_blight":                           ("Tomate", "Kraut- und Knollenfäule"),
    "Tomato_Leaf_Mold":                             ("Tomate", "Schimmelfleck"),
    "Tomato_Septoria_leaf_spot":                    ("Tomate", "Septoria-Blattfleck"),
    "Tomato_Spider_mites_Two_spotted_spider_mite":  ("Tomate", "Spinnmilbenbefall"),
    "Tomato__Target_Spot":                          ("Tomate", "Zielfleckenkrankheit"),
    "Tomato__Tomato_YellowLeaf__Curl_Virus":        ("Tomate", "Gelbkräuselkrankheit"),
    "Tomato__Tomato_mosaic_virus":                  ("Tomate", "Mosaikvirus"),
    "Tomato_healthy":                               ("Tomate", "Gesund ✅"),
}

DISEASE_INFO = {
    "Bakterienbrand":        "Bakterieninfektion – dunkle, wassergetränkte Flecken auf Blättern und Früchten. Breitet sich bei Feuchtigkeit schnell aus.",
    "Frühfäule":             "Pilzkrankheit – konzentrische braune Ringe auf Blättern, ähnlich wie Schießscheiben. Befällt ältere Blätter zuerst.",
    "Kraut- und Knollenfäule":"Gefährlichster Pflanzenpilz der Welt – löste die Irische Hungersnot 1845 aus. Zerstört ganze Felder in wenigen Tagen.",
    "Schimmelfleck":         "Pilzbefall – gelbe Flecken oben, grau-brauner Schimmel unten. Gedeiht bei hoher Luftfeuchtigkeit.",
    "Septoria-Blattfleck":   "Pilzkrankheit – viele kleine kreisrunde Flecken mit hellem Zentrum und dunklem Rand. Schwer zu erkennen im Frühstadium.",
    "Spinnmilbenbefall":     "Winzige Milben (kaum sichtbar!) saugen Pflanzensaft – Blätter werden bronzefarben und sterben ab.",
    "Zielfleckenkrankheit":  "Pilzkrankheit – braune Flecken mit konzentrischen Ringen. Befällt alle oberirdischen Pflanzenteile.",
    "Gelbkräuselkrankheit":  "Virusinfektion durch Weißfliegen übertragen – Blätter kräuseln sich und verfärben sich gelb. Kein Heilmittel.",
    "Mosaikvirus":           "Virusinfektion – mosaikartige hell-dunkel Verfärbungen. Kein Heilmittel, Pflanze muss entfernt werden.",
}

def translate_label(raw: str):
    """Return (plant_de, disease_de, is_healthy)."""
    entry = LABEL_DE.get(raw)
    if entry:
        plant, disease = entry
        return plant, disease, "Gesund" in disease
    # Fallback: clean up raw name
    clean = raw.replace("___", " – ").replace("_", " ")
    return "Pflanze", clean, "healthy" in raw.lower()

def get_info(disease_de: str):
    for kw, info in DISEASE_INFO.items():
        if kw.lower() in disease_de.lower():
            return info
    return None

def show_result(results):
    if not results:
        st.error("Keine Ergebnisse.")
        return
    label, conf = results[0]
    plant, disease, is_healthy = translate_label(label)
    box = "healthy-box" if is_healthy else "sick-box"
    emoji = "✅" if is_healthy else "🔴"
    st.markdown(f"""
    <div class="{box}">
        <div class="result-label">{emoji} KI-Diagnose</div>
        <div class="result-disease">{disease}</div>
        <div class="result-conf">Pflanze: <strong>{plant}</strong> &nbsp;·&nbsp; Konfidenz: <strong>{conf}%</strong></div>
    </div>""", unsafe_allow_html=True)
    info = get_info(disease)
    if info and not is_healthy:
        st.info(f"ℹ️ **Was ist das?** {info}")
    if len(results) > 1:
        with st.expander("Weitere Möglichkeiten"):
            for lbl, c in results[1:]:
                _, d, _ = translate_label(lbl)
                st.markdown(f"- **{d}** – {c}%")

def load_local_images():
    import random
    img_dir = Path(__file__).parent / "images"
    if not img_dir.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    files = [f for f in img_dir.iterdir() if f.suffix in exts]
    # Shuffle once per session, stable across reruns
    if "image_order" not in st.session_state:
        random.shuffle(files)
        st.session_state["image_order"] = [str(f) for f in files]
    return [Path(f) for f in st.session_state["image_order"][:20]]

# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🍅 KI-<span class="accent">Pflanzendoktor</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Kannst du erkennen was die KI sieht?</div>', unsafe_allow_html=True)
st.markdown("""
<div class="context-box">
💡 <strong>Warum das wichtig ist:</strong> 40% der weltweiten Ernte gehen durch Pflanzenkrankheiten verloren.
In vielen Ländern gibt es kaum Agrarberater. Diese KI wurde auf <strong>tausenden Tomatenblatt-Fotos</strong>
trainiert und erkennt Krankheiten oft <em>bevor</em> sie mit bloßem Auge sichtbar sind.
</div>
""", unsafe_allow_html=True)
st.divider()

# Check model files exist
if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.error("⚠️ Modelldateien fehlen! Bitte `best_model.h5` und `class_indices.json` in den App-Ordner legen.")
    st.info("Download: https://huggingface.co/abdullahzunorain/tomato_leaf_disease_det_model_v1")
    st.stop()

# Load model
try:
    model, idx_to_class = load_model()
except Exception as e:
    st.error(f"Modell konnte nicht geladen werden: {e}")
    st.stop()

tab1, tab2 = st.tabs(["📋 Beispielbilder", "📤 Eigenes Bild hochladen"])

with tab1:
    st.markdown("**Schau dir das Bild an – was glaubst du, was ist mit der Pflanze los?**")
    st.caption("Erst selbst überlegen, dann die KI fragen!")
    image_files = load_local_images()
    if not image_files:
        st.warning("Keine Bilder im `images/` Ordner gefunden.")
    else:
        cols = st.columns(2)
        for i, fpath in enumerate(image_files):
            with cols[i % 2]:
                try:
                    img = Image.open(fpath)
                    st.image(img, width="stretch")
                    if st.button("🔍 KI fragen", key=f"s{i}"):
                        with st.spinner("KI analysiert..."):
                            results = predict(model, idx_to_class, img)
                        show_result(results)
                except Exception:
                    st.warning(f"Fehler: {fpath.name}")

with tab2:
    st.markdown("**Lade ein Tomatenblatts-Bild hoch:**")
    uploaded = st.file_uploader("Bild (JPG/PNG)", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if uploaded:
        img = Image.open(uploaded)
        c1, c2 = st.columns(2)
        with c1:
            st.image(img, caption="Dein Bild", width="stretch")
        with c2:
            with st.spinner("🔍 KI analysiert..."):
                results = predict(model, idx_to_class, img)
            show_result(results)

st.divider()
st.caption("🌍 KI-Tag an Schulen · Modell trainiert auf PlantVillage Tomaten-Datensatz · läuft komplett lokal")