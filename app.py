import streamlit as st
import json
import requests
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

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

# ── Paths & URLs ──────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MODEL_PATH  = BASE_DIR / "best_model.pth"
CLASSES_PATH= BASE_DIR / "class_indices.json"
HF_BASE     = "https://huggingface.co/sch-leo/tomato-disease-detection/resolve/main"
MODEL_URL   = f"{HF_BASE}/best_model.pth"
CLASSES_URL = f"{HF_BASE}/class_indices.json"

def download_file(url, dest):
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

if not CLASSES_PATH.exists():
    with st.spinner("Klassen werden geladen..."):
        download_file(CLASSES_URL, CLASSES_PATH)

if not MODEL_PATH.exists():
    with st.spinner("KI-Modell wird heruntergeladen..."):
        download_file(MODEL_URL, MODEL_PATH)

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Modell wird geladen...")
def load_model():
    with open(CLASSES_PATH) as f:
        idx_to_class = json.load(f)
    n_classes = len(idx_to_class)
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, n_classes)
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, idx_to_class

TRANSFORM = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(model, idx_to_class, img: Image.Image):
    tensor = TRANSFORM(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    top3 = torch.topk(probs, 3)
    return [(idx_to_class[str(i.item())], round(v.item()*100, 1))
            for i, v in zip(top3.indices, top3.values)]

# ── Labels ────────────────────────────────────────────────────────────────────
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
    "Bakterienbrand":        "Bakterieninfektion - dunkle Flecken auf Blättern und Früchten. Breitet sich bei Feuchtigkeit schnell aus.",
    "Frühfäule":             "Pilzkrankheit - konzentrische braune Ringe auf Blättern, ähnlich wie Schießscheiben.",
    "Kraut- und Knollenfäule":"Gefährlichster Pflanzenpilz - löste die Irische Hungersnot 1845 aus. Zerstört Felder in Tagen.",
    "Schimmelfleck":         "Pilzbefall - gelbe Flecken oben, grau-brauner Schimmel unten. Gedeiht bei Feuchtigkeit.",
    "Septoria-Blattfleck":   "Pilzkrankheit - viele kleine Flecken mit hellem Zentrum. Im Frühstadium schwer zu erkennen.",
    "Spinnmilbenbefall":     "Winzige Milben saugen Pflanzensaft - kaum sichtbar, Blätter werden bronzefarben.",
    "Zielfleckenkrankheit":  "Pilzkrankheit - braune Flecken mit konzentrischen Ringen.",
    "Gelbkräuselkrankheit":  "Virusinfektion durch Weißfliegen - Blätter kräuseln sich gelb. Kein Heilmittel.",
    "Mosaikvirus":           "Virusinfektion - mosaikartige Verfärbungen. Kein Heilmittel, Pflanze muss entfernt werden.",
}

def translate_label(raw):
    entry = LABEL_DE.get(raw)
    if entry:
        plant, disease = entry
        return plant, disease, "Gesund" in disease
    return "Tomate", raw.replace("_", " "), "healthy" in raw.lower()

def get_info(disease_de):
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
    box   = "healthy-box" if is_healthy else "sick-box"
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
    img_dir = BASE_DIR / "images"
    if not img_dir.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    files = [f for f in img_dir.iterdir() if f.suffix in exts]
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
In vielen Ländern gibt es kaum Agrarberater. Diese KI wurde auf <strong>16.000 Tomatenblatt-Fotos</strong>
trainiert und erreicht eine Genauigkeit von <strong>99,5%</strong> – besser als die meisten Experten.
</div>
""", unsafe_allow_html=True)
st.divider()

model, idx_to_class = load_model()

tab1, tab2 = st.tabs(["📋 Beispielbilder", "📤 Eigenes Bild hochladen"])

with tab1:
    st.markdown("**Schau dir das Bild an - was glaubst du, was ist mit der Pflanze los?**")
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
                    st.image(img, width='stretch')
                    if st.button("🔍 KI fragen", key=f"s{i}"):
                        with st.spinner("KI analysiert..."):
                            results = predict(model, idx_to_class, img)
                        show_result(results)
                except Exception:
                    st.warning(f"Fehler: {fpath.name}")

with tab2:
    st.markdown("**Lade ein Tomatenblatt-Bild hoch:**")
    uploaded = st.file_uploader("Bild (JPG/PNG)", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if uploaded:
        img = Image.open(uploaded)
        c1, c2 = st.columns(2)
        with c1:
            st.image(img, caption="Dein Bild", width='stretch')
        with c2:
            with st.spinner("🔍 KI analysiert..."):
                results = predict(model, idx_to_class, img)
            show_result(results)

st.divider()
st.caption("🌍 KI-Tag an Schulen · KI als Pflanzendoktor")