import streamlit as st
import requests
import io
from PIL import Image

st.set_page_config(page_title="KI-Pflanzendoktor", page_icon="🌿", layout="centered")

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

HF_MODEL   = "ombhojane/healthyPlantsModel"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

def query_hf(image: Image.Image):
    token = st.secrets.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    buf.seek(0)
    resp = requests.post(HF_API_URL, headers=headers, data=buf.read(), timeout=30)
    if resp.status_code == 503:
        return None
    resp.raise_for_status()
    return resp.json()

LABEL_DE = {
    "Apple___Apple_scab":("Apfel","Schorf"),
    "Apple___Black_rot":("Apfel","Schwarzfäule"),
    "Apple___Cedar_apple_rust":("Apfel","Rostfleckenkrankheit"),
    "Apple___healthy":("Apfel","Gesund ✅"),
    "Blueberry___healthy":("Heidelbeere","Gesund ✅"),
    "Cherry_(including_sour)___Powdery_mildew":("Kirsche","Echter Mehltau"),
    "Cherry_(including_sour)___healthy":("Kirsche","Gesund ✅"),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":("Mais","Grauer Blattfleck"),
    "Corn_(maize)___Common_rust_":("Mais","Gemeiner Rost"),
    "Corn_(maize)___Northern_Leaf_Blight":("Mais","Nördliche Blattfleckenkrankheit"),
    "Corn_(maize)___healthy":("Mais","Gesund ✅"),
    "Grape___Black_rot":("Weintraube","Schwarzfäule"),
    "Grape___Esca_(Black_Measles)":("Weintraube","Schwarzer Masern"),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":("Weintraube","Blattfleckenkrankheit"),
    "Grape___healthy":("Weintraube","Gesund ✅"),
    "Orange___Haunglongbing_(Citrus_greening)":("Orange","Huanglongbing"),
    "Peach___Bacterial_spot":("Pfirsich","Bakterienbrand"),
    "Peach___healthy":("Pfirsich","Gesund ✅"),
    "Pepper,_bell___Bacterial_spot":("Paprika","Bakterienbrand"),
    "Pepper,_bell___healthy":("Paprika","Gesund ✅"),
    "Potato___Early_blight":("Kartoffel","Frühfäule"),
    "Potato___Late_blight":("Kartoffel","Kraut- und Knollenfäule"),
    "Potato___healthy":("Kartoffel","Gesund ✅"),
    "Raspberry___healthy":("Himbeere","Gesund ✅"),
    "Soybean___healthy":("Sojabohne","Gesund ✅"),
    "Squash___Powdery_mildew":("Kürbis","Echter Mehltau"),
    "Strawberry___Leaf_scorch":("Erdbeere","Blattdürre"),
    "Strawberry___healthy":("Erdbeere","Gesund ✅"),
    "Tomato___Bacterial_spot":("Tomate","Bakterienbrand"),
    "Tomato___Early_blight":("Tomate","Frühfäule"),
    "Tomato___Late_blight":("Tomate","Kraut- und Knollenfäule"),
    "Tomato___Leaf_Mold":("Tomate","Schimmelfleck"),
    "Tomato___Septoria_leaf_spot":("Tomate","Septoria-Blattfleck"),
    "Tomato___Spider_mites Two-spotted_spider_mite":("Tomate","Spinnmilbenbefall"),
    "Tomato___Target_Spot":("Tomate","Zielfleckenkrankheit"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":("Tomate","Gelbkräuselkrankheit"),
    "Tomato___Tomato_mosaic_virus":("Tomate","Mosaikvirus"),
    "Tomato___healthy":("Tomate","Gesund ✅"),
}

DISEASE_INFO = {
    "Schorf":"Pilzkrankheit – dunkle Flecken auf Früchten und Blättern, führt zu Ernteverlusten.",
    "Schwarzfäule":"Pilzinfektion – zerstört Früchte innerhalb weniger Tage.",
    "Mehltau":"Weißer Belag auf Blättern hemmt Photosynthese – häufig bei Feuchtigkeit.",
    "Rost":"Pilzsporenbefall – orangefarbene Pusteln, breitet sich über Wind aus.",
    "Fäule":"Schimmelpilz zerstört Pflanzengewebe, oft durch Feuchtigkeit ausgelöst.",
    "Bakterienbrand":"Bakterieninfektion – dunkle Flecken, kann ganze Felder befallen.",
    "Mosaikvirus":"Virusinfektion – mosaikartige Verfärbungen, kein Heilmittel möglich.",
    "Spinnmilben":"Winzige Milben saugen Pflanzensaft – kaum mit bloßem Auge sichtbar.",
    "Huanglongbing":"Unheilbare Bakterienkrankheit – bedroht weltweit die Zitrusproduktion.",
    "Grauer Blattfleck":"Pilzkrankheit – grau-braune Flecken, reduziert Ertrag stark.",
    "Blattfleckenkrankheit":"Pilzbefall – braune Flecken mit gelbem Rand.",
    "Blattdürre":"Pilzinfektion – Blätter trocknen von den Rändern her ein.",
    "Gelbkräuselkrankheit":"Virusinfektion durch Weißfliegen – Blätter kräuseln sich gelb.",
    "Frühfäule":"Pilzkrankheit – konzentrische braune Ringe auf Blättern.",
    "Kraut- und Knollenfäule":"Gefährlichster Pflanzenpilz – löste die Irische Hungersnot 1845 aus!",
    "Septoria-Blattfleck":"Pilzkrankheit – viele kleine braune Flecken mit dunklem Rand.",
}

def translate_label(raw):
    entry = LABEL_DE.get(raw)
    if entry:
        plant, disease = entry
        return plant, disease, "Gesund" in disease
    parts = raw.replace("_"," ").split("___")
    if len(parts)==2:
        return parts[0], parts[1], "healthy" in raw.lower()
    return raw, "", False

def get_info(disease_de):
    for kw, info in DISEASE_INFO.items():
        if kw.lower() in disease_de.lower():
            return info
    return None

def show_result(results):
    if not results or not isinstance(results, list):
        st.error("Unerwartete Antwort vom Modell.")
        return
    top = results[0]
    plant, disease, is_healthy = translate_label(top["label"])
    conf = round(top["score"]*100, 1)
    box  = "healthy-box" if is_healthy else "sick-box"
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
    if len(results)>1:
        with st.expander("Weitere Möglichkeiten"):
            for r in results[1:4]:
                p,d,h = translate_label(r["label"])
                c = round(r["score"]*100,1)
                st.markdown(f"- **{d}** ({p}) – {c}%")

SAMPLES = {
    "Tomate – Kraut- und Knollenfäule": {
        "url":"https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Potato_late_blight.jpg/320px-Potato_late_blight.jpg",
        "hint":"Schau auf die Blattränder – siehst du etwas Auffälliges?",
    },
    "Mais – Gemeiner Rost": {
        "url":"https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Corn_leaf_rust.jpg/320px-Corn_leaf_rust.jpg",
        "hint":"Welche Farbe haben die Flecken auf dem Blatt?",
    },
    "Weintraube – Schwarzfäule": {
        "url":"https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Blackrot_lesions_on_grape_leaf.jpg/320px-Blackrot_lesions_on_grape_leaf.jpg",
        "hint":"Sind die Flecken gleichmäßig oder haben sie ein Muster?",
    },
    "Apfel – Schorf": {
        "url":"https://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/Venturia_inaequalis_on_Bramley%27s_Seedling.jpg/320px-Venturia_inaequalis_on_Bramley%27s_Seedling.jpg",
        "hint":"Was fällt dir an der Oberfläche auf?",
    },
}

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🌿 KI-<span class="accent">Pflanzendoktor</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Kannst du erkennen was die KI sieht?</div>', unsafe_allow_html=True)
st.markdown("""
<div class="context-box">
💡 <strong>Warum das wichtig ist:</strong> 40% der weltweiten Ernte gehen durch Pflanzenkrankheiten verloren.
In vielen Ländern gibt es kaum Agrarberater. Diese KI gibt jedem Bauern einen Experten in der Hosentasche –
trainiert auf <strong>54.000 Blattfotos</strong>, erkennt <strong>38 Krankheiten</strong> in Sekunden.
</div>
""", unsafe_allow_html=True)
st.divider()

tab1, tab2 = st.tabs(["📋 Beispielbilder", "📤 Eigenes Bild hochladen"])

with tab1:
    st.markdown("**Schau dir das Bild an – was glaubst du, was ist mit der Pflanze los?**")
    st.caption("Erst selbst überlegen, dann die KI fragen!")
    cols = st.columns(2)
    for i, (name, data) in enumerate(SAMPLES.items()):
        with cols[i%2]:
            try:
                r = requests.get(data["url"], timeout=8)
                img = Image.open(io.BytesIO(r.content))
                st.image(img, use_container_width=True)
                st.caption(f"💭 *{data['hint']}*")
                if st.button("KI fragen →", key=f"s{i}"):
                    with st.spinner("🔍 KI analysiert..."):
                        res = query_hf(img)
                    if res is None:
                        st.warning("Modell startet gerade – 20 Sek. warten und nochmal versuchen.")
                    else:
                        st.markdown(f"**Lösung: {name}**")
                        show_result(res)
            except:
                st.warning("Bild konnte nicht geladen werden.")

with tab2:
    st.markdown("**Lade ein Bild eines Pflanzenblatts hoch:**")
    uploaded = st.file_uploader("Bild (JPG/PNG)", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if uploaded:
        img = Image.open(uploaded)
        c1, c2 = st.columns(2)
        with c1:
            st.image(img, caption="Dein Bild", use_container_width=True)
        with c2:
            with st.spinner("🔍 KI analysiert..."):
                res = query_hf(img)
            if res is None:
                st.warning("Modell startet – kurz warten und Seite neu laden.")
            else:
                show_result(res)

st.divider()
st.caption("🌍 KI-Tag an Schulen · Modell: PlantVillage MobileNetV2 · 54.000 Bilder · 38 Klassen")
