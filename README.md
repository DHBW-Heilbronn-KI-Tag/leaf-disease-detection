# 🌿 KI-Pflanzendoktor

Interaktive Streamlit-App für den KI-Tag an Schulen.  
Schüler*innen schätzen selbst was mit einem Pflanzenblatt nicht stimmt – dann analysiert die KI dasselbe Bild.

## 🚀 Deployment auf Streamlit Cloud (kostenlos)

### Schritt 1 – GitHub Repository anlegen
1. Gehe auf [github.com](https://github.com) und erstelle ein neues Repository (z.B. `ki-pflanzendoktor`)
2. Lade die Dateien hoch:
   - `app.py`
   - `requirements.txt`
   - `README.md`

### Schritt 2 – Streamlit Cloud verbinden
1. Gehe auf [share.streamlit.io](https://share.streamlit.io)
2. Mit GitHub-Account anmelden
3. „New app" → Repository auswählen → `app.py` als Main file
4. „Deploy" klicken

### Schritt 3 – Fertig!
Die App ist unter einer öffentlichen URL erreichbar, z.B.:  
`https://dein-name-ki-pflanzendoktor.streamlit.app`

Diese URL einfach im Browser auf dem Stations-Laptop öffnen – kein Python, kein Install nötig.

## 📋 Was die App macht
- Beispielbilder kranker Pflanzen zum Analysieren (kein Upload nötig)
- Eigene Bilder hochladen möglich
- KI erkennt Pflanzenkrankheit + zeigt Konfidenz
- Kurze Erklärung was die Krankheit bedeutet
- Kontext: warum das gesellschaftlich relevant ist

## 🧠 Modell
- MobileNetV2 trainiert auf PlantVillage Dataset (54.000+ Bilder, 38 Klassen)
- Wird beim ersten Start automatisch geladen (~14 MB)
- 14 Pflanzenarten, 26 Krankheiten + gesunde Varianten

## 📁 Projektstruktur
```
ki-pflanzendoktor/
├── app.py           # Streamlit App
├── requirements.txt # Python-Abhängigkeiten
└── README.md        # Diese Datei
```
