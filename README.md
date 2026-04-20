# 🌿 KI-Pflanzendoktor

Interaktive Streamlit-App für den KI-Tag an Schulen.  
Schüler*innen schätzen selbst was mit einem Pflanzenblatt nicht stimmt – dann analysiert die KI dasselbe Bild.


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
