# 🍅 KI-Pflanzendoktor

Interaktive App für den KI-Tag an Schulen – Schüler*innen schätzen selbst was mit einem Tomatenblatt nicht stimmt, dann analysiert die KI dasselbe Bild.

## Voraussetzungen

- Python 3.10+
- Dateien `best_model.h5` + `class_indices.json` im App-Ordner  
  → Download: https://huggingface.co/abdullahzunorain/tomato_leaf_disease_det_model_v1

## Setup

```bash
pip install -r requirements.txt
```

## Bilder hinzufügen

Tomatenblatt-Fotos (JPG/PNG) in den `images/` Ordner legen.  
Empfehlung: je 1–2 Bilder pro Krankheit aus dem PlantVillage-Datensatz.

## Starten

```bash
streamlit run app.py
```

## Projektstruktur

```
├── app.py
├── requirements.txt
├── best_model.h5
├── class_indices.json
└── images/
    └── *.jpg
```