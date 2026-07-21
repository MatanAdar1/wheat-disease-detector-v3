import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
import base64
import io
import gdown
from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# ─────────────────────────────────────────
# Model Constants & Automatic Download
# ─────────────────────────────────────────
MODEL_PATH = 'best_resnet18_wheat.pt'
FILE_ID = '14X2jR9P6LSoNl4jS_1fX5j2G-Npx8Gk0'  # ה-ID של קובץ המודל בגוגל דרייב
CONFIDENCE_THRESHOLD = 0.25

@st.cache_resource
def load_wheat_model():
    # אם הקובץ לא קיים בשרת/במחשב המקומי - נוריד אותו אוטומטית
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏳ מוריד את קובץ המודל (חד פעמי)..."):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
            
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        lbl = checkpoint.get("classes", list(DISEASE_INFO.keys()))
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, len(lbl))
        m.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        m.eval()
        return m, lbl
    except Exception as e:
        st.error(f"שגיאה בטעינת המודל: {e}")
        return None, None