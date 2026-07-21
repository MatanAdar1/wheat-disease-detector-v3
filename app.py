@st.cache_resource
def load_wheat_model():
    if not os.path.exists(MODEL_PATH):
        try:
            # שימוש בזיהוי ה-ID הישיר ובמנגנון fuzzy לעקיפת חסימות דרייב
            gdown.download(id=FILE_ID, output=MODEL_PATH, quiet=False, fuzzy=True)
        except Exception as e:
            st.error(f"שגיאה בהורדת המודל מ-Google Drive: {e}")
            return None, None
            
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
