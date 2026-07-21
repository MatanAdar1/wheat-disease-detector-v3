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
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# ─────────────────────────────────────────
# Page Configuration (חייב להיות השורה הראשונה בקוד)
# ─────────────────────────────────────────
st.set_page_config(page_title="מערכת ניטור וחיישני חיטה v3 🌾", page_icon="🌾", layout="wide")

# ─────────────────────────────────────────
# MongoDB Connection (עם Timeout של 5 שניות למניעת תקיעה)
# ─────────────────────────────────────────
@st.cache_resource
def get_db():
    if "MONGODB_URI" not in st.secrets:
        st.error("❌ חסר משתנה MONGODB_URI ב-Secrets של Streamlit!")
        return None
    try:
        uri = st.secrets["MONGODB_URI"]
        # הגבלת זמן המתנה ל-5000 מילי-שניות (5 שניות) למניעת מסך לבן
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client["wheat_disease_db"]  # מחובר למסד הנתונים הקיים ב-Atlas
    except Exception as e:
        st.error(f"❌ שגיאה בחיבור ל-MongoDB: {e}")
        st.info("💡 ודא שב-MongoDB Atlas מוגדר Network Access -> Allow Access From Anywhere (0.0.0.0/0)")
        return None

db = get_db()
plants_col = db["plants"] if db is not None else None
readings_col = db["sensor_readings"] if db is not None else None
diagnoses_col = db["diagnoses"] if db is not None else None

# ─────────────────────────────────────────
# Navigation State
# ─────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "current_plant_idx" not in st.session_state:
    st.session_state.current_plant_idx = 0

# ─────────────────────────────────────────
# CSS Styling (יישור מימין לשמאל RTL)
# ─────────────────────────────────────────
st.markdown("""
<style>
.stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p, label, [data-testid="stWidgetLabel"] {
    text-align: right !important; direction: rtl !important;
}
.stButton>button, .stSelectbox, .stTextArea {
    direction: rtl !important; text-align: right !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    text-align: right !important; direction: rtl !important;
}
[data-testid="stDataFrame"] { direction: rtl !important; }
.custom-card {
    background: #ffffff; padding: 24px; border-radius: 12px;
    border-right: 6px solid #2e7d32; margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.04);
}
.home-box {
    background: #ffffff; padding: 40px 30px; border-radius: 16px;
    border: 1px solid #eaeaea; text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.03);
    transition: transform .3s ease, box-shadow .3s ease, border-color .3s ease;
    height: 100%;
}
.home-box:hover {
    transform: translateY(-6px);
    box-shadow: 0 12px 30px rgba(46,125,50,.12);
    border-color: #2e7d32;
}
.main-title {
    font-size: 3rem !important; font-weight: 800 !important;
    background: linear-gradient(45deg, #2e7d32, #1565c0);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center !important; margin-bottom: 10px !important;
}
.subtitle {
    font-size: 1.25rem !important; color: #666 !important;
    text-align: center !important; margin-bottom: 40px !important;
}
.db-badge {
    display: inline-block; background: #e8f5e9; color: #2e7d32;
    border-radius: 20px; padding: 4px 14px; font-size: 0.85rem;
    font-weight: 600; margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Model Constants & Automatic Download
# ─────────────────────────────────────────
MODEL_PATH = 'best_resnet18_wheat.pt'
FILE_ID = '14X2jR9P6LSoNl4jS_1fX5j2G-Npx8Gk0'
CONFIDENCE_THRESHOLD = 0.25

DISEASE_INFO = {
    "BlackPoint": {
        "heb": "חוד שחור (Black Point)",
        "desc": "מחלה הנגרמת על ידי קומפלקס פטריות בשלבי הבשלת הגרעין. מתאפיינת בהשחרה של קצה הגרעין.",
        "tip": "מומלץ להפחית את משטר ההשקיה בשלבי ההבשלה."
    },
    "FusariumFootRot": {
        "heb": "ריקבון בסיס הקנה (Fusarium)",
        "desc": "מחלה פטרייתית קרקעית התוקפת את השורשים ובסיס הקנה. חוסמת צינורות הובלה.",
        "tip": "יש ליישם מחזור זרעים קפדני. להימנע מהשקיית יתר."
    },
    "HealthyLeaf": {
        "heb": "עלה בריא (Healthy)",
        "desc": "העלה מציג חיוניות גבוהה, צבע ירוק אחיד ושטח פנים נקי.",
        "tip": "מצב מצוין! להמשיך במשטר הטיפוח הנוכחי."
    },
    "LeafBlight": {
        "heb": "קמלת עלים (Leaf Blight)",
        "desc": "מחלה פטרייתית המתבטאת בכתמים חומים-אפרפרים על העלים.",
        "tip": "מומלץ לרסס בקוטלי פטריות עם זיהוי הסימנים הראשונים."
    },
    "WheatBlast": {
        "heb": "פיריקורליית החיטה (Wheat Blast)",
        "desc": "מחלה פטרייתית הרסנית — השיבולת הופכת ללבנה ויבשה תוך ימים.",
        "tip": "מצב חירום חקלאי. לבודד את האזור הנגוע מיד ולרסס בקוטלי פטריות."
    }
}

@st.cache_resource
def load_wheat_model():
    if not os.path.exists(MODEL_PATH):
        try:
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=True)
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
        st.error(f"שגיאה שטעינת המודל: {e}")
        return None, None

model, labels = load_wheat_model()

transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def run_model(image):
    if model is None:
        return None, None
    with torch.no_grad():
        out = model(transform(image).unsqueeze(0))
        prob = torch.nn.functional.softmax(out[0], dim=0)
        conf, pred = torch.max(prob, 0)
    if conf.item() < CONFIDENCE_THRESHOLD:
        return None, conf.item()
    return labels[pred.item()], conf.item()

# ─────────────────────────────────────────
# MongoDB Helper Functions
# ─────────────────────────────────────────
def process_and_save_sensor_file(df: pd.DataFrame):
    if db is None:
        st.error("אין חיבור פעיל ל-MongoDB.")
        return
        
    plants_col.delete_many({})
    readings_col.delete_many({})
    
    plant_ids = set()
    for col in df.columns:
        if col != 'Timestamp':
            parts = col.split(' ')
            if parts[0].isdigit():
                plant_ids.add(int(parts[0]))
    
    plant_ids = sorted(list(plant_ids))
    latest_row = df.iloc[-1]
    
    plants_docs = []
    for pid in plant_ids:
        m_col = f"{pid} moisture"
        t_col = f"{pid} temperature"
        
        latest_m = float(latest_row[m_col]) if m_col in df.columns else 0.0
        latest_t = float(latest_row[t_col]) if t_col in df.columns else 0.0
        
        treatment = "Drought" if latest_m < 0.1 else "Control"
        
        plants_docs.append({
            "id": pid,
            "name": f"צמח חיטה #{pid}",
            "latest_moisture": round(latest_m, 4),
            "latest_temperature": round(latest_t, 2),
            "#Treatment": treatment
        })
    
    if plants_docs:
        plants_col.insert_many(plants_docs)
        
    readings_docs = []
    for idx, row in df.iterrows():
        ts = str(row['Timestamp'])
        for pid in plant_ids:
            m_col = f"{pid} moisture"
            t_col = f"{pid} temperature"
            if m_col in df.columns and t_col in df.columns:
                readings_docs.append({
                    "plant_id": pid,
                    "timestamp": ts,
                    "moisture": float(row[m_col]),
                    "temperature": float(row[t_col])
                })
    
    if readings_docs:
        readings_col.insert_many(readings_docs)

def get_all_plants():
    if plants_col is None: return []
    return list(plants_col.find({}, {"_id": 0}).sort("id", 1))

def get_plant_readings(plant_id):
    if readings_col is None: return []
    return list(readings_col.find({"plant_id": plant_id}, {"_id": 0}).sort("timestamp", 1))

def save_diagnosis(plant_id, plant_name, image, class_name, diagnosis_heb, notes):
    if diagnoses_col is None: return
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    doc = {
        "plant_id": plant_id,
        "plant_name": plant_name,
        "class_name": class_name,
        "diagnosis": diagnosis_heb,
        "notes": notes,
        "image_b64": img_b64,
        "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "created_at": datetime.now(timezone.utc),
    }
    diagnoses_col.insert_one(doc)

def load_diagnoses(plant_id):
    if diagnoses_col is None: return []
    return list(diagnoses_col.find({"plant_id": plant_id}, {"_id": 0}).sort("created_at", 1))

def delete_diagnosis(plant_id, timestamp):
    if diagnoses_col is None: return
    diagnoses_col.delete_one({"plant_id": plant_id, "timestamp": timestamp})

# ─────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────
if st.session_state.page == "home":
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<div class='main-title'>מערכת ניטור חיישנים ואבחון מחלות חיטה v3</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>מבצעים: נבו הלר ומתן אדר | מנחה: אסי ברק</div>", unsafe_allow_html=True)
    
    if db is not None:
        st.markdown("<div style='text-align:center'><span class='db-badge'>🟢 מחובר ל-MongoDB Atlas v3</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:center'><span class='db-badge' style='background:#ffebee;color:#c62828'>🔴 לא מחובר ל-MongoDB (בדוק Network Access ב-Atlas)</span></div>", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.markdown("""<div class="home-box">
          <span style='font-size:3.5rem'>📸</span>
          <h3 style='color:#1565c0;margin-top:15px'>אבחון חזותי מהיר</h3>
          <p style='color:#666;font-size:1rem;line-height:1.6'>
            בדיקה מיידית של עלה במודל ResNet18 — העלאת תמונה וקבלת אבחון והנחיות.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("הפעל אבחון מהיר 🚀", use_container_width=True, key="btn_single", type="primary"):
            st.session_state.page = "single_diagnosis"; st.rerun()

    with col2:
        st.markdown("""<div class="home-box">
          <span style='font-size:3.5rem'>📊</span>
          <h3 style='color:#2e7d32;margin-top:15px'>ניטור חיישנים וניסוי</h3>
          <p style='color:#666;font-size:1rem;line-height:1.6'>
            ניטור בזמן אמת של טמפרטורה ולחות, גרפים רציפים ותיעוד אבחוני חממה.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("פתח מערכת ניסוי 🔬", use_container_width=True, key="btn_exp", type="primary"):
            st.session_state.page = "experiment_management"; st.rerun()

    with col3:
        st.markdown("""<div class="home-box">
          <span style='font-size:3.5rem'>⚙️</span>
          <h3 style='color:#6a1b9a;margin-top:15px'>ניהול נתונים</h3>
          <p style='color:#666;font-size:1rem;line-height:1.6'>
            העלאת קובצי חיישנים מעודכנים (Excel/CSV), ייצוא נתונים וסנכרון ענן.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("ניהול נתונים ⚙️", use_container_width=True, key="btn_admin", type="primary"):
            st.session_state.page = "data_management"; st.rerun()

# ─────────────────────────────────────────
# QUICK DIAGNOSIS
# ─────────────────────────────────────────
elif st.session_state.page == "single_diagnosis":
    if st.button("🔙 חזרה לדף הבית", key="back1"):
        st.session_state.page = "home"; st.rerun()
    st.markdown("<h2 style='color:#1565c0'>📸 אבחון חזותי מהיר (ResNet18)</h2>", unsafe_allow_html=True)
    st.divider()

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("### 📥 הזנת תמונה")
            method = st.radio("בחר דרך:", ("צילום במצלמה 📸", "העלאת קובץ 📁"), key="sm")
            img_file = (st.camera_input("צלם עלה", key="sc")
                        if "מצלמה" in method
                        else st.file_uploader("בחר קובץ", type=["jpg","png","jpeg"], key="su"))
    with c2:
        if img_file:
            image = Image.open(img_file).convert("RGB")
            st.image(image, caption="התמונה שהוזנה", use_container_width=True)
            class_name, conf = run_model(image)
            if class_name is None:
                st.error("⚠️ לא זוהה עלה רלוונטי. נסה לצלם מקרוב ובתאורה טובה.")
            else:
                info = DISEASE_INFO.get(class_name, {"heb": class_name, "desc": "", "tip": ""})
                st.markdown(f"### 🎯 אבחון: <span style='color:#1565c0'><b>{info['heb']}</b></span> (רמת ביטחון: {conf*100:.1f}%)", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="custom-card" style="border-right-color:#1565c0;background:#f1f8ff">
                  <h4 style="color:#1565c0;margin-top:0">🔬 תיאור המחלה:</h4>
                  <p style="line-height:1.6">{info['desc']}</p>
                  <hr style="border:0;border-top:1px solid #d0e2ff;margin:16px 0">
                  <h4 style="color:#2e7d32;margin-top:0">💡 המלצות טיפול:</h4>
                  <p style="line-height:1.6">{info['tip']}</p>
                </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# EXPERIMENT MANAGEMENT
# ─────────────────────────────────────────
elif st.session_state.page == "experiment_management":
    if st.button("🔙 חזרה לדף הבית", key="back2"):
        st.session_state.page = "home"; st.rerun()
    st.markdown("<h2 style='color:#2e7d32'>📊 ניטור חיישנים ומעקב ניסוי</h2>", unsafe_allow_html=True)
    st.divider()

    plants = get_all_plants()
    if not plants:
        st.warning("⚠️ טרם נטענו נתוני חיישנים במסד הנתונים. אנא עבור לדף ניהול הנתונים והעלה את קובץ ה-Excel/CSV.")
        st.stop()

    labels_list = [f"צמח #{p['id']} | טיפול: {p.get('#Treatment','—')}" for p in plants]
    nav1, nav2, nav3 = st.columns([1, 2, 1])
    with nav1:
        if st.button("➡️ הקודם", use_container_width=True):
            if st.session_state.current_plant_idx > 0:
                st.session_state.current_plant_idx -= 1; st.rerun()
    with nav3:
        if st.button("הבא ⬅️", use_container_width=True):
            if st.session_state.current_plant_idx < len(labels_list) - 1:
                st.session_state.current_plant_idx += 1; st.rerun()
    with nav2:
        sel = st.selectbox("בחר צמח לניטור:", labels_list, index=st.session_state.current_plant_idx, key="psel")
        st.session_state.current_plant_idx = labels_list.index(sel)

    plant = plants[st.session_state.current_plant_idx]
    plant_id = int(plant["id"])
    plant_name = str(plant["name"])

    # Metrics Cards
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔢 מזהה צמח", f"#{plant_id}")
        c2.metric("💧 לחות קרקע נוכחית", f"{plant.get('latest_moisture', 0):.4f}")
        c3.metric("🌡️ טמפרטורה נוכחית", f"{plant.get('latest_temperature', 0):.2f} °C")
        c4.metric("🧪 קבוצת טיפול", str(plant.get("#Treatment", "—")))

    # Dynamic Time-Series Chart from MongoDB
    st.subheader(f"📈 גרף ניטור חיישנים בזמן אמת — {plant_name}")
    readings = get_plant_readings(plant_id)
    if readings:
        rdf = pd.DataFrame(readings)
        rdf['timestamp'] = pd.to_datetime(rdf['timestamp'])
        rdf = rdf.set_index('timestamp')
        
        tab_m, tab_t = st.tabs(["💧 מדד לחות (Moisture)", "🌡️ טמפרטורה (Temperature °C)"])
        with tab_m:
            st.line_chart(rdf[['moisture']], height=250)
        with tab_t:
            st.line_chart(rdf[['temperature']], height=250)
    else:
        st.info("אין נתוני סדרות זמן זמינים לצמח זה.")

    st.divider()
    st.subheader("📸 הוספת אבחון חדש לצמח זה")
    with st.container(border=True):
        d1, d2 = st.columns(2, gap="medium")
        with d1:
            method = st.radio("דרך הזנה:", ("מצלמה 📸", "קובץ 📁"), key="em")
            img_file = (st.camera_input("צלם עלה", key="ec")
                        if "מצלמה" in method
                        else st.file_uploader("בחר קובץ", type=["jpg","png","jpeg"], key="eu"))
        with d2:
            notes = st.text_area("✍️ הערות מהחממה:", placeholder="הקלד תיאור או הערות ניטור...", height=150)

        if img_file:
            image = Image.open(img_file).convert("RGB")
            class_name, _ = run_model(image)
            auto_diag = DISEASE_INFO.get(class_name, {"heb": "לא זוהה"})["heb"] if class_name else "לא זוהה"
            st.markdown(f"**🔍 תוצאת מודל:** {auto_diag}")
            if st.button("💾 שמור אבחון ב-MongoDB", use_container_width=True, type="primary"):
                save_diagnosis(plant_id, plant_name, image, class_name or "Unknown", auto_diag, notes or "ללא הערות")
                st.success("✅ האבחון נשמר בהצלחה ב-MongoDB Atlas!")
                st.rerun()

    st.divider()
    st.subheader(f"🗄️ היסטוריית אבחונים — {plant_name}")
    history = load_diagnoses(plant_id)
    if history:
        for rec in reversed(history):
            with st.container(border=True):
                h1, h2 = st.columns([1, 3], gap="medium")
                with h1:
                    try:
                        img_bytes = base64.b64decode(rec["image_b64"])
                        st.image(Image.open(io.BytesIO(img_bytes)), use_container_width=True)
                    except Exception:
                        st.info("תמונה לא זמינה")
                with h2:
                    st.markdown(f"### 📅 `{rec['timestamp']}`")
                    st.markdown(f"**🔬 אבחון:** {rec['diagnosis']}")
                    st.markdown(f"**📝 הערות:** {rec['notes']}")
                    if st.button("🗑️ מחק רשומה", key=f"del_{rec['timestamp']}"):
                        delete_diagnosis(plant_id, rec["timestamp"])
                        st.rerun()
    else:
        st.info("אין עדיין אבחונים שמורים לצמח זה.")

# ─────────────────────────────────────────
# DATA MANAGEMENT
# ─────────────────────────────────────────
elif st.session_state.page == "data_management":
    if st.button("🔙 חזרה לדף הבית", key="back3"):
        st.session_state.page = "home"; st.rerun()
    st.markdown("<h2 style='color:#6a1b9a'>⚙️ ניהול נתונים וסנכרון MongoDB</h2>", unsafe_allow_html=True)
    st.divider()

    st.subheader("📥 העלאת קובץ חיישנים מעודכן (Excel / CSV)")
    with st.container(border=True):
        uploaded_file = st.file_uploader("בחר קובץ Excel או CSV של החיישנים:", type=["xlsx", "csv"], key="sensor_file")
        if uploaded_file:
            df_upload = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
            st.write(f"📊 הקובץ נטען: **{len(df_upload)}** נקודות זמן, **{len(df_upload.columns)}** עמודות מדדים.")
            st.dataframe(df_upload.head(5), use_container_width=True)
            
            if st.button("✅ אישור ועדכון מסד הנתונים MongoDB", type="primary", use_container_width=True):
                with st.spinner("מעבד נתונים ומעדכן ב-MongoDB Atlas..."):
                    process_and_save_sensor_file(df_upload)
                st.success("✅ הנתונים עודכנו בהצלחה ב-MongoDB!")
                st.balloons()

    st.divider()
    st.subheader("📊 סטטוס מסד הנתונים")
    with st.container(border=True):
        if db is not None:
            s1, s2, s3 = st.columns(3)
            s1.metric("🌱 צמחים במסד הנתונים", plants_col.count_documents({}))
            s2.metric("📈 סה\"כ מדידות חיישנים", readings_col.count_documents({}))
            s3.metric("📸 סה\"כ אבחונים מתועדים", diagnoses_col.count_documents({}))
        else:
            st.error("אין חיבור פעיל ל-MongoDB Atlas.")
