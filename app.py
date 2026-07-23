import streamlit as st
import pandas as pd

# ─────────────────────────────────────────────────────────
# 1. מילון מידע והמלצות טיפול לפי מחלה
# ─────────────────────────────────────────────────────────
DISEASE_INFO = {
    "BlackPoint": {
        "heb": "חוד שחור (Black Point)",
        "desc": "מחלה הנגרמת על ידי קומפלקס פטריות בשלבי הבשלת הגרעין. מתאפיינת בהשחרה של קצה הגרעין בעקבות לחות גבוהה.",
        "tip": "להפחית משטר השקיה בשלבי ההבשלה, לאוורר את החממה/השדה ולהשתמש בזרעים מחוטאים בעונה הבאה."
    },
    "FusariumFootRot": {
        "heb": "ריקבון בסיס הקנה (Fusarium)",
        "desc": "מחלה פטרייתית קרקעית התוקפת את השורשים ובסיס הקנה, חוסמת צינורות הובלה ומביאה לנבילה.",
        "tip": "יישום מחזור זרעים קפדני (להימנע מדגניים 2 עונות), הימנעות מהשקיית יתר ויירור הקרקע."
    },
    "HealthyLeaf": {
        "heb": "עלה בריא (Healthy)",
        "desc": "העלה מציג חיוניות גבוהה, צבע ירוק אחיד ושטח פנים נקי. הפוטוסינתזה מתנהלת בצורה אופטימלית.",
        "tip": "מצב מצוין! יש להמשיך במשטר הטיפוח הנוכחי ולבצע ניטור שבועי שגרתי."
    },
    "LeafBlight": {
        "heb": "קמלת עלים (Leaf Blight)",
        "desc": "מחלה פטרייתית המתבטאת בכתמים חומים-אפרפרים על העלים, ומפחיתה דרסטית את כושר הפוטוסינתזה.",
        "tip": "ריסוס בקוטלי פטריות מורשים עם הופעת הכתמים הראשונים, והרחקת שאריות צמחים נגועים."
    },
    "WheatBlast": {
        "heb": "פיריקורליית החיטה (Wheat Blast)",
        "desc": "מחלה פטרייתית הרסנית במיוחד — השיבולת הופכת ללבנה ויבשה תוך ימים ומונעת התפתחות גרגירים.",
        "tip": "🚨 מצב חירום חקלאי! עזל/בודד את האזור הנגוע מיד ורסס בקוטלי פטריות סיסטמיים בדחיפות."
    }
}

# ─────────────────────────────────────────────────────────
# 2. מנוע השילוב: מדדים + ניתוח תמונה + המלצות
# ─────────────────────────────────────────────────────────
def evaluate_combined_plant_status(plant_data: dict, model_class_name: str):
    """
    משקלל את מדדי החיישנים/עקה יחד עם זיהוי המודל ומפיק הערכה משולבת
    """
    treatment = str(plant_data.get("#Treatment", "Control"))
    stress = float(plant_data.get("stressDegree", 0.0))
    
    disease_data = DISEASE_INFO.get(model_class_name, DISEASE_INFO["HealthyLeaf"])
    disease_heb = disease_data["heb"]
    is_healthy_img = (model_class_name == "HealthyLeaf")
    
    # חישוב רמת העקה הפיזיולוגית
    has_high_stress = (treatment == "Drought" and stress > 0.15)
    has_moderate_stress = (treatment == "Drought" and stress <= 0.15 and stress > 0.05)
    
    # לוגיקת קבלת החלטות משולבת
    if not is_healthy_img and has_high_stress:
        status_level = "CRITICAL"
        title = "⚠️ קריטי — פגיעה משולבת (מחלה + עקת יובש חריפה)"
        color = "#d32f2f" # אדום
        bg_color = "#ffebee"
        summary = f"הצמח סובל גם ממחלת **{disease_heb}** וגם ממדד עקת יובש גבוה ({stress:.3f})."
        recommendation = f"**טיפול דחוף:** 1. טיפול פטרייתי: {disease_data['tip']} \n2. **השקיה:** תגבור מיידי ומבוקר של השקיה להפחתת עקת היובש."
        
    elif not is_healthy_img:
        status_level = "WARNING"
        title = f"☣️ זיהוי מחלה חזותית: {disease_heb}"
        color = "#f57c00" # כתום
        bg_color = "#fff3e0"
        summary = f"ניתוח התמונה זיהה את המחלה **{disease_heb}**. מדד העקה הפיזיולוגית הוא {stress:.3f}."
        recommendation = f"**טיפול מומלץ:** {disease_data['tip']}"
        
    elif has_high_stress or has_moderate_stress:
        status_level = "WARNING"
        title = "💧 עקת יובש פיזיולוגית (ללא סימני מחלה חזותית)"
        color = "#1976d2" # כחול/אזהרת מים
        bg_color = "#e3f2fd"
        summary = f"העלים נראים בריאים חזותית, אך מדדי העקה מצביעים על חוסר במים (מדד: {stress:.3f})."
        recommendation = "**טיפול מומלץ:** איזון ומשטר השקיה מוגבר למניעת התייבשות העלים והידרדרות."
        
    else:
        status_level = "HEALTHY"
        title = "✅ מצב תקין ומצוין"
        color = "#2e7d32" # ירוק
        bg_color = "#e8f5e9"
        summary = "הצמח מציג מדדי עקה תקינים וגם ניתוח התמונה מראה עלה בריא לחלוטין."
        recommendation = f"**טיפול מומלץ:** {disease_data['tip']}"

    return {
        "status_level": status_level,
        "title": title,
        "color": color,
        "bg_color": bg_color,
        "summary": summary,
        "recommendation": recommendation,
        "disease_heb": disease_heb,
        "disease_desc": disease_data["desc"]
    }

# ─────────────────────────────────────────────────────────
# 3. רכיב התצוגה ב-Streamlit
# ─────────────────────────────────────────────────────────
def render_plant_health_dashboard(plant_data: dict, latest_diagnosis_class: str = "HealthyLeaf"):
    """
    מציג את הלוח המשולב בממשק
    """
    eval_result = evaluate_combined_plant_status(plant_data, latest_diagnosis_class)
    
    st.markdown(f"### 🩺 הערכת מצב הצמח והמלצות טיפול")
    
    # כרטיסייה ראשית מעוצבת
    st.markdown(f"""
    <div style="
        background-color: {eval_result['bg_color']};
        border-right: 6px solid {eval_result['color']};
        padding: 18px 22px;
        border-radius: 10px;
        margin-bottom: 20px;
        direction: rtl;
        text-align: right;
    ">
        <h4 style="color: {eval_result['color']}; margin-top: 0; margin-bottom: 8px;">
            {eval_result['title']}
        </h4>
        <p style="font-size: 1.05rem; margin-bottom: 8px; color: #2c3e50;">
            {eval_result['summary']}
        </p>
        <hr style="border: 0; border-top: 1px solid rgba(0,0,0,0.1); margin: 10px 0;">
        <p style="font-size: 1.05rem; margin-bottom: 0; color: #1b5e20; font-weight: 600;">
            💡 {eval_result['recommendation']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # פירוט מדדים וניתוח
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="📊 מדד עקה (Stress Degree)", 
            value=f"{float(plant_data.get('stressDegree', 0)):.3f}",
            delta="עקה גבוהה" if float(plant_data.get('stressDegree', 0)) > 0.15 else "תקין",
            delta_color="inverse"
        )
    with col2:
        st.metric(
            label="🔍 אבחון תמונה (Model Diagnosis)", 
            value=eval_result["disease_heb"]
        )
        
    with st.expander("📖 למידע מורחב על המחלה/המצב הפיזיולוגי"):
        st.write(f"**תיאור מפורט:** {eval_result['disease_desc']}")
