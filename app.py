import os
import time
import requests
import random
import logging
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from contextlib import contextmanager
from dotenv import load_dotenv
import google.generativeai as genai
from history_model import train_history_model, predict_with_history

# ===============================
# LOAD ENV & FLASK SETUP
# ===============================
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))

# ===============================
# GEMINI SETUP
# ===============================
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file")

genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# ===============================
# DATABASE HELPERS
# ===============================
DATABASE = "database.db"

@contextmanager
def get_db():
    """Context manager for safe DB connections."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            city TEXT DEFAULT 'Unknown'
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            message TEXT NOT NULL,
            response TEXT NOT NULL,
            disease TEXT DEFAULT 'General',
            city TEXT DEFAULT 'Unknown',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        # Create indexes for performance
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chats_username ON chats(username)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chats_disease ON chats(disease)")

        # Default admin
        cur.execute("SELECT id FROM users WHERE username='admin'")
        if not cur.fetchone():
            admin_pw = generate_password_hash("admin123")
            cur.execute(
                "INSERT INTO users (username, password, role, city) VALUES (?, ?, ?, ?)",
                ("admin", admin_pw, "admin", "Dehradun")
            )
        conn.commit()

init_db()

# ===============================
# HELPERS
# ===============================
def get_nearby_hospitals(lat, lon):
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:15];
    (
      node["amenity"="hospital"](around:3000,{lat},{lon});
      way["amenity"="hospital"](around:3000,{lat},{lon});
    );
    out center tags;
    """
    try:
        headers = {'User-Agent': 'AIMedAssistantProject/1.0'}
        res = requests.get(overpass_url, params={'data': query}, headers=headers, timeout=20)
        res.raise_for_status()
        data = res.json()

        hospital_list = []
        for element in data.get('elements', []):
            tags = element.get('tags', {})
            name = tags.get('name')
            if not name:
                continue

            h_lat = element.get('lat') or element.get('center', {}).get('lat')
            h_lon = element.get('lon') or element.get('center', {}).get('lon')
            map_link = f"https://www.google.com/maps/search/?api=1&query={h_lat},{h_lon}"

            addr_parts = [tags.get(f'addr:{k}', '') for k in ['street', 'city'] if tags.get(f'addr:{k}')]
            addr = ", ".join(addr_parts)

            rating = float(tags.get('rating', 0.0))
            if rating == 0:
                rating = round(random.uniform(3.5, 4.9), 1)

            name_lower = name.lower()
            is_gov = any(w in name_lower for w in ['gov', 'civil', 'district', 'phc', 'chc', 'esi'])
            h_type = "Government" if is_gov else "Private"

            hospital_list.append({
                "name": name, "type": h_type,
                "rating": rating, "address": addr or "On Map", "link": map_link
            })

        sorted_hospitals = sorted(hospital_list, key=lambda x: x['rating'], reverse=True)
        result_str = "".join([
            f"- **{h['name']}**\n  ⭐ {h['rating']}/5 | 🏥 {h['type']} | 🔗 [Map]({h['link']})\n\n"
            for h in sorted_hospitals[:10]
        ])
        return result_str if result_str else "No hospitals found within 3km."
    except Exception as e:
        logger.warning(f"Hospital fetch failed: {e}")
        return "Could not fetch hospital data at this time."

def extract_disease(text):
    """Safely extract disease name from Markdown response."""
    try:
        if "### 🦠 Possible Diseases" in text:
            section = text.split("### 🦠 Possible Diseases")[1].split("###")[0]
            first_line = section.strip().split('\n')[0]
            # Clean markdown formatting
            first_line = first_line.replace('*', '').replace('-', '').strip()
            return first_line[:40] + "..." if len(first_line) > 40 else first_line
    except Exception:
        pass
    return "Triage / General"

def login_required(f):
    """Decorator for routes that require login."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session or session.get("role") != "admin":
            return redirect(url_for("chat_page"))
        return f(*args, **kwargs)
    return decorated

# ===============================
# AUTH ROUTES
# ===============================
@app.route("/")
def home():
    session.clear()
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        city = request.form.get("city", "Unknown").strip().title()

        if not username or not password:
            return render_template("register.html", error="Username and password are required.")

        with get_db() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    "INSERT INTO users (username, password, city) VALUES (?, ?, ?)",
                    (username, generate_password_hash(password), city)
                )
                conn.commit()
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                return render_template("register.html", error="Username already exists.")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE username=?", (username,))
            user = cur.fetchone()

        if user and check_password_hash(user["password"], password):
            session.clear()
            session["user"] = username
            session["role"] = user["role"]
            session["city"] = user["city"].strip().title() if user["city"] else "Unknown"
            session["chat_state"] = "triage_1"
            return redirect(url_for("admin") if user["role"] == "admin" else url_for("chat_page"))
        
        return render_template("login.html", error="Invalid username or password.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ===============================
# CHAT
# ===============================
@app.route("/chat", methods=["GET"])
@login_required
def chat_page():
    return render_template("chat.html", username=session["user"])

@app.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    user_msg = data.get("message", "").strip()
    lang = data.get("lang", "en-US")
    lat = data.get("lat")
    lon = data.get("lon")

    if not user_msg:
        return jsonify({"response": "Please type a message.", "follow_ups": []})

    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT message, response FROM chats WHERE username=? ORDER BY timestamp DESC LIMIT 4",
                (session["user"],)
            )
            recent_chats = list(reversed(cur.fetchall()))

        chat_history_text = "".join([
            f"Patient: {row['message']}\nDoctor: {row['response'].split('===FOLLOWUPS===')[0].strip()}\n\n"
            for row in recent_chats
        ])

        history_model, vectorizer = train_history_model(session["user"])
        history_pred = predict_with_history(history_model, vectorizer, user_msg) if history_model else "No prior data."

        current_state = session.get("chat_state", "triage_1")
        
        # Override state for hospital requests
        hospital_keywords = ['hospital', 'clinic', 'emergency', 'doctor near', 'nearby']
        if lat and lon and any(w in user_msg.lower() for w in hospital_keywords):
            current_state = "diagnosis"

        # Save the state BEFORE advancing it — used later for disease extraction
        state_for_this_turn = current_state

        if current_state == "triage_1":
            behavior_logic = "TRIAGE MODE (Phase 1): Ask EXACTLY ONE empathetic follow-up question. Do NOT produce Final Diagnosis Format."
            session["chat_state"] = "triage_2"
        elif current_state == "triage_2":
            behavior_logic = f"TRIAGE MODE (Phase 2): ML History insight: '{history_pred}'. Ask ONE question linking current symptoms to past patterns. Do NOT produce Final Diagnosis Format."
            session["chat_state"] = "diagnosis"
        else:
            behavior_logic = "DIAGNOSIS MODE: STOP asking questions. Produce FINAL DIAGNOSIS FORMAT now."
            session["chat_state"] = "triage_1"

        hospital_context = ""
        if state_for_this_turn == "diagnosis" and lat and lon:
            hospital_context = f"\n[SYSTEM: Nearby hospitals found:\n{get_nearby_hospitals(lat, lon)}\nDisplay exactly as formatted above.]"

        prompt = f"""You are an advanced, empathetic AI Healthcare Assistant. You are NOT a replacement for real doctors.

[Recent Conversation History]
{chat_history_text}
[Current Patient Message]
{user_msg}

CRITICAL RULES:
1. Always respond in language: {lang}
2. BEHAVIOR: {behavior_logic}
3. FINAL DIAGNOSIS FORMAT (ONLY in DIAGNOSIS MODE):
   ### 🦠 Possible Conditions
   ### ⚠️ Severity Level
   ### 💊 Treatment Suggestions
   ### 🌿 Home Remedies
   ### ✅ Beneficial Foods
   ### ❌ Foods to Avoid
   ### 🩺 When to See a Doctor
   ### ⚠️ Disclaimer
   *This is AI-generated information and not a medical diagnosis. Always consult a qualified physician.*
4. End EVERY response with follow-up suggestions:
   ===FOLLOWUPS=== Option 1 | Option 2 | Option 3
{hospital_context}"""

        reply = ""
        follow_ups = []
        raw_reply = ""

        for attempt in range(3):
            try:
                response = gemini_model.generate_content(prompt)
                raw_reply = response.text.strip()
                if "===FOLLOWUPS===" in raw_reply:
                    parts = raw_reply.split("===FOLLOWUPS===")
                    reply = parts[0].strip()
                    follow_ups = [q.strip() for q in parts[1].split("|") if q.strip()][:4]
                else:
                    reply = raw_reply
                break
            except Exception as e:
                logger.warning(f"Gemini attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    return jsonify({
                        "response": "⚠️ The AI is currently unavailable. Please try again in a moment.",
                        "follow_ups": []
                    })

        # Extract disease only when this turn WAS the diagnosis turn (AI just produced the full diagnosis)
        disease_name = extract_disease(reply) if state_for_this_turn == "diagnosis" else "Triage Phase"

        # Save `reply` (already stripped of followup chips) instead of raw_reply
        with get_db() as conn:
            conn.execute(
                "INSERT INTO chats (username, message, response, disease, city) VALUES (?, ?, ?, ?, ?)",
                (session["user"], user_msg, reply, disease_name, session.get("city", "Unknown"))
            )
            conn.commit()

        return jsonify({"response": reply, "follow_ups": follow_ups})

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"response": f"⚠️ An unexpected error occurred. Please try again.", "follow_ups": []})

# ===============================
# DASHBOARD
# ===============================
@app.route("/dashboard")
@login_required
def dashboard():
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """SELECT disease, response, timestamp FROM chats
               WHERE username=? AND disease NOT IN ('Triage Phase', 'Triage / General', 'General')
               ORDER BY timestamp DESC""",
            (session["user"],)
        )
        data = cur.fetchall()
    return render_template("dashboard.html", data=data)

# ===============================
# ADMIN PANEL
# ===============================
@app.route("/admin")
@admin_required
def admin():
    with get_db() as conn:
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) as c FROM users WHERE role='user'")
        total_users = cur.fetchone()["c"]

        cur.execute("SELECT COUNT(*) as c FROM chats")
        total_chats = cur.fetchone()["c"]

        cur.execute("""
            SELECT disease, COUNT(*) as cnt FROM chats
            WHERE disease NOT IN ('Triage Phase', 'Triage / General', 'General')
            GROUP BY disease ORDER BY cnt DESC LIMIT 8
        """)
        diseases = cur.fetchall()

        cur.execute("SELECT city, COUNT(*) as cnt FROM users WHERE role='user' GROUP BY city ORDER BY cnt DESC")
        locations = cur.fetchall()

        cur.execute("""
            SELECT username, disease, city, timestamp FROM chats
            WHERE disease NOT IN ('Triage Phase', 'Triage / General')
            ORDER BY timestamp DESC LIMIT 15
        """)
        patients = cur.fetchall()

    return render_template("admin.html",
        total_users=total_users,
        total_chats=total_chats,
        disease_labels=[d["disease"] for d in diseases],
        disease_counts=[d["cnt"] for d in diseases],
        location_labels=[l["city"] for l in locations],
        location_counts=[l["cnt"] for l in locations],
        patients=patients
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)