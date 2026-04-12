from flask import Flask, render_template, request, redirect, session, jsonify
import sqlite3, os
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import google.generativeai as genai
from history_model import train_history_model, predict_with_history

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "fallback")

# -------- GEMINI --------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# -------- DB --------
def get_db():
    return sqlite3.connect("database.db")

def init_db():
    conn = get_db()
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users
    (id INTEGER PRIMARY KEY AUTOINCREMENT,
     username TEXT UNIQUE,
     password TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS chats
    (id INTEGER PRIMARY KEY AUTOINCREMENT,
     username TEXT,
     message TEXT,
     response TEXT,
     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    conn.commit()
    conn.close()

init_db()

# -------- ROUTES --------

@app.route('/')
def home():
    if 'user' in session:
        return render_template("chat.html", username=session['user'])
    return redirect('/login')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        user = request.form['username'].strip()
        pwd = generate_password_hash(request.form['password'].strip())

        try:
            conn = get_db()
            c = conn.cursor()
            c.execute("INSERT INTO users (username,password) VALUES (?,?)",(user,pwd))
            conn.commit()
            conn.close()
            return redirect('/login')
        except:
            return render_template("register.html", error="Username already exists")

    return render_template("register.html")

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        user = request.form['username'].strip()
        pwd = request.form['password'].strip()

        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?",(user,))
        data = c.fetchone()
        conn.close()

        if data and check_password_hash(data[2], pwd):
            session['user'] = user
            return redirect('/')
        return render_template("login.html", error="Invalid login")

    return render_template("login.html")

# -------- CHAT --------
@app.route('/chat', methods=['POST'])
def chat():
    if 'user' not in session:
        return jsonify({"response": "Unauthorized"}), 401

    user_msg = request.json.get('message', '').strip()

    history_model, vectorizer = train_history_model(session['user'])
    history_pred = None

    if history_model:
        history_pred = predict_with_history(history_model, vectorizer, user_msg)

    prompt = f"""
You are an advanced, empathetic AI Healthcare Assistant. 

User symptoms: {user_msg}
ML History insight: {history_pred}

Provide a structured, highly readable response using MARKDOWN formatting. 
Use appropriate medical emojis for each section. DO NOT use generic plain text. Use bullet points and bold text to make it readable.

Required Sections:
### 🦠 Possible Diseases
(List likely conditions based on symptoms)

### ⚠️ Severity
(State whether it is Low, Medium, or High and briefly why)

### 💊 Treatment & Common Medicines
(Provide general treatment guidance and name ONLY common, over-the-counter medicines)

### 🌿 Home Remedies
(List safe, practical home remedies)

### 🩺 When to see a Doctor
(Give clear indicators of when professional medical help is required)
"""

    try:
        ai_response = model.generate_content(prompt)
        reply = ai_response.text.strip()
    except Exception as e:
        reply = f"⚠️ **Error connecting to AI:** {str(e)}"

    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO chats (username,message,response) VALUES (?,?,?)",
              (session['user'], user_msg, reply))
    conn.commit()
    conn.close()

    return jsonify({"response": reply})

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/login')
        
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT message,response,timestamp FROM chats WHERE username=? ORDER BY timestamp DESC",
              (session['user'],))
    data = c.fetchall()
    conn.close()

    return render_template("dashboard.html", data=data)

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

if __name__ == "__main__":
    app.run(debug=True)