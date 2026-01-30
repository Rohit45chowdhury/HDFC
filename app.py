from flask import Flask, render_template, request, redirect, url_for, session, flash
import pickle, os, re, json
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
from functools import wraps
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


if os.name == "nt":  
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:  
    pytesseract.pytesseract.tesseract_cmd = "tesseract"


app = Flask(__name__)
app.secret_key = "super_secret_key_123"

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}
USERS_FILE = "users.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f:
        json.dump({}, f)


# load model
loan_model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

sms_model = pickle.load(open("smsmodel.pkl", "rb"))
sms_vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

creditmodel = pickle.load(open("creditmodel.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))


# nlp
def ensure_nltk_data():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]

    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)


ensure_nltk_data()

ps = PorterStemmer()

SCALED_COLUMNS = [
    "loan_amount_requested",
    "loan_tenure_months",
    "monthly_income",
    "cibil_score",
    "applicant_age"
]


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(path):
    ext = path.rsplit(".", 1)[1].lower()
    text = ""
    if ext == "pdf":
        for img in convert_from_path(path):
            text += pytesseract.image_to_string(img)
    else:
        text = pytesseract.image_to_string(Image.open(path))
    return text

def calculate_document_fraud(text):
    risk = 0
    if not re.search(r"\d{9,18}", text): risk += 30
    if not re.search(r"\d{2}/\d{2}/\d{4}", text): risk += 20
    if "bank" not in text.lower(): risk += 25
    if "salary" not in text.lower(): risk += 25
    return min(risk, 100)

def calculate_review_score(c):
    score = 0
    score += min(c["cibil_score"] / 900 * 40, 40)
    score += min(c["monthly_income"] / 100000 * 25, 25)
    score += 10 if 23 <= c["applicant_age"] <= 45 else 5
    return round(score, 2)

def calculate_loan_fraud(c):
    risk = 0
    if c["cibil_score"] < 650: risk += 30
    if c["loan_amount_requested"] > c["monthly_income"] * 10: risk += 30
    if c["applicant_age"] < 22: risk += 20
    return min(risk, 100)

def is_strong_password(password):
    return (
        len(password) >= 6 and
        re.search(r"[A-Z]", password) and
        re.search(r"[a-z]", password) and
        re.search(r"\d", password)
    )

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [
        ps.stem(i)
        for i in tokens
        if i.isalnum() and i not in stopwords.words("english")
    ]
    return " ".join(tokens)

def fraud_decision(prob):
    if prob < 0.30:
        return "AUTO APPROVED"
    elif prob < 0.60:
        return "OTP / 2FA REQUIRED"
    else:
        return "TRANSACTION BLOCKED"


@app.route("/")
def home():
    return render_template("hdfc.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].lower().strip()
        password = request.form["password"]

        if len(name) < 3 or name.isnumeric():
            flash("Invalid name", "danger")
            return redirect(url_for("register"))

        if not is_strong_password(password):
            flash("Password must contain upper, lower & number", "danger")
            return redirect(url_for("register"))

        with open(USERS_FILE) as f:
            users = json.load(f)

        if email in users:
            flash("Email already exists", "danger")
            return redirect(url_for("register"))

        users[email] = {
            "name": name,
            "password": generate_password_hash(password)
        }

        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=4)

        flash("Registration successful", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].lower()
        password = request.form["password"]

        with open(USERS_FILE) as f:
            users = json.load(f)

        if email not in users or not check_password_hash(users[email]["password"], password):
            flash("Invalid credentials", "danger")
            return redirect(url_for("login"))

        session.clear()
        session["user"] = email
        return redirect(url_for("home"))

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# loan
@app.route("/loansure")
@login_required
def loansure():
    return render_template(
        "index.html",
        form_data=session.get("form_data", {}),
        prediction=session.get("prediction"),
        review_score=session.get("review_score"),
        loan_fraud=session.get("loan_fraud"),
        doc_fraud=session.get("doc_fraud"),
        error=session.get("error")
    )

@app.route("/predict-loan", methods=["POST"])
@login_required
def predict_loan():
    session.pop("doc_fraud", None)
    session.pop("error", None)

    session["form_data"] = request.form.to_dict()

    data = {
        "loan_type": request.form["loan_type"],
        "loan_amount_requested": float(request.form["loan_amount"]),
        "loan_tenure_months": int(request.form["tenure"]),
        "employment_status": request.form["employment"],
        "monthly_income": float(request.form["income"]),
        "cibil_score": int(request.form["cibil"]),
        "property_ownership_status": request.form["property"],
        "applicant_age": int(request.form["age"]),
        "gender": request.form["gender"]
    }

    df = pd.DataFrame([data])

    for col in label_encoders:
        df[col] = label_encoders[col].transform(df[col])

    df[SCALED_COLUMNS] = scaler.transform(df[SCALED_COLUMNS])
    df = df[features]

    session["prediction"] = loan_model.predict(df)[0]
    session["review_score"] = calculate_review_score(data)
    session["loan_fraud"] = calculate_loan_fraud(data)

    return redirect(url_for("loansure"))

# document vriefy
@app.route("/verify", methods=["POST"])
@login_required
def verify():
    file = request.files.get("document")
    

    if not file or not allowed_file(file.filename):
        session["error"] = "Invalid file format"
        return redirect(url_for("loansure"))
    
    
    flash("Document verified successfully", "success")

    path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(path)

    text = extract_text(path)
    session["doc_fraud"] = calculate_document_fraud(text)
    session["error"] = None

    
    return redirect(url_for("loansure"))

# sms
@app.route("/sms", methods=["GET", "POST"])
@login_required
def sms():
    result = None
    message = ""

    if request.method == "POST":
        message = request.form["message"]
        vector = sms_vectorizer.transform([transform_text(message)])
        prediction = sms_model.predict(vector)[0]
        result = "🚨 Spam Message" if prediction == 1 else "✅ Not Spam"

    return render_template("sms.html", result=result, message=message)

# credit card
@app.route("/CreditSecure", methods=["GET", "POST"])
@login_required
def credit_secure():
    result = None
    probability = None

    if request.method == "POST":
        data = {
            "amt": float(request.form["amt"]),
            "gender": int(request.form["gender"]),
            "distance_km": float(request.form["distance_km"]),
            "city_pop_level": int(request.form["city_pop_level"]),
            "trans_hour": int(request.form["trans_hour"]),
            "trans_day": int(request.form["trans_day"]),
            "is_night": int(request.form["is_night"])
        }

        df = pd.DataFrame([data])

        for col in columns:
            if col not in df:
                df[col] = 0

        df = df[columns]

        probability = creditmodel.predict_proba(df)[0][1]
        result = fraud_decision(probability)

    return render_template("credit.html", result=result, probability=probability)


if __name__ == "__main__":
    app.run(debug=True)
