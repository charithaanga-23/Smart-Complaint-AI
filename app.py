from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

import os


# ============================================================
# FLASK SETUP
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# TRAINING DATA
# ============================================================

TRAINING_DATA = [

    # ---------------- BATTERY ----------------
    ("phone battery drains very quickly", "Battery Issue"),
    ("mobile battery is draining fast", "Battery Issue"),
    ("battery dies quickly", "Battery Issue"),
    ("battery does not last long", "Battery Issue"),
    ("phone battery charge is reducing quickly", "Battery Issue"),
    ("laptop battery drains very fast", "Battery Issue"),
    ("device battery is not working properly", "Battery Issue"),
    ("battery percentage drops quickly", "Battery Issue"),
    ("phone battery problem", "Battery Issue"),
    ("battery is not holding charge", "Battery Issue"),

    # ---------------- CHARGING ----------------
    ("phone is not charging", "Charging Issue"),
    ("mobile won't charge", "Charging Issue"),
    ("charger is not working", "Charging Issue"),
    ("charging stopped working", "Charging Issue"),
    ("phone charging very slowly", "Charging Issue"),
    ("device does not charge properly", "Charging Issue"),
    ("charging cable is not working", "Charging Issue"),
    ("laptop is not charging", "Charging Issue"),
    ("charging port problem", "Charging Issue"),

    # ---------------- SCREEN ----------------
    ("phone screen is broken", "Screen/Display Issue"),
    ("mobile screen is cracked", "Screen/Display Issue"),
    ("display is cracked", "Screen/Display Issue"),
    ("screen is not working", "Screen/Display Issue"),
    ("screen has black lines", "Screen/Display Issue"),
    ("phone display colour is dull", "Screen/Display Issue"),
    ("phone screen colour is faded", "Screen/Display Issue"),
    ("mobile display looks dull", "Screen/Display Issue"),
    ("screen colours are not clear", "Screen/Display Issue"),
    ("display colours look washed out", "Screen/Display Issue"),
    ("phone screen looks faded", "Screen/Display Issue"),
    ("screen brightness is low", "Screen/Display Issue"),
    ("phone screen is dim", "Screen/Display Issue"),
    ("display is flickering", "Screen/Display Issue"),
    ("phone screen is flickering", "Screen/Display Issue"),
    ("touch screen is not working", "Screen/Display Issue"),
    ("phone display is too dark", "Screen/Display Issue"),
    ("screen colour has changed", "Screen/Display Issue"),
    ("phone display problem", "Screen/Display Issue"),

    # ---------------- HARDWARE ----------------
    ("motherboard is damaged", "Hardware Issue"),
    ("computer motherboard not working", "Hardware Issue"),
    ("system board failure", "Hardware Issue"),
    ("laptop motherboard issue", "Hardware Issue"),
    ("keyboard is not working", "Hardware Issue"),
    ("mouse is not working", "Hardware Issue"),
    ("usb port is damaged", "Hardware Issue"),
    ("device hardware is broken", "Hardware Issue"),
    ("laptop hardware problem", "Hardware Issue"),
    ("computer hardware problem", "Hardware Issue"),

    # ---------------- SOFTWARE ----------------
    ("application keeps crashing", "Software Issue"),
    ("app crashes every time", "Software Issue"),
    ("software is not working", "Software Issue"),
    ("phone software has a problem", "Software Issue"),
    ("application is not opening", "Software Issue"),
    ("program keeps freezing", "Software Issue"),
    ("system software error", "Software Issue"),
    ("website is not working", "Software Issue"),
    ("app is stuck", "Software Issue"),
    ("software problem", "Software Issue"),

    # ---------------- INTERNET / CONNECTIVITY ----------------
    ("internet connection is not working", "Connectivity Issue"),
    ("wifi keeps disconnecting", "Connectivity Issue"),
    ("wifi is not working", "Connectivity Issue"),
    ("network problem", "Connectivity Issue"),
    ("unable to connect to internet", "Connectivity Issue"),
    ("mobile data is not working", "Connectivity Issue"),
    ("bluetooth is not connecting", "Connectivity Issue"),
    ("internet is very slow", "Connectivity Issue"),
    ("network keeps dropping", "Connectivity Issue"),
    ("cannot connect to wifi", "Connectivity Issue"),

    # ---------------- DAMAGED PRODUCT ----------------
    ("product arrived damaged", "Damaged Product"),
    ("item was broken during delivery", "Damaged Product"),
    ("received damaged product", "Damaged Product"),
    ("package is damaged", "Damaged Product"),
    ("product is broken", "Damaged Product"),
    ("item arrived broken", "Damaged Product"),
    ("box was damaged", "Damaged Product"),
    ("product was damaged when delivered", "Damaged Product"),

    # ---------------- WRONG PRODUCT ----------------
    ("wrong product was delivered", "Wrong Product"),
    ("received the wrong item", "Wrong Product"),
    ("incorrect product delivered", "Wrong Product"),
    ("this is not what I ordered", "Wrong Product"),
    ("I received a different product", "Wrong Product"),
    ("wrong item received", "Wrong Product"),
    ("order contains the wrong product", "Wrong Product"),

    # ---------------- REFUND ----------------
    ("refund has not arrived", "Refund Issue"),
    ("money has not been refunded", "Refund Issue"),
    ("refund is delayed", "Refund Issue"),
    ("I want a refund", "Refund Issue"),
    ("order was cancelled but refund is missing", "Refund Issue"),
    ("refund not received", "Refund Issue"),
    ("where is my refund", "Refund Issue"),
    ("refund money is missing", "Refund Issue"),

    # ---------------- PAYMENT ----------------
    ("payment failed", "Payment Issue"),
    ("payment was declined", "Payment Issue"),
    ("money was deducted but order failed", "Payment Issue"),
    ("I was charged twice", "Payment Issue"),
    ("payment problem", "Payment Issue"),
    ("card payment is not working", "Payment Issue"),
    ("online payment failed", "Payment Issue"),
    ("money was deducted incorrectly", "Payment Issue"),

    # ---------------- DELIVERY ----------------
    ("delivery is late", "Delivery Delay"),
    ("order has not arrived", "Delivery Delay"),
    ("package delivery delayed", "Delivery Delay"),
    ("delivery is taking too long", "Delivery Delay"),
    ("my order is delayed", "Delivery Delay"),
    ("where is my package", "Delivery Delay"),
    ("package has not arrived", "Delivery Delay"),
    ("order has not been delivered", "Delivery Delay"),

    # ---------------- CANCELLATION ----------------
    ("I want to cancel my order", "Order Cancellation"),
    ("cancel my purchase", "Order Cancellation"),
    ("I need to cancel the order", "Order Cancellation"),
    ("please cancel my order", "Order Cancellation"),
    ("how can I cancel my order", "Order Cancellation"),

    # ---------------- REPLACEMENT ----------------
    ("I want a replacement", "Replacement Request"),
    ("please replace this product", "Replacement Request"),
    ("I need a new product", "Replacement Request"),
    ("can you replace my item", "Replacement Request"),
    ("product replacement required", "Replacement Request"),

    # ---------------- WARRANTY ----------------
    ("I want to claim warranty", "Warranty Issue"),
    ("product is under warranty", "Warranty Issue"),
    ("warranty claim problem", "Warranty Issue"),
    ("how can I use my warranty", "Warranty Issue"),
    ("warranty service required", "Warranty Issue"),

    # ---------------- PRODUCT QUALITY ----------------
    ("product quality is poor", "Product Quality Issue"),
    ("bad quality product", "Product Quality Issue"),
    ("quality is not as expected", "Product Quality Issue"),
    ("product quality is disappointing", "Product Quality Issue"),
    ("material quality is poor", "Product Quality Issue"),
    ("product does not look good", "Product Quality Issue"),
    ("item quality is bad", "Product Quality Issue"),
    ("poor quality item", "Product Quality Issue"),

    # ---------------- ACCOUNT / LOGIN ----------------
    ("I cannot login", "Account/Login Issue"),
    ("unable to login", "Account/Login Issue"),
    ("my account is not working", "Account/Login Issue"),
    ("password is not working", "Account/Login Issue"),
    ("I cannot access my account", "Account/Login Issue"),
    ("login problem", "Account/Login Issue"),
    ("account access problem", "Account/Login Issue"),

    # ---------------- CUSTOMER SERVICE ----------------
    ("customer service is not responding", "Customer Service Issue"),
    ("support team is not helping", "Customer Service Issue"),
    ("I cannot contact customer support", "Customer Service Issue"),
    ("customer care is not responding", "Customer Service Issue"),
    ("support is taking too long", "Customer Service Issue"),

    # ---------------- SUBSCRIPTION ----------------
    ("subscription problem", "Subscription Issue"),
    ("I want to cancel my subscription", "Subscription Issue"),
    ("subscription payment problem", "Subscription Issue"),
    ("subscription renewed unexpectedly", "Subscription Issue"),
    ("subscription is not working", "Subscription Issue"),

    # ---------------- PRIVACY / SECURITY ----------------
    ("someone accessed my account", "Security Issue"),
    ("my account was hacked", "Security Issue"),
    ("I see suspicious activity", "Security Issue"),
    ("security problem with my account", "Security Issue"),
    ("unauthorized account access", "Security Issue"),

    # ---------------- GENERAL PRODUCT COMPLAINTS ----------------
    ("product stopped working", "General Product Complaint"),
    ("I am unhappy with the product", "General Product Complaint"),
    ("I have a problem with my product", "General Product Complaint"),
    ("there is a problem with my order", "General Product Complaint"),
    ("product is not satisfactory", "General Product Complaint"),
    ("I am not satisfied with my purchase", "General Product Complaint"),
]


TRAINING_TEXTS = [item[0] for item in TRAINING_DATA]
TRAINING_LABELS = [item[1] for item in TRAINING_DATA]


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

# Word features understand words and phrases.
# Character features help with spelling variations and unusual wording.

features = FeatureUnion([
    (
        "word_tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    ),
    (
        "char_tfidf",
        TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True
        )
    )
])


model = Pipeline([
    ("features", features),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])


model.fit(TRAINING_TEXTS, TRAINING_LABELS)


# ============================================================
# DEPARTMENT MAPPING
# ============================================================

DEPARTMENT_MAP = {

    "Battery Issue":
        "Electronics Support",

    "Charging Issue":
        "Electronics Support",

    "Screen/Display Issue":
        "Technical Support",

    "Hardware Issue":
        "Technical Support",

    "Software Issue":
        "Technical Support",

    "Connectivity Issue":
        "Technical Support",

    "Damaged Product":
        "Returns & Replacement",

    "Wrong Product":
        "Returns & Replacement",

    "Refund Issue":
        "Billing & Payments",

    "Payment Issue":
        "Billing & Payments",

    "Delivery Delay":
        "Logistics & Delivery",

    "Order Cancellation":
        "Order Management",

    "Replacement Request":
        "Returns & Replacement",

    "Warranty Issue":
        "Warranty Support",

    "Product Quality Issue":
        "Quality & Customer Care",

    "Account/Login Issue":
        "Account Support",

    "Customer Service Issue":
        "Customer Support",

    "Subscription Issue":
        "Billing & Payments",

    "Security Issue":
        "Account Security",

    "General Product Complaint":
        "Customer Support",

    "General Complaint":
        "Customer Support"
}


# ============================================================
# RESPONSE MAPPING
# ============================================================

RESPONSE_MAP = {

    "Battery Issue":
        "Please inspect the battery and device power system. Our technical team should review the issue.",

    "Charging Issue":
        "Please check the charger, cable and charging port. Our technical team can assist with further inspection.",

    "Screen/Display Issue":
        "Please arrange a display inspection and replacement assessment.",

    "Hardware Issue":
        "Please arrange a technical inspection. The hardware support team should review the device.",

    "Software Issue":
        "Please provide the device and software details so our technical team can investigate the software issue.",

    "Connectivity Issue":
        "Please check the network settings and connection status. Our technical support team can assist further.",

    "Damaged Product":
        "Please provide the order details and photos of the damage so the returns team can arrange a resolution.",

    "Wrong Product":
        "Please verify the order and delivered item so the returns team can arrange the correct product.",

    "Refund Issue":
        "Please verify the payment and refund transaction with the billing team.",

    "Payment Issue":
        "Please verify the payment transaction with the billing team and provide the transaction details if required.",

    "Delivery Delay":
        "Please verify the shipment status and provide the latest delivery update.",

    "Order Cancellation":
        "Please verify the order details so the order management team can check the cancellation request.",

    "Replacement Request":
        "Please provide the order and product details so the returns team can process the replacement request.",

    "Warranty Issue":
        "Please provide the product and purchase details so the warranty team can verify the warranty claim.",

    "Product Quality Issue":
        "Please review the product quality concern and route it to customer care for resolution.",

    "Account/Login Issue":
        "Please verify your account details and contact account support for assistance with login or account access.",

    "Customer Service Issue":
        "Your complaint has been routed to customer support for further assistance.",

    "Subscription Issue":
        "Please verify the subscription and billing details so the support team can investigate the issue.",

    "Security Issue":
        "Please secure the account and route the complaint to the account security team for investigation.",

    "General Product Complaint":
        "Please provide the order and product details so our customer support team can investigate the complaint.",

    "General Complaint":
        "Thank you for reporting the issue. Our customer support team should review the complaint and provide an appropriate resolution."
}


# ============================================================
# PRIORITY
# ============================================================

HIGH_PRIORITY_CATEGORIES = {
    "Battery Issue",
    "Charging Issue",
    "Hardware Issue",
    "Screen/Display Issue",
    "Security Issue"
}


MEDIUM_PRIORITY_CATEGORIES = {
    "Connectivity Issue",
    "Damaged Product",
    "Wrong Product",
    "Refund Issue",
    "Payment Issue",
    "Warranty Issue",
    "Replacement Request",
    "Account/Login Issue",
    "Subscription Issue"
}


URGENT_WORDS = [
    "fire",
    "smoke",
    "burning",
    "burn",
    "explosion",
    "electric shock",
    "shock",
    "sparking",
    "spark",
    "overheating",
    "overheated",
    "danger",
    "dangerous",
    "unsafe",
    "injury",
    "injured",
    "hacked",
    "stolen"
]


def get_priority(category, complaint):

    text = complaint.lower()

    # Safety/security related words override normal category priority.
    for word in URGENT_WORDS:
        if word in text:
            return "High"

    if category in HIGH_PRIORITY_CATEGORIES:
        return "High"

    if category in MEDIUM_PRIORITY_CATEGORIES:
        return "Medium"

    return "Low"


# ============================================================
# CATEGORY PREDICTION
# ============================================================

def predict_category(complaint):

    probabilities = model.predict_proba([complaint])[0]

    best_index = probabilities.argmax()

    category = model.classes_[best_index]

    confidence = probabilities[best_index]

    # If the model is uncertain, don't force an unrelated category.
    if confidence < 0.32:
        return "General Complaint", confidence

    return category, confidence


# ============================================================
# FRONTEND ROUTES
# ============================================================

@app.get("/")
def home():
    return send_from_directory(".", "login.html")


@app.get("/login.html")
def login_page():
    return send_from_directory(".", "login.html")


@app.get("/index.html")
def index_page():
    return send_from_directory(".", "index.html")


@app.get("/style.css")
def style_css():
    return send_from_directory(".", "style.css")


@app.get("/script.js")
def script_js():
    return send_from_directory(".", "script.js")


# ============================================================
# PREDICTION API
# ============================================================

@app.post("/predict")
def predict():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "No JSON data received."
        }), 400


    required_fields = [
        "gender",
        "age",
        "product_category",
        "complaint_text"
    ]


    missing = [
        field
        for field in required_fields
        if field not in data
        or str(data[field]).strip() == ""
    ]


    if missing:

        return jsonify({
            "error":
                "Missing fields: "
                + ", ".join(missing)
        }), 400


    # Validate age
    try:

        age = int(data["age"])

        if age < 1 or age > 120:

            return jsonify({
                "error":
                    "Age must be between 1 and 120."
            }), 400

    except (TypeError, ValueError):

        return jsonify({
            "error":
                "Age must be a valid number."
        }), 400


    complaint = str(
        data["complaint_text"]
    ).strip()


    if len(complaint) < 5:

        return jsonify({
            "error":
                "Complaint description is too short."
        }), 400


    # ========================================================
    # AI PREDICTION
    # ========================================================

    category, confidence = predict_category(
        complaint
    )


    priority = get_priority(
        category,
        complaint
    )


    department = DEPARTMENT_MAP.get(
        category,
        "Customer Support"
    )


    suggested_response = RESPONSE_MAP.get(
        category,
        RESPONSE_MAP["General Complaint"]
    )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return jsonify({

        "complaint_category":
            category,

        "priority":
            priority,

        "department":
            department,

        "suggested_response":
            suggested_response

    })


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
