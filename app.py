from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

import re


app = Flask(__name__)
CORS(app)


# ============================================================
# TRAINING DATA
# ============================================================

TRAINING_DATA = {

    "Battery Issue": [
        "battery drains quickly",
        "battery draining very fast",
        "phone battery dies quickly",
        "laptop battery drains",
        "battery backup is poor",
        "battery does not last",
        "battery percentage drops quickly",
        "my battery is getting weak",
        "phone battery problem",
        "battery is not working properly"
    ],

    "Charging Issue": [
        "phone is not charging",
        "charger is not working",
        "charging stopped",
        "device charges very slowly",
        "charging port is not working",
        "mobile is not taking charge",
        "charger problem",
        "charging cable is not working",
        "laptop is not charging",
        "charging issue"
    ],

    "Screen/Display Issue": [
        "screen is broken",
        "display is cracked",
        "screen is not working",
        "display is black",
        "screen has lines",
        "screen is flickering",
        "display is damaged",
        "touch screen is not working",
        "phone screen problem",
        "screen colour is changing",
        "screen colour is dull",
        "display colour is dull",
        "screen brightness problem"
    ],

    "Hardware Issue": [
        "motherboard is damaged",
        "motherboard is not working",
        "hardware is damaged",
        "laptop hardware problem",
        "computer hardware failure",
        "device is physically damaged",
        "keyboard is not working",
        "mouse is not working",
        "speaker is not working",
        "hardware problem"
    ],

    "Software Issue": [
        "software is not working",
        "application keeps crashing",
        "app crashes",
        "phone software problem",
        "system is freezing",
        "computer is freezing",
        "software error",
        "operating system problem",
        "application is not opening",
        "app is not responding"
    ],

    "Connectivity Issue": [
        "internet is not working",
        "wifi is not working",
        "wifi keeps disconnecting",
        "network problem",
        "cannot connect to internet",
        "bluetooth is not working",
        "mobile network problem",
        "internet connection is slow",
        "network keeps disconnecting",
        "connection problem"
    ],

    "Damaged Product": [
        "product arrived damaged",
        "item was broken when delivered",
        "package arrived damaged",
        "received damaged product",
        "product was damaged during delivery",
        "box was damaged",
        "item is broken",
        "product is physically damaged"
    ],

    "Wrong Product": [
        "wrong product delivered",
        "received wrong item",
        "incorrect product delivered",
        "this is not what i ordered",
        "different product received",
        "wrong item received",
        "order contains wrong product"
    ],

    "Refund Issue": [
        "refund has not arrived",
        "refund is delayed",
        "money has not been refunded",
        "i have not received my refund",
        "refund problem",
        "refund pending",
        "where is my refund",
        "cancelled order refund missing"
    ],

    "Payment Issue": [
        "payment failed",
        "payment was declined",
        "money was deducted",
        "payment problem",
        "payment is not going through",
        "charged twice",
        "double payment",
        "transaction failed",
        "money deducted but order failed"
    ],

    "Delivery Delay": [
        "delivery is late",
        "order has not arrived",
        "package is delayed",
        "delivery is taking too long",
        "my order is late",
        "package has not arrived",
        "shipment is delayed",
        "delivery delay"
    ],

    "Order Cancellation": [
        "i want to cancel my order",
        "cancel my order",
        "please cancel the order",
        "order cancellation",
        "i need to cancel my purchase",
        "how can i cancel my order"
    ],

    "Replacement Request": [
        "i want a replacement",
        "please replace my product",
        "replace the damaged item",
        "i need a new product",
        "requesting replacement",
        "can you replace this item",
        "product replacement"
    ],

    "Warranty Issue": [
        "product is under warranty",
        "warranty claim",
        "warranty problem",
        "i want to claim warranty",
        "is my product covered by warranty",
        "warranty service required"
    ],

    "Product Quality Issue": [
        "product quality is poor",
        "bad quality product",
        "quality is not good",
        "product is not as expected",
        "poor quality",
        "quality problem",
        "product quality issue"
    ],

    "Account/Login Issue": [
        "cannot login",
        "login is not working",
        "forgot password",
        "account is locked",
        "cannot access my account",
        "login problem",
        "password is not working",
        "account access problem"
    ],

    "Customer Service Issue": [
        "customer service was not helpful",
        "support team did not help",
        "customer care is not responding",
        "support is poor",
        "nobody responded to my complaint",
        "customer service problem"
    ],

    "Subscription Issue": [
        "subscription problem",
        "cancel my subscription",
        "subscription is not working",
        "subscription charged me",
        "subscription renewal problem",
        "membership issue"
    ],

    "Security Issue": [
        "someone accessed my account",
        "account security problem",
        "unauthorized transaction",
        "someone used my account",
        "security problem",
        "suspicious activity",
        "my account was hacked"
    ],

    "General Product Complaint": [
        "product is not working properly",
        "i am unhappy with the product",
        "problem with my product",
        "issue with the item",
        "product problem",
        "item is not satisfactory"
    ]
}


# ============================================================
# CREATE TRAINING DATA
# ============================================================

TRAINING_TEXTS = []
TRAINING_LABELS = []

for category, complaints in TRAINING_DATA.items():

    for complaint in complaints:
        TRAINING_TEXTS.append(complaint)
        TRAINING_LABELS.append(category)


# ============================================================
# AI MODEL
# ============================================================

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
        "character_tfidf",
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
# DEPARTMENTS
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
        "Orders & Customer Care",

    "Replacement Request":
        "Returns & Replacement",

    "Warranty Issue":
        "Warranty Support",

    "Product Quality Issue":
        "Quality & Customer Care",

    "Account/Login Issue":
        "Account Support",

    "Customer Service Issue":
        "Customer Care",

    "Subscription Issue":
        "Subscription Support",

    "Security Issue":
        "Security Support",

    "General Product Complaint":
        "Customer Support"
}


# ============================================================
# PRIORITY
# ============================================================

HIGH_PRIORITY = {
    "Battery Issue",
    "Charging Issue",
    "Hardware Issue",
    "Screen/Display Issue",
    "Security Issue"
}


MEDIUM_PRIORITY = {
    "Damaged Product",
    "Wrong Product",
    "Refund Issue",
    "Payment Issue",
    "Connectivity Issue",
    "Replacement Request",
    "Warranty Issue"
}


URGENT_WORDS = [
    "fire",
    "smoke",
    "burn",
    "burning",
    "explosion",
    "electric shock",
    "electricity",
    "sparking",
    "spark",
    "overheating",
    "overheated",
    "danger",
    "dangerous"
]


def get_priority(category, complaint):

    text = complaint.lower()

    if any(word in text for word in URGENT_WORDS):
        return "High"

    if category in HIGH_PRIORITY:
        return "High"

    if category in MEDIUM_PRIORITY:
        return "Medium"

    return "Low"


# ============================================================
# COMPLAINT-SPECIFIC RESPONSE
# ============================================================

def generate_response(category, complaint):

    text = complaint.lower()

    if category == "Battery Issue":
        return (
            "We identified a battery-related complaint. "
            "Please have the device and battery checked by the technical team. "
            "If the device is overheating or showing unusual battery behavior, "
            "stop using it and request technical assistance."
        )

    if category == "Charging Issue":
        return (
            "We identified a charging-related complaint. "
            "Please check the charger, charging cable and charging port. "
            "If the issue continues, the technical team should inspect the device."
        )

    if category == "Screen/Display Issue":
        return (
            "We identified a screen or display complaint. "
            "Please arrange a display inspection. "
            "The technical team can determine whether repair or replacement is required."
        )

    if category == "Hardware Issue":
        return (
            "We identified a hardware-related complaint. "
            "Please arrange a technical inspection so the hardware team can "
            "identify the affected component and recommend a suitable solution."
        )

    if category == "Software Issue":
        return (
            "We identified a software-related complaint. "
            "Please check for available software updates and restart the device. "
            "If the problem continues, technical support should investigate it."
        )

    if category == "Connectivity Issue":
        return (
            "We identified a connectivity-related complaint. "
            "Please check your network settings and connection. "
            "If the issue continues, technical support should investigate the connection."
        )

    if category == "Damaged Product":
        return (
            "We identified a damaged-product complaint. "
            "Please provide the order details and available product/delivery evidence "
            "so the returns team can review replacement or resolution options."
        )

    if category == "Wrong Product":
        return (
            "We identified a wrong-product complaint. "
            "Please verify the order details and delivered item. "
            "The returns team can review the order and arrange the appropriate resolution."
        )

    if category == "Refund Issue":
        return (
            "We identified a refund-related complaint. "
            "Please verify the order and payment details. "
            "The billing team should check the refund transaction and provide an update."
        )

    if category == "Payment Issue":
        return (
            "We identified a payment-related complaint. "
            "Please verify the transaction details. "
            "The billing team should check the payment status and resolve the issue."
        )

    if category == "Delivery Delay":
        return (
            "We identified a delivery-delay complaint. "
            "Please provide the order details so the logistics team can check "
            "the latest shipment status and delivery information."
        )

    if category == "Order Cancellation":
        return (
            "We identified an order-cancellation request. "
            "The orders team should verify the order status and process the "
            "cancellation according to the applicable order policy."
        )

    if category == "Replacement Request":
        return (
            "We identified a replacement request. "
            "Please provide the order and product details so the returns team "
            "can review replacement eligibility."
        )

    if category == "Warranty Issue":
        return (
            "We identified a warranty-related complaint. "
            "Please provide the purchase and product details so the warranty "
            "team can verify coverage and guide you through the next steps."
        )

    if category == "Product Quality Issue":
        return (
            "We identified a product-quality complaint. "
            "Please provide the product and order details so the quality and "
            "customer-care team can investigate the concern."
        )

    if category == "Account/Login Issue":
        return (
            "We identified an account or login complaint. "
            "Please verify your login details and use the account recovery "
            "options if necessary. Account support should assist if access remains unavailable."
        )

    if category == "Customer Service Issue":
        return (
            "We identified a customer-service complaint. "
            "The customer-care team should review the previous interaction "
            "and provide an appropriate follow-up."
        )

    if category == "Subscription Issue":
        return (
            "We identified a subscription-related complaint. "
            "Please verify the subscription and payment details so the "
            "subscription support team can review the issue."
        )

    if category == "Security Issue":
        return (
            "We identified a security-related complaint. "
            "Please secure the account and report the suspicious activity "
            "to the security support team for investigation."
        )

    return (
        "Your complaint could not be confidently assigned to a specific category. "
        "It has been routed to Customer Support for manual review so the appropriate "
        "team can investigate the issue."
    )


# ============================================================
# ROUTES FOR FRONTEND FILES
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


    # --------------------------------------------------------
    # REQUIRED FIELDS
    # --------------------------------------------------------

    required_fields = [
        "customer_name",
        "phone_number",
        "gender",
        "age",
        "product_category",
        "complaint_text"
    ]


    missing = []

    for field in required_fields:

        value = data.get(field)

        if value is None or str(value).strip() == "":
            missing.append(field)


    if missing:

        return jsonify({
            "error": "Missing fields: " + ", ".join(missing)
        }), 400


    # --------------------------------------------------------
    # NAME VALIDATION
    # --------------------------------------------------------

    customer_name = str(data["customer_name"]).strip()

    if len(customer_name) < 2:

        return jsonify({
            "error": "Please enter a valid customer name."
        }), 400


    # --------------------------------------------------------
    # PHONE VALIDATION
    # --------------------------------------------------------

    phone_number = str(data["phone_number"]).strip()

    phone_clean = re.sub(r"[\s\-()]", "", phone_number)

    if not re.fullmatch(r"\+?[0-9]{7,15}", phone_clean):

        return jsonify({
            "error": "Please enter a valid phone number."
        }), 400


    # --------------------------------------------------------
    # AGE VALIDATION
    # --------------------------------------------------------

    try:

        age = int(data["age"])

    except (TypeError, ValueError):

        return jsonify({
            "error": "Age must be a valid number."
        }), 400


    if age < 18:

        return jsonify({
            "error": "Only customers aged 18 or above can submit a complaint."
        }), 403


    if age > 120:

        return jsonify({
            "error": "Please enter a valid age."
        }), 400


    # --------------------------------------------------------
    # OTHER FIELDS
    # --------------------------------------------------------

    gender = str(data["gender"]).strip()

    product_category = str(
        data["product_category"]
    ).strip()


    complaint = str(
        data["complaint_text"]
    ).strip()


    if len(complaint) < 5:

        return jsonify({
            "error": "Please provide a more detailed complaint."
        }), 400


    if len(complaint) > 1000:

        return jsonify({
            "error": "Complaint must be 1000 characters or less."
        }), 400


    # --------------------------------------------------------
    # AI PREDICTION
    # --------------------------------------------------------

    probabilities = model.predict_proba([complaint])[0]

    best_index = probabilities.argmax()

    confidence = probabilities[best_index]

    predicted_category = model.classes_[best_index]


    # --------------------------------------------------------
    # LOW-CONFIDENCE FALLBACK
    # --------------------------------------------------------

    if confidence < 0.32:

        category = "General Product Complaint"

        department = "Customer Support"

    else:

        category = predicted_category

        department = DEPARTMENT_MAP.get(
            category,
            "Customer Support"
        )


    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

    priority = get_priority(
        category,
        complaint
    )


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    suggested_response = generate_response(
        category,
        complaint
    )


    # --------------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------------

    return jsonify({

        "complaint_category": category,

        "priority": priority,

        "department": department,

        "suggested_response": suggested_response

    })


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
