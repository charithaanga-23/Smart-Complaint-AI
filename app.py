from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

app = Flask(__name__)
CORS(app)

# Small built-in training set so the project runs immediately.
# You can replace/extend this data later with your real dataset.
TRAINING_TEXTS = [
    "phone battery drains quickly", "mobile battery charge drain", "battery dies very fast",
    "laptop battery is draining", "device battery problem",
    "motherboard is damaged", "computer motherboard not working", "system board failure",
    "laptop motherboard issue",
    "phone screen is broken", "display is cracked", "mobile screen not working",
    "screen has black lines",
    "internet connection is not working", "wifi keeps disconnecting", "network problem",
    "unable to connect to internet",
    "product arrived damaged", "item was broken on delivery", "received damaged product",
    "package is damaged",
    "wrong product was delivered", "received the wrong item", "incorrect product delivered",
    "refund has not arrived", "money has not been refunded", "refund is delayed",
    "i want a refund", "order was cancelled but refund missing",
    "delivery is late", "order has not arrived", "package delivery delayed",
    "delivery is taking too long",
    "product quality is poor", "bad quality product", "quality is not as expected",
    "product stopped working after purchase"
]

TRAINING_LABELS = [
    "Battery Issue", "Battery Issue", "Battery Issue", "Battery Issue", "Battery Issue",
    "Hardware Issue", "Hardware Issue", "Hardware Issue", "Hardware Issue",
    "Screen/Display Issue", "Screen/Display Issue", "Screen/Display Issue", "Screen/Display Issue",
    "Connectivity Issue", "Connectivity Issue", "Connectivity Issue", "Connectivity Issue",
    "Damaged Product", "Damaged Product", "Damaged Product", "Damaged Product",
    "Wrong Product", "Wrong Product", "Wrong Product",
    "Refund Issue", "Refund Issue", "Refund Issue", "Refund Issue", "Refund Issue",
    "Delivery Delay", "Delivery Delay", "Delivery Delay", "Delivery Delay",
    "Product Quality Issue", "Product Quality Issue", "Product Quality Issue",
    "Product Quality Issue"
]

model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(TRAINING_TEXTS, TRAINING_LABELS)

DEPARTMENT_MAP = {
    "Battery Issue": "Electronics Support",
    "Hardware Issue": "Technical Support",
    "Screen/Display Issue": "Technical Support",
    "Connectivity Issue": "Technical Support",
    "Damaged Product": "Returns & Replacement",
    "Wrong Product": "Returns & Replacement",
    "Refund Issue": "Billing & Payments",
    "Delivery Delay": "Logistics & Delivery",
    "Product Quality Issue": "Quality & Customer Care"
}

RESPONSE_MAP = {
    "Battery Issue": "Please inspect the battery and device power system. Our technical team should review the issue.",
    "Hardware Issue": "Please arrange a technical inspection. The hardware support team should review the device.",
    "Screen/Display Issue": "Please arrange a display inspection and replacement assessment.",
    "Connectivity Issue": "Please check the network settings and route the complaint to technical support.",
    "Damaged Product": "Please provide the order details so the returns team can arrange a replacement or resolution.",
    "Wrong Product": "Please verify the order and delivered item so the returns team can arrange the correct product.",
    "Refund Issue": "Please verify the payment and refund transaction with the billing team.",
    "Delivery Delay": "Please verify the shipment status and provide the latest delivery update.",
    "Product Quality Issue": "Please review the product quality concern and route it to customer care for resolution."
}

HIGH_PRIORITY = {
    "Battery Issue", "Hardware Issue", "Screen/Display Issue"
}

MEDIUM_PRIORITY = {
    "Damaged Product", "Wrong Product", "Refund Issue", "Connectivity Issue"
}

def get_priority(category, complaint):
    text = complaint.lower()

    urgent_words = [
        "fire", "smoke", "burn", "explosion", "electric shock",
        "danger", "sparking", "overheating", "motherboard"
    ]

    if category in HIGH_PRIORITY or any(word in text for word in urgent_words):
        return "High"
    if category in MEDIUM_PRIORITY:
        return "Medium"
    return "Low"

@app.get("/")
def home():
    return send_from_directory(".", "login.html")

@app.get("/login.html")
def login_page():
    return send_from_directory(".", "login.html")

@app.get("/index.html")
def index_page():
    return send_from_directory(".", "index.html")
@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "No JSON data received."}), 400

    required_fields = ["gender", "age", "product_category", "complaint_text"]
    missing = [field for field in required_fields if field not in data or str(data[field]).strip() == ""]

    if missing:
        return jsonify({"error": "Missing fields: " + ", ".join(missing)}), 400

    try:
        age = int(data["age"])
        if age < 1 or age > 120:
            return jsonify({"error": "Age must be between 1 and 120."}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "Age must be a valid number."}), 400

    complaint = str(data["complaint_text"]).strip()

    if len(complaint) < 5:
        return jsonify({"error": "Complaint description is too short."}), 400

    category = model.predict([complaint])[0]
    priority = get_priority(category, complaint)
    department = DEPARTMENT_MAP.get(category, "Customer Support")
    suggested_response = RESPONSE_MAP.get(
        category,
        "Please review the complaint and assign it to the appropriate support team."
    )

    return jsonify({
        "complaint_category": category,
        "priority": priority,
        "department": department,
        "suggested_response": suggested_response
    })

if __name__ == "__main__":
    import os

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
