from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


app = Flask(__name__)
CORS(app)


# ============================================================
# TRAINING DATA
# ============================================================

TRAINING_DATA = [

    # ---------------- BATTERY ISSUE ----------------

    ("phone battery drains quickly", "Battery Issue"),
    ("mobile battery charge drain", "Battery Issue"),
    ("battery dies very fast", "Battery Issue"),
    ("laptop battery is draining", "Battery Issue"),
    ("device battery problem", "Battery Issue"),
    ("phone battery drains too fast", "Battery Issue"),
    ("mobile battery not lasting", "Battery Issue"),
    ("battery percentage drops quickly", "Battery Issue"),
    ("phone battery gets empty quickly", "Battery Issue"),
    ("battery is not holding charge", "Battery Issue"),


    # ---------------- HARDWARE ISSUE ----------------

    ("motherboard is damaged", "Hardware Issue"),
    ("computer motherboard not working", "Hardware Issue"),
    ("system board failure", "Hardware Issue"),
    ("laptop motherboard issue", "Hardware Issue"),
    ("computer hardware problem", "Hardware Issue"),
    ("laptop hardware is damaged", "Hardware Issue"),
    ("device hardware not working", "Hardware Issue"),
    ("computer internal hardware problem", "Hardware Issue"),


    # ---------------- SCREEN / DISPLAY ISSUE ----------------

    ("phone screen is broken", "Screen/Display Issue"),
    ("display is cracked", "Screen/Display Issue"),
    ("mobile screen not working", "Screen/Display Issue"),
    ("screen has black lines", "Screen/Display Issue"),

    # New display / colour examples
    ("phone display colour is dull", "Screen/Display Issue"),
    ("phone screen colour is faded", "Screen/Display Issue"),
    ("mobile display looks dull", "Screen/Display Issue"),
    ("phone screen colours are not clear", "Screen/Display Issue"),
    ("screen colour has changed", "Screen/Display Issue"),
    ("display colours look washed out", "Screen/Display Issue"),
    ("phone screen looks faded", "Screen/Display Issue"),
    ("mobile screen looks faded", "Screen/Display Issue"),
    ("screen has poor colour quality", "Screen/Display Issue"),
    ("display colour is not normal", "Screen/Display Issue"),
    ("phone colour is getting dull", "Screen/Display Issue"),
    ("mobile colour is getting dull", "Screen/Display Issue"),
    ("phone display is becoming dull", "Screen/Display Issue"),
    ("screen colour looks dull", "Screen/Display Issue"),
    ("phone screen colour looks different", "Screen/Display Issue"),
    ("display colour is fading", "Screen/Display Issue"),
    ("phone display is not bright", "Screen/Display Issue"),
    ("mobile display is not bright", "Screen/Display Issue"),
    ("screen brightness is low", "Screen/Display Issue"),
    ("phone screen brightness problem", "Screen/Display Issue"),
    ("display brightness problem", "Screen/Display Issue"),
    ("phone screen is dim", "Screen/Display Issue"),
    ("mobile screen is dim", "Screen/Display Issue"),
    ("screen is too dark", "Screen/Display Issue"),
    ("phone display is too dark", "Screen/Display Issue"),
    ("screen has display problem", "Screen/Display Issue"),
    ("phone display problem", "Screen/Display Issue"),
    ("mobile display problem", "Screen/Display Issue"),
    ("screen is flickering", "Screen/Display Issue"),
    ("phone screen is flickering", "Screen/Display Issue"),
    ("display is flickering", "Screen/Display Issue"),
    ("touch screen is not working", "Screen/Display Issue"),
    ("phone touch screen problem", "Screen/Display Issue"),


    # ---------------- CONNECTIVITY ISSUE ----------------

    ("internet connection is not working", "Connectivity Issue"),
    ("wifi keeps disconnecting", "Connectivity Issue"),
    ("network problem", "Connectivity Issue"),
    ("unable to connect to internet", "Connectivity Issue"),
    ("wifi is not working", "Connectivity Issue"),
    ("mobile network is not working", "Connectivity Issue"),
    ("internet keeps disconnecting", "Connectivity Issue"),
    ("phone cannot connect to wifi", "Connectivity Issue"),


    # ---------------- DAMAGED PRODUCT ----------------

    ("product arrived damaged", "Damaged Product"),
    ("item was broken on delivery", "Damaged Product"),
    ("received damaged product", "Damaged Product"),
    ("package is damaged", "Damaged Product"),
    ("product was damaged during delivery", "Damaged Product"),
    ("item arrived broken", "Damaged Product"),
    ("package arrived damaged", "Damaged Product"),


    # ---------------- WRONG PRODUCT ----------------

    ("wrong product was delivered", "Wrong Product"),
    ("received the wrong item", "Wrong Product"),
    ("incorrect product delivered", "Wrong Product"),
    ("wrong item received", "Wrong Product"),
    ("order contains wrong product", "Wrong Product"),
    ("different product was delivered", "Wrong Product"),


    # ---------------- REFUND ISSUE ----------------

    ("refund has not arrived", "Refund Issue"),
    ("money has not been refunded", "Refund Issue"),
    ("refund is delayed", "Refund Issue"),
    ("i want a refund", "Refund Issue"),
    ("order was cancelled but refund missing", "Refund Issue"),
    ("refund money is missing", "Refund Issue"),
    ("refund has not been received", "Refund Issue"),


    # ---------------- DELIVERY DELAY ----------------

    ("delivery is late", "Delivery Delay"),
    ("order has not arrived", "Delivery Delay"),
    ("package delivery delayed", "Delivery Delay"),
    ("delivery is taking too long", "Delivery Delay"),
    ("my order is late", "Delivery Delay"),
    ("package has not arrived yet", "Delivery Delay"),
    ("delivery is delayed", "Delivery Delay"),


    # ---------------- PRODUCT QUALITY ----------------

    ("product quality is poor", "Product Quality Issue"),
    ("bad quality product", "Product Quality Issue"),
    ("quality is not as expected", "Product Quality Issue"),
    ("product stopped working after purchase", "Product Quality Issue"),
    ("product quality is bad", "Product Quality Issue"),
    ("product is not good quality", "Product Quality Issue"),
    ("quality of product is poor", "Product Quality Issue"),
]


# Separate training texts and labels
TRAINING_TEXTS = [item[0] for item in TRAINING_DATA]
TRAINING_LABELS = [item[1] for item in TRAINING_DATA]


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2)
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])


# Train model
model.fit(
    TRAINING_TEXTS,
    TRAINING_LABELS
)


# ============================================================
# DEPARTMENT MAPPING
# ============================================================

DEPARTMENT_MAP = {

    "Battery Issue":
        "Electronics Support",

    "Hardware Issue":
        "Technical Support",

    "Screen/Display Issue":
        "Technical Support",

    "Connectivity Issue":
        "Technical Support",

    "Damaged Product":
        "Returns & Replacement",

    "Wrong Product":
        "Returns & Replacement",

    "Refund Issue":
        "Billing & Payments",

    "Delivery Delay":
        "Logistics & Delivery",

    "Product Quality Issue":
        "Quality & Customer Care"
}


# ============================================================
# RESPONSE MAPPING
# ============================================================

RESPONSE_MAP = {

    "Battery Issue":
        "Please inspect the battery and device power system. Our technical team should review the issue.",

    "Hardware Issue":
        "Please arrange a technical inspection. The hardware support team should review the device.",

    "Screen/Display Issue":
        "Please arrange a display inspection and replacement assessment.",

    "Connectivity Issue":
        "Please check the network settings and route the complaint to technical support.",

    "Damaged Product":
        "Please provide the order details so the returns team can arrange a replacement or resolution.",

    "Wrong Product":
        "Please verify the order and delivered item so the returns team can arrange the correct product.",

    "Refund Issue":
        "Please verify the payment and refund transaction with the billing team.",

    "Delivery Delay":
        "Please verify the shipment status and provide the latest delivery update.",

    "Product Quality Issue":
        "Please review the product quality concern and route it to customer care for resolution."
}


# ============================================================
# PRIORITY
# ============================================================

HIGH_PRIORITY = {
    "Battery Issue",
    "Hardware Issue",
    "Screen/Display Issue"
}


MEDIUM_PRIORITY = {
    "Damaged Product",
    "Wrong Product",
    "Refund Issue",
    "Connectivity Issue"
}


def get_priority(category, complaint):

    text = complaint.lower()

    urgent_words = [

        "fire",
        "smoke",
        "burn",
        "explosion",
        "electric shock",
        "danger",
        "sparking",
        "overheating",
        "motherboard"
    ]

    if (
        category in HIGH_PRIORITY
        or any(word in text for word in urgent_words)
    ):
        return "High"

    if category in MEDIUM_PRIORITY:
        return "Medium"

    return "Low"


# ============================================================
# FRONTEND ROUTES
# ============================================================

@app.get("/")
def home():

    return send_from_directory(
        ".",
        "login.html"
    )


@app.get("/login.html")
def login_page():

    return send_from_directory(
        ".",
        "login.html"
    )


@app.get("/index.html")
def index_page():

    return send_from_directory(
        ".",
        "index.html"
    )


# ============================================================
# PREDICTION API
# ============================================================

@app.post("/predict")
def predict():

    data = request.get_json(
        silent=True
    )


    if not data:

        return jsonify({
            "error":
                "No JSON data received."
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

        age = int(
            data["age"]
        )

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


    # Complaint text
    complaint = str(
        data["complaint_text"]
    ).strip()


    if len(complaint) < 5:

        return jsonify({
            "error":
                "Complaint description is too short."
        }), 400


    # ========================================================
    # ML PREDICTION
    # ========================================================

    category = model.predict(
        [complaint]
    )[0]


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

        "Please review the complaint and assign it to the appropriate support team."
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
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    import os

    app.run(

        host="0.0.0.0",

        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        )
    )
