from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import re


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)
CORS(app)


# =========================================================
# TRAINING DATA
# =========================================================

training_data = [

    # =====================================================
    # BATTERY
    # =====================================================

    ("my battery is draining very quickly", "Battery Issue"),
    ("phone battery drains fast", "Battery Issue"),
    ("battery life is very poor", "Battery Issue"),
    ("battery drains quickly", "Battery Issue"),
    ("my phone battery is getting low quickly", "Battery Issue"),
    ("battery does not last long", "Battery Issue"),
    ("my battery drains too fast", "Battery Issue"),
    ("battery is overheating", "Battery Issue"),
    ("battery is swollen", "Battery Issue"),

    # =====================================================
    # CHARGING
    # =====================================================

    ("my phone is not charging", "Charging Issue"),
    ("charger is not working", "Charging Issue"),
    ("charging is very slow", "Charging Issue"),
    ("device does not charge", "Charging Issue"),
    ("phone takes too long to charge", "Charging Issue"),
    ("charging stopped working", "Charging Issue"),
    ("my charger does not work", "Charging Issue"),

    # =====================================================
    # SCREEN / DISPLAY
    # =====================================================

    ("my screen is broken", "Screen/Display Issue"),
    ("screen is cracked", "Screen/Display Issue"),
    ("display is not working", "Screen/Display Issue"),
    ("phone screen is black", "Screen/Display Issue"),
    ("screen is flickering", "Screen/Display Issue"),
    ("display is flickering", "Screen/Display Issue"),
    ("screen has lines", "Screen/Display Issue"),
    ("my display is damaged", "Screen/Display Issue"),

    # =====================================================
    # HARDWARE
    # =====================================================

    ("my phone is not turning on", "Hardware Issue"),
    ("device keeps shutting down", "Hardware Issue"),
    ("hardware is not working", "Hardware Issue"),
    ("phone suddenly stopped working", "Hardware Issue"),
    ("device has a hardware problem", "Hardware Issue"),
    ("my device stopped working", "Hardware Issue"),
    ("phone does not turn on", "Hardware Issue"),

    # =====================================================
    # SOFTWARE
    # =====================================================

    ("application keeps crashing", "Software Issue"),
    ("software is not working", "Software Issue"),
    ("phone is freezing", "Software Issue"),
    ("app crashes frequently", "Software Issue"),
    ("system is very slow", "Software Issue"),
    ("application is not responding", "Software Issue"),
    ("software keeps crashing", "Software Issue"),

    # =====================================================
    # CONNECTIVITY
    # =====================================================

    ("wifi is not connecting", "Connectivity Issue"),
    ("bluetooth is not working", "Connectivity Issue"),
    ("internet connection is not working", "Connectivity Issue"),
    ("mobile network is not available", "Connectivity Issue"),
    ("my phone cannot connect to wifi", "Connectivity Issue"),
    ("network connection is poor", "Connectivity Issue"),
    ("internet is not working", "Connectivity Issue"),

    # =====================================================
    # DAMAGED PRODUCT
    # =====================================================

    ("product arrived damaged", "Damaged Product"),
    ("package arrived broken", "Damaged Product"),
    ("item was damaged during delivery", "Damaged Product"),
    ("received a damaged product", "Damaged Product"),
    ("product is physically damaged", "Damaged Product"),
    ("package was damaged", "Damaged Product"),
    ("item arrived broken", "Damaged Product"),

    # =====================================================
    # WRONG PRODUCT
    # =====================================================

    ("I received the wrong product", "Wrong Product"),
    ("wrong item was delivered", "Wrong Product"),
    ("I received a different product", "Wrong Product"),
    ("the item I received is not what I ordered", "Wrong Product"),
    ("wrong product was sent", "Wrong Product"),
    ("I got the wrong item", "Wrong Product"),
    ("product received is different from my order", "Wrong Product"),

    # =====================================================
    # REFUND
    # =====================================================

    ("my refund has not arrived", "Refund Issue"),
    ("refund is still pending", "Refund Issue"),
    ("I have not received my refund", "Refund Issue"),
    ("refund was not processed", "Refund Issue"),
    ("where is my refund", "Refund Issue"),
    ("refund is delayed", "Refund Issue"),
    ("I am waiting for my refund", "Refund Issue"),

    # =====================================================
    # PAYMENT
    # =====================================================

    ("payment failed", "Payment Issue"),
    ("money was deducted but order failed", "Payment Issue"),
    ("payment is not going through", "Payment Issue"),
    ("I was charged twice", "Payment Issue"),
    ("payment problem", "Payment Issue"),
    ("my payment failed", "Payment Issue"),
    ("money was deducted", "Payment Issue"),

    # =====================================================
    # DELIVERY
    # =====================================================

    ("my package has not arrived yet", "Delivery Delay"),
    ("delivery is late", "Delivery Delay"),
    ("order has not arrived", "Delivery Delay"),
    ("my package is delayed", "Delivery Delay"),
    ("delivery is taking too long", "Delivery Delay"),
    ("my order is late", "Delivery Delay"),
    ("package delivery is delayed", "Delivery Delay"),

    # =====================================================
    # ORDER CANCELLATION
    # =====================================================

    ("I want to cancel my order", "Order Cancellation"),
    ("please cancel my order", "Order Cancellation"),
    ("cancel the product I ordered", "Order Cancellation"),
    ("I need to cancel my order", "Order Cancellation"),

    # =====================================================
    # REPLACEMENT
    # =====================================================

    ("I want a replacement", "Replacement Request"),
    ("please replace my product", "Replacement Request"),
    ("I need a replacement item", "Replacement Request"),
    ("replace the damaged product", "Replacement Request"),
    ("I want to replace the product", "Replacement Request"),

    # =====================================================
    # WARRANTY
    # =====================================================

    ("my product is under warranty", "Warranty Issue"),
    ("I want to claim warranty", "Warranty Issue"),
    ("warranty service is required", "Warranty Issue"),
    ("I need warranty support", "Warranty Issue"),

    # =====================================================
    # PRODUCT QUALITY / COLOR / APPEARANCE
    # =====================================================

    ("the color is different from the picture", "Product Quality Issue"),
    ("the colour is different from the picture", "Product Quality Issue"),
    ("the product color does not match the image", "Product Quality Issue"),
    ("the product colour does not match the image", "Product Quality Issue"),
    ("I received a different color than shown online", "Product Quality Issue"),
    ("the color of the product is not as advertised", "Product Quality Issue"),
    ("the product looks different from the picture", "Product Quality Issue"),
    ("the product looks different from the image", "Product Quality Issue"),
    ("the item appearance is different from the listing", "Product Quality Issue"),
    ("the received product does not match the product image", "Product Quality Issue"),
    ("the product color is wrong", "Product Quality Issue"),
    ("color does not match the website", "Product Quality Issue"),
    ("colour does not match the website", "Product Quality Issue"),
    ("product looks different than shown", "Product Quality Issue"),
    ("product quality is poor", "Product Quality Issue"),
    ("quality of the product is not good", "Product Quality Issue"),
    ("the product is not as advertised", "Product Quality Issue"),
    ("item is different from what was shown", "Product Quality Issue"),
    ("product does not look like the picture", "Product Quality Issue"),
    ("product appearance does not match", "Product Quality Issue"),
    ("the product is different from the image", "Product Quality Issue"),

    # =====================================================
    # ACCOUNT / LOGIN
    # =====================================================

    ("I cannot login to my account", "Account/Login Issue"),
    ("login is not working", "Account/Login Issue"),
    ("I forgot my password", "Account/Login Issue"),
    ("my account is not accessible", "Account/Login Issue"),
    ("I cannot access my account", "Account/Login Issue"),
    ("unable to login", "Account/Login Issue"),

    # =====================================================
    # CUSTOMER SERVICE
    # =====================================================

    ("customer service is not responding", "Customer Service Issue"),
    ("support team has not replied", "Customer Service Issue"),
    ("I need help from customer support", "Customer Service Issue"),
    ("customer support is not responding", "Customer Service Issue"),

    # =====================================================
    # SUBSCRIPTION
    # =====================================================

    ("I want to cancel my subscription", "Subscription Issue"),
    ("subscription was charged incorrectly", "Subscription Issue"),
    ("my subscription is not working", "Subscription Issue"),
    ("subscription problem", "Subscription Issue"),

    # =====================================================
    # SECURITY
    # =====================================================

    ("someone accessed my account", "Security Issue"),
    ("I think my account was hacked", "Security Issue"),
    ("there is suspicious activity on my account", "Security Issue"),
    ("my account has been hacked", "Security Issue"),
    ("unauthorized access to my account", "Security Issue"),

    # =====================================================
    # GENERAL
    # =====================================================

    ("I have a problem with my product", "General Product Complaint"),
    ("there is an issue with my order", "General Product Complaint"),
    ("I am not satisfied with my product", "General Product Complaint"),
    ("I have a complaint about my product", "General Product Complaint"),
]


# =========================================================
# MACHINE LEARNING MODEL
# =========================================================

texts = [item[0] for item in training_data]
labels = [item[1] for item in training_data]


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


model.fit(texts, labels)


# =========================================================
# DEPARTMENT MAPPING
# =========================================================

department_mapping = {

    "Battery Issue": "Electronics Support",
    "Charging Issue": "Electronics Support",

    "Screen/Display Issue": "Technical Support",
    "Hardware Issue": "Technical Support",
    "Software Issue": "Technical Support",
    "Connectivity Issue": "Technical Support",

    "Damaged Product": "Returns & Replacement",
    "Wrong Product": "Returns & Replacement",
    "Replacement Request": "Returns & Replacement",

    "Refund Issue": "Billing & Payments",
    "Payment Issue": "Billing & Payments",

    "Delivery Delay": "Logistics & Delivery",
    "Order Cancellation": "Order Management",

    "Warranty Issue": "Warranty Support",

    "Product Quality Issue": "Customer Support",

    "Account/Login Issue": "Account Support",
    "Customer Service Issue": "Customer Support",

    "Subscription Issue": "Billing & Payments",

    "Security Issue": "Security Support",

    "General Product Complaint": "Customer Support",

    "Invalid Complaint": "Customer Support"
}


# =========================================================
# NORMALIZE TEXT
# =========================================================

def clean_text(text):

    text = str(text).lower().strip()

    text = re.sub(r"\s+", " ", text)

    return text


# =========================================================
# INVALID / RANDOM INPUT DETECTION
# =========================================================

def is_valid_complaint(complaint):

    text = clean_text(complaint)

    # Remove punctuation for easier checking
    words = re.findall(r"[a-zA-Z]+", text)

    if not words:
        return False

    # Reject extremely short random text
    if len(text) < 5:
        return False

    # -----------------------------------------------------
    # Product / service related keywords
    # -----------------------------------------------------

    complaint_keywords = [

        # Products / devices
        "product",
        "item",
        "device",
        "phone",
        "mobile",
        "computer",
        "laptop",
        "tablet",
        "charger",
        "battery",
        "screen",
        "display",
        "hardware",
        "software",
        "application",
        "app",

        # Connectivity
        "wifi",
        "wi-fi",
        "internet",
        "network",
        "bluetooth",
        "connection",
        "connect",

        # Orders / delivery
        "order",
        "package",
        "delivery",
        "delivered",
        "shipping",
        "received",
        "arrived",

        # Payments / refunds
        "payment",
        "paid",
        "refund",
        "money",
        "transaction",
        "charged",
        "billing",

        # Problems
        "broken",
        "damaged",
        "cracked",
        "faulty",
        "defective",
        "problem",
        "issue",
        "complaint",
        "not working",
        "doesn't work",
        "does not work",
        "failed",
        "failure",
        "poor quality",
        "quality",
        "wrong",

        # Requests
        "replace",
        "replacement",
        "cancel",
        "cancellation",
        "warranty",
        "support",
        "customer service",

        # Account
        "account",
        "login",
        "password",
        "access",

        # Subscription
        "subscription",

        # Security
        "hacked",
        "security",
        "unauthorized",
        "suspicious",

        # Safety
        "fire",
        "smoke",
        "burning",
        "spark",
        "sparking",
        "overheating"
    ]

    # -----------------------------------------------------
    # Known complaint keyword
    # -----------------------------------------------------

    if any(keyword in text for keyword in complaint_keywords):
        return True

    # -----------------------------------------------------
    # Common general complaint phrases
    # -----------------------------------------------------

    general_phrases = [

        "not satisfied",
        "need help",
        "need assistance",
        "want help",
        "customer complaint",
        "service problem",
        "service issue",
        "having trouble",
        "having a problem",
        "something is wrong"
    ]

    if any(phrase in text for phrase in general_phrases):
        return True

    # -----------------------------------------------------
    # Detect obvious gibberish
    # -----------------------------------------------------

    # If a single long word has no vowels, it is likely random.
    if len(words) == 1:

        word = words[0].lower()

        if len(word) >= 7 and not re.search(r"[aeiou]", word):
            return False

    # If there are no recognized complaint terms,
    # treat the input as unrelated.
    return False


# =========================================================
# RULE-BASED CATEGORY DETECTION
# =========================================================

def apply_specific_rules(complaint):

    text = clean_text(complaint)

    # -----------------------------------------------------
    # SECURITY
    # -----------------------------------------------------

    security_words = [
        "hacked",
        "hack",
        "suspicious activity",
        "unauthorized access",
        "someone accessed my account",
        "account hacked"
    ]

    if any(word in text for word in security_words):
        return "Security Issue"


    # -----------------------------------------------------
    # FIRE / SMOKE / ELECTRICAL
    # -----------------------------------------------------

    safety_words = [
        "fire",
        "smoke",
        "burning smell",
        "burning",
        "explosion",
        "electric shock",
        "sparking",
        "spark"
    ]

    if any(word in text for word in safety_words):

        if any(word in text for word in [
            "battery",
            "phone",
            "device",
            "charger",
            "charging"
        ]):
            return "Hardware Issue"


    # -----------------------------------------------------
    # BATTERY
    # -----------------------------------------------------

    battery_words = [
        "battery draining",
        "battery drains",
        "battery drain",
        "battery life",
        "battery is low",
        "battery problem",
        "battery issue",
        "battery swollen",
        "swollen battery",
        "battery overheating",
        "battery overheat"
    ]

    if any(word in text for word in battery_words):
        return "Battery Issue"


    # -----------------------------------------------------
    # CHARGING
    # -----------------------------------------------------

    charging_words = [
        "not charging",
        "does not charge",
        "doesn't charge",
        "charger not working",
        "charger is not working",
        "charging problem",
        "charging issue",
        "slow charging",
        "charging slowly"
    ]

    if any(word in text for word in charging_words):
        return "Charging Issue"


    # -----------------------------------------------------
    # SCREEN / DISPLAY
    # -----------------------------------------------------

    screen_words = [
        "screen",
        "display",
        "flickering",
        "flicker",
        "cracked screen",
        "black screen"
    ]

    if any(word in text for word in screen_words):
        return "Screen/Display Issue"


    # -----------------------------------------------------
    # SOFTWARE
    # -----------------------------------------------------

    software_words = [
        "software",
        "application",
        "app",
        "crashing",
        "crashes",
        "freezing",
        "frozen",
        "not responding",
        "system is slow",
        "phone is slow"
    ]

    if any(word in text for word in software_words):
        return "Software Issue"


    # -----------------------------------------------------
    # CONNECTIVITY
    # -----------------------------------------------------

    connectivity_words = [
        "wifi",
        "wi-fi",
        "bluetooth",
        "internet",
        "network",
        "connection",
        "cannot connect",
        "can't connect",
        "not connecting"
    ]

    if any(word in text for word in connectivity_words):
        return "Connectivity Issue"


    # -----------------------------------------------------
    # HARDWARE
    # -----------------------------------------------------

    hardware_words = [
        "hardware",
        "not turning on",
        "does not turn on",
        "doesn't turn on",
        "won't turn on",
        "device stopped working",
        "phone stopped working",
        "completely stopped working"
    ]

    if any(word in text for word in hardware_words):
        return "Hardware Issue"


    # -----------------------------------------------------
    # COLOR / IMAGE / APPEARANCE
    # -----------------------------------------------------

    appearance_words = [
        "color is different",
        "colour is different",
        "different color",
        "different colour",
        "color does not match",
        "colour does not match",
        "does not match the picture",
        "doesn't match the picture",
        "does not match the image",
        "doesn't match the image",
        "different from the picture",
        "different from the image",
        "different than shown",
        "different from what was shown",
        "looks different",
        "look different",
        "appearance is different",
        "not as shown",
        "not as advertised",
        "product quality",
        "poor quality",
        "bad quality",
        "quality is not good"
    ]

    if any(word in text for word in appearance_words):
        return "Product Quality Issue"


    # -----------------------------------------------------
    # WRONG PRODUCT
    # -----------------------------------------------------

    wrong_product_words = [
        "wrong product",
        "wrong item",
        "different product",
        "different item",
        "not what i ordered"
    ]

    if any(word in text for word in wrong_product_words):
        return "Wrong Product"


    # -----------------------------------------------------
    # DAMAGED PRODUCT
    # -----------------------------------------------------

    damaged_words = [
        "arrived damaged",
        "received damaged",
        "product is damaged",
        "item is damaged",
        "package is damaged",
        "arrived broken",
        "received broken",
        "physically damaged"
    ]

    if any(word in text for word in damaged_words):
        return "Damaged Product"


    # -----------------------------------------------------
    # REFUND
    # -----------------------------------------------------

    refund_words = [
        "refund",
        "money back",
        "refund pending",
        "refund delayed",
        "refund not received"
    ]

    if any(word in text for word in refund_words):
        return "Refund Issue"


    # -----------------------------------------------------
    # PAYMENT
    # -----------------------------------------------------

    payment_words = [
        "payment failed",
        "payment problem",
        "payment issue",
        "charged twice",
        "money was deducted",
        "transaction failed"
    ]

    if any(word in text for word in payment_words):
        return "Payment Issue"


    # -----------------------------------------------------
    # DELIVERY
    # -----------------------------------------------------

    delivery_words = [
        "delivery is late",
        "delivery is delayed",
        "package is delayed",
        "package has not arrived",
        "order has not arrived",
        "order is late",
        "delivery delay"
    ]

    if any(word in text for word in delivery_words):
        return "Delivery Delay"


    # -----------------------------------------------------
    # ORDER CANCELLATION
    # -----------------------------------------------------

    cancellation_words = [
        "cancel my order",
        "cancel the order",
        "cancel my product",
        "order cancellation",
        "want to cancel"
    ]

    if any(word in text for word in cancellation_words):
        return "Order Cancellation"


    # -----------------------------------------------------
    # REPLACEMENT
    # -----------------------------------------------------

    replacement_words = [
        "replacement",
        "replace the product",
        "replace my product",
        "want to replace"
    ]

    if any(word in text for word in replacement_words):
        return "Replacement Request"


    # -----------------------------------------------------
    # WARRANTY
    # -----------------------------------------------------

    warranty_words = [
        "warranty",
        "warranty claim",
        "claim warranty"
    ]

    if any(word in text for word in warranty_words):
        return "Warranty Issue"


    # -----------------------------------------------------
    # LOGIN
    # -----------------------------------------------------

    login_words = [
        "cannot login",
        "can't login",
        "login problem",
        "login issue",
        "unable to login",
        "forgot password",
        "cannot access my account"
    ]

    if any(word in text for word in login_words):
        return "Account/Login Issue"


    # -----------------------------------------------------
    # CUSTOMER SERVICE
    # -----------------------------------------------------

    service_words = [
        "customer service",
        "customer support",
        "support team",
        "support is not responding"
    ]

    if any(word in text for word in service_words):
        return "Customer Service Issue"


    # -----------------------------------------------------
    # SUBSCRIPTION
    # -----------------------------------------------------

    subscription_words = [
        "subscription",
        "subscription problem",
        "subscription issue",
        "cancel subscription"
    ]

    if any(word in text for word in subscription_words):
        return "Subscription Issue"


    # -----------------------------------------------------
    # GENERAL PRODUCT COMPLAINT
    # -----------------------------------------------------

    general_words = [
        "problem with my product",
        "issue with my product",
        "complaint about my product",
        "not satisfied with my product",
        "problem with my order",
        "issue with my order"
    ]

    if any(word in text for word in general_words):
        return "General Product Complaint"


    return None


# =========================================================
# PRIORITY DETECTION
# =========================================================

def determine_priority(category, complaint):

    text = clean_text(complaint)

    # =====================================================
    # HIGH PRIORITY
    # =====================================================

    high_words = [
        "fire",
        "smoke",
        "burning",
        "burn",
        "explosion",
        "electric shock",
        "electricity",
        "sparking",
        "spark",
        "overheating",
        "overheated",
        "very hot",
        "extremely hot",
        "danger",
        "dangerous",
        "safety issue",
        "unsafe",
        "battery exploding",
        "battery swollen",
        "swollen battery",
        "urgent",
        "emergency"
    ]

    if any(word in text for word in high_words):
        return "High"


    # =====================================================
    # SERIOUS DEVICE FAILURE
    # =====================================================

    serious_failure_words = [
        "not turning on",
        "does not turn on",
        "doesn't turn on",
        "won't turn on",
        "completely stopped working",
        "device stopped working",
        "phone stopped working"
    ]

    if any(word in text for word in serious_failure_words):
        return "High"


    # =====================================================
    # HIGH PRIORITY CATEGORIES
    # =====================================================

    if category in {
        "Battery Issue",
        "Charging Issue",
        "Hardware Issue",
        "Security Issue"
    }:
        return "High"


    # =====================================================
    # MEDIUM PRIORITY
    # =====================================================

    medium_words = [
        "broken",
        "cracked",
        "damaged",
        "not working",
        "doesn't work",
        "does not work",
        "failed",
        "failure",
        "wrong product",
        "wrong item",
        "refund",
        "replacement",
        "payment failed",
        "payment problem",
        "cannot login",
        "can't login",
        "login problem",
        "poor quality",
        "bad quality",
        "flickering",
        "slow charging",
        "charging slowly"
    ]

    if any(word in text for word in medium_words):
        return "Medium"


    # =====================================================
    # MEDIUM CATEGORIES
    # =====================================================

    if category in {
        "Screen/Display Issue",
        "Damaged Product",
        "Wrong Product",
        "Refund Issue",
        "Payment Issue",
        "Replacement Request",
        "Warranty Issue",
        "Delivery Delay",
        "Order Cancellation"
    }:
        return "Medium"


    return "Low"


# =========================================================
# SUGGESTED RESPONSE
# =========================================================

def generate_response(category, complaint):

    text = clean_text(complaint)


    # -----------------------------------------------------
    # INVALID COMPLAINT
    # -----------------------------------------------------

    if category == "Invalid Complaint":

        return (
            "Please enter a valid complaint related to a "
            "product or service so that the AI system can "
            "analyze it correctly."
        )


    # -----------------------------------------------------
    # COLOR / APPEARANCE
    # -----------------------------------------------------

    if category == "Product Quality Issue":

        if any(word in text for word in [
            "color",
            "colour",
            "picture",
            "image",
            "shown",
            "appearance"
        ]):

            return (
                "The product appearance or color does not match "
                "the displayed product information. Our customer "
                "support team should review the product details "
                "and assist with the appropriate resolution."
            )

        return (
            "The product appears to have a quality-related issue. "
            "Our customer support team should review the product "
            "details and assist with the appropriate resolution."
        )


    # -----------------------------------------------------
    # BATTERY
    # -----------------------------------------------------

    if category == "Battery Issue":

        return (
            "Please inspect the battery and device power system. "
            "Our technical team should review the issue."
        )


    # -----------------------------------------------------
    # CHARGING
    # -----------------------------------------------------

    if category == "Charging Issue":

        return (
            "Please check the charger, charging port and power "
            "connection. Our technical team can assist with "
            "further inspection."
        )


    # -----------------------------------------------------
    # SCREEN
    # -----------------------------------------------------

    if category == "Screen/Display Issue":

        return (
            "Please inspect the device display for physical or "
            "technical issues. Our technical support team should "
            "review the problem."
        )


    # -----------------------------------------------------
    # HARDWARE
    # -----------------------------------------------------

    if category == "Hardware Issue":

        return (
            "Our technical team should inspect the device hardware "
            "and determine the appropriate repair or replacement."
        )


    # -----------------------------------------------------
    # SOFTWARE
    # -----------------------------------------------------

    if category == "Software Issue":

        return (
            "Please check for software updates and restart the "
            "device. Our technical support team can investigate "
            "further if the issue continues."
        )


    # -----------------------------------------------------
    # CONNECTIVITY
    # -----------------------------------------------------

    if category == "Connectivity Issue":

        return (
            "Please check your network or connection settings. "
            "Our technical support team can help troubleshoot "
            "the connectivity issue."
        )


    # -----------------------------------------------------
    # DAMAGED PRODUCT
    # -----------------------------------------------------

    if category == "Damaged Product":

        return (
            "The product appears to have a physical damage-related "
            "complaint. Please provide the order details so our "
            "returns team can assist."
        )


    # -----------------------------------------------------
    # WRONG PRODUCT
    # -----------------------------------------------------

    if category == "Wrong Product":

        return (
            "The received product does not match the order. "
            "Our returns and replacement team should review the "
            "order and arrange the appropriate resolution."
        )


    # -----------------------------------------------------
    # REFUND
    # -----------------------------------------------------

    if category == "Refund Issue":

        return (
            "Please check the refund status associated with your "
            "order. Our billing team can investigate the pending "
            "or missing refund."
        )


    # -----------------------------------------------------
    # PAYMENT
    # -----------------------------------------------------

    if category == "Payment Issue":

        return (
            "Our billing team should review the payment transaction "
            "and verify whether the payment was successfully processed."
        )


    # -----------------------------------------------------
    # DELIVERY
    # -----------------------------------------------------

    if category == "Delivery Delay":

        return (
            "Please check the latest tracking information for your "
            "order. Our logistics team can investigate the delivery delay."
        )


    # -----------------------------------------------------
    # CANCELLATION
    # -----------------------------------------------------

    if category == "Order Cancellation":

        return (
            "Your cancellation request should be reviewed by the "
            "order management team."
        )


    # -----------------------------------------------------
    # REPLACEMENT
    # -----------------------------------------------------

    if category == "Replacement Request":

        return (
            "Our returns and replacement team should review the "
            "product issue and assist with the replacement process."
        )


    # -----------------------------------------------------
    # WARRANTY
    # -----------------------------------------------------

    if category == "Warranty Issue":

        return (
            "Please provide the product and purchase details so "
            "our warranty team can verify the warranty and assist you."
        )


    # -----------------------------------------------------
    # ACCOUNT
    # -----------------------------------------------------

    if category == "Account/Login Issue":

        return (
            "Please verify your account credentials and try again. "
            "Our account support team can assist if the problem continues."
        )


    # -----------------------------------------------------
    # CUSTOMER SERVICE
    # -----------------------------------------------------

    if category == "Customer Service Issue":

        return (
            "Our customer support team should review the communication "
            "issue and assist you as soon as possible."
        )


    # -----------------------------------------------------
    # SUBSCRIPTION
    # -----------------------------------------------------

    if category == "Subscription Issue":

        return (
            "Our billing team should review your subscription details "
            "and assist with the issue."
        )


    # -----------------------------------------------------
    # SECURITY
    # -----------------------------------------------------

    if category == "Security Issue":

        return (
            "This complaint requires security review. Please avoid "
            "sharing sensitive account information and contact our "
            "security support team."
        )


    # -----------------------------------------------------
    # GENERAL
    # -----------------------------------------------------

    return (
        "Your complaint has been routed to Customer Support "
        "for further review."
    )


# =========================================================
# ROUTES
# =========================================================

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


@app.get("/style.css")
def style_css():

    return send_from_directory(
        ".",
        "style.css"
    )


@app.get("/script.js")
def script_js():

    return send_from_directory(
        ".",
        "script.js"
    )


# =========================================================
# GOOGLE SEARCH CONSOLE VERIFICATION
# =========================================================

@app.get("/google5502be35b740b60.html")
def google_verification():

    return send_from_directory(
        ".",
        "google5502be35b740b60.html"
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return jsonify({

        "status": "ok",

        "model": "Smart Complaint AI",

        "training_examples": len(training_data)
    })


# =========================================================
# PREDICT API
# =========================================================

@app.post("/predict")
def predict():

    data = request.get_json(
        silent=True
    ) or {}


    # =====================================================
    # REQUIRED FIELDS
    # =====================================================

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

        if str(
            data.get(field, "")
        ).strip() == "":

            missing.append(field)


    if missing:

        return jsonify({

            "error":
                "Missing fields: "
                + ", ".join(missing)

        }), 400


    # =====================================================
    # CUSTOMER NAME
    # =====================================================

    customer_name = str(
        data.get("customer_name", "")
    ).strip()


    if len(customer_name) < 2:

        return jsonify({

            "error":
                "Please enter a valid customer name."

        }), 400


    # =====================================================
    # PHONE NUMBER
    # =====================================================

    phone_number = str(
        data.get("phone_number", "")
    ).strip()


    normalized_phone = re.sub(
        r"[\s\-\(\)]",
        "",
        phone_number
    )


    if not re.fullmatch(
        r"\+?[0-9]{7,15}",
        normalized_phone
    ):

        return jsonify({

            "error":
                "Please enter a valid phone number."

        }), 400


    # =====================================================
    # AGE
    # =====================================================

    try:

        age = int(
            data.get("age")
        )

    except (
        ValueError,
        TypeError
    ):

        return jsonify({

            "error":
                "Age must be a valid number."

        }), 400


    if age < 18:

        return jsonify({

            "error":
                "Customer must be 18 years or older."

        }), 400


    if age > 120:

        return jsonify({

            "error":
                "Please enter a valid age."

        }), 400


    # =====================================================
    # COMPLAINT
    # =====================================================

    complaint = str(
        data.get("complaint_text", "")
    ).strip()


    if len(complaint) < 5:

        return jsonify({

            "error":
                "Complaint must contain at least 5 characters."

        }), 400


    if len(complaint) > 1000:

        return jsonify({

            "error":
                "Complaint cannot exceed 1000 characters."

        }), 400


    # =====================================================
    # PRODUCT CATEGORY
    # =====================================================

    product_category = str(
        data.get("product_category", "")
    ).strip()


    # =====================================================
    # STEP 1
    # INPUT RELEVANCE CHECK
    # =====================================================

    if not is_valid_complaint(complaint):

        return jsonify({

            "complaint_category":
                "Invalid Complaint",

            "priority":
                "Low",

            "department":
                "Customer Support",

            "suggested_response":
                "Please enter a valid complaint related to "
                "a product or service so that the AI system "
                "can analyze it correctly."
        })


    # =====================================================
    # STEP 2
    # RULE-BASED CLASSIFICATION
    # =====================================================

    rule_category = apply_specific_rules(
        complaint
    )


    # =====================================================
    # STEP 3
    # MACHINE LEARNING CLASSIFICATION
    # =====================================================

    probabilities = model.predict_proba(
        [complaint]
    )[0]


    classes = model.named_steps[
        "classifier"
    ].classes_


    best_index = probabilities.argmax()


    ml_category = classes[
        best_index
    ]


    confidence = float(
        probabilities[best_index]
    )


    # =====================================================
    # STEP 4
    # FINAL CATEGORY
    # =====================================================

    if rule_category:

        final_category = rule_category

    else:

        final_category = ml_category


    # =====================================================
    # STEP 5
    # ML CONFIDENCE CHECK
    # =====================================================

    # If there is no rule match and the ML model is not
    # sufficiently confident, use the general category
    # only for meaningful complaint text.
    #
    # Random/unrelated text has already been rejected above.

    if (
        not rule_category
        and confidence < 0.20
    ):

        final_category = (
            "General Product Complaint"
        )


    # =====================================================
    # STEP 6
    # DEPARTMENT
    # =====================================================

    department = department_mapping.get(
        final_category,
        "Customer Support"
    )


    # =====================================================
    # STEP 7
    # PRIORITY
    # =====================================================

    priority = determine_priority(
        final_category,
        complaint
    )


    # =====================================================
    # STEP 8
    # SUGGESTED RESPONSE
    # =====================================================

    suggested_response = generate_response(
        final_category,
        complaint
    )


    # =====================================================
    # FINAL JSON RESPONSE
    # =====================================================

    return jsonify({

        "complaint_category":
            final_category,

        "priority":
            priority,

        "department":
            department,

        "suggested_response":
            suggested_response
    })


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    print("")
    print("======================================")
    print("       SMART COMPLAINT AI")
    print("======================================")

    print("Server:")
    print("http://127.0.0.1:5000")

    print("")

    print("Health:")
    print("http://127.0.0.1:5000/health")

    print("======================================")
    print("")


    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
