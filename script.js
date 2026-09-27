const API_URL = "/predict";

const form = document.getElementById("complaintForm");
const submitBtn = document.getElementById("submitBtn");

const errorBox = document.getElementById("error");
const resultSection = document.getElementById("result");


form.addEventListener("submit", async function (event) {

    event.preventDefault();

    errorBox.textContent = "";

    submitBtn.disabled = true;
    submitBtn.textContent = "Analyzing...";


    // --------------------------------------------------
    // COLLECT FORM DATA
    // --------------------------------------------------

    const customerName =
        document.getElementById("customerName").value.trim();

    const phoneNumber =
        document.getElementById("phoneNumber").value.trim();

    const preferenceId =
        document.getElementById("preferenceId").value.trim();

    const gender =
        document.getElementById("gender").value;

    const age =
        Number(document.getElementById("age").value);

    const productCategory =
        document.getElementById("product_category").value;

    const complaintText =
        document.getElementById("complaint_text").value.trim();


    // --------------------------------------------------
    // NAME VALIDATION
    // --------------------------------------------------

    if (!customerName) {

        errorBox.textContent =
            "Please enter your name.";

        resetButton();
        return;
    }


    if (customerName.length < 2) {

        errorBox.textContent =
            "Please enter a valid name.";

        resetButton();
        return;
    }


    // --------------------------------------------------
    // PHONE VALIDATION
    // --------------------------------------------------

    if (!phoneNumber) {

        errorBox.textContent =
            "Please enter your phone number.";

        resetButton();
        return;
    }


    const cleanPhone =
        phoneNumber.replace(/[\s\-()]/g, "");


    if (!/^\+?[0-9]{7,15}$/.test(cleanPhone)) {

        errorBox.textContent =
            "Please enter a valid phone number.";

        resetButton();
        return;
    }


    // --------------------------------------------------
    // AGE VALIDATION
    // --------------------------------------------------

    if (!age) {

        errorBox.textContent =
            "Please enter your age.";

        resetButton();
        return;
    }


    if (age < 18) {

        errorBox.textContent =
            "Only customers aged 18 or above can submit a complaint.";

        resetButton();
        return;
    }


    if (age > 120) {

        errorBox.textContent =
            "Please enter a valid age.";

        resetButton();
        return;
    }


    // --------------------------------------------------
    // GENDER
    // --------------------------------------------------

    if (!gender) {

        errorBox.textContent =
            "Please select gender.";

        resetButton();
        return;
    }


    // --------------------------------------------------
    // PRODUCT CATEGORY
    // --------------------------------------------------

    if (!productCategory) {

        errorBox.textContent =
            "Please select product category.";

        resetButton();
        return;
    }


    // --------------------------------------------------
    // COMPLAINT
    // --------------------------------------------------

    if (!complaintText) {

        errorBox.textContent =
            "Please enter your complaint.";

        resetButton();
        return;
    }


    if (complaintText.length < 5) {

        errorBox.textContent =
            "Please provide a more detailed complaint.";

        resetButton();
        return;
    }


    if (complaintText.length > 1000) {

        errorBox.textContent =
            "Complaint must be 1000 characters or less.";

        resetButton();
        return;
    }


    // --------------------------------------------------
    // DATA SENT TO FLASK
    // --------------------------------------------------

    const data = {

        customer_name: customerName,

        phone_number: phoneNumber,

        preference_id: preferenceId,

        gender: gender,

        age: age,

        product_category: productCategory,

        complaint_text: complaintText
    };


    // --------------------------------------------------
    // SEND TO BACKEND
    // --------------------------------------------------

    try {

        const response = await fetch(API_URL, {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(data)
        });


        const result =
            await response.json();


        // Backend error

        if (!response.ok) {

            throw new Error(
                result.error || "Backend error"
            );
        }


        // ------------------------------------------------
        // DISPLAY AI RESULT
        // ------------------------------------------------

        document.getElementById("category").textContent =
            result.complaint_category;


        document.getElementById("priority").textContent =
            result.priority;


        document.getElementById("department").textContent =
            result.department;


        document.getElementById("response").textContent =
            result.suggested_response;


        resultSection.classList.remove("hidden");


        // Scroll to result

        resultSection.scrollIntoView({

            behavior: "smooth",

            block: "start"

        });


    } catch (error) {

        errorBox.textContent =
            "Error: " + error.message;
    }


    resetButton();

});


// ------------------------------------------------------
// RESET BUTTON
// ------------------------------------------------------

function resetButton() {

    submitBtn.disabled = false;

    submitBtn.textContent =
        "Analyze Complaint";
}


// ------------------------------------------------------
// NEW COMPLAINT
// ------------------------------------------------------

document
    .getElementById("newComplaint")
    .addEventListener("click", function () {


        document.getElementById("customerName").value = "";

        document.getElementById("phoneNumber").value = "";

        document.getElementById("preferenceId").value = "";

        document.getElementById("gender").value = "";

        document.getElementById("age").value = "";

        document.getElementById("product_category").value = "";

        document.getElementById("complaint_text").value = "";


        // Character counter

        const count =
            document.getElementById("count");


        if (count) {

            count.textContent = "0";
        }


        // Hide result

        resultSection.classList.add("hidden");


        // Clear error

        errorBox.textContent = "";


        // Scroll to top

        window.scrollTo({

            top: 0,

            behavior: "smooth"

        });

    });
