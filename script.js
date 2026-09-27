const API_URL = "/predict";

const form = document.getElementById("complaintForm");
const submitBtn = document.getElementById("submitBtn");

const errorBox = document.getElementById("error");
const resultSection = document.getElementById("result");


// ===============================
// ANALYZE COMPLAINT
// ===============================

form.addEventListener("submit", async function (event) {

    event.preventDefault();

    errorBox.textContent = "";

    submitBtn.disabled = true;
    submitBtn.textContent = "Analyzing...";


    // Get values from the new frontend
    const data = {

        gender:
            document.getElementById("gender").value,

        age:
            Number(document.getElementById("age").value),

        product_category:
            document.getElementById("product_category").value,

        complaint_text:
            document.getElementById("complaint_text").value.trim()

    };


    // Basic validation
    if (!data.gender) {

        errorBox.textContent =
            "Please select gender.";

        submitBtn.disabled = false;
        submitBtn.textContent = "Analyze Complaint";

        return;
    }


    if (!data.age) {

        errorBox.textContent =
            "Please enter age.";

        submitBtn.disabled = false;
        submitBtn.textContent = "Analyze Complaint";

        return;
    }


    if (!data.product_category) {

        errorBox.textContent =
            "Please select product category.";

        submitBtn.disabled = false;
        submitBtn.textContent = "Analyze Complaint";

        return;
    }


    if (!data.complaint_text) {

        errorBox.textContent =
            "Please enter your complaint.";

        submitBtn.disabled = false;
        submitBtn.textContent = "Analyze Complaint";

        return;
    }


    try {

        // Send complaint to Flask backend
        const response = await fetch(API_URL, {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(data)

        });


        const result = await response.json();


        if (!response.ok) {

            throw new Error(
                result.error || "Backend error"
            );

        }


        // Display AI results
        document.getElementById("category").textContent =
            result.complaint_category;

        document.getElementById("priority").textContent =
            result.priority;

        document.getElementById("department").textContent =
            result.department;

        document.getElementById("response").textContent =
            result.suggested_response;


        // Show results
        resultSection.classList.remove("hidden");


        // Scroll to results
        resultSection.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });


    }

    catch (error) {

        errorBox.textContent =
            "Error: " + error.message;

    }


    submitBtn.disabled = false;
    submitBtn.textContent = "Analyze Complaint";

});


// ===============================
// ANALYZE ANOTHER COMPLAINT
// ===============================

document
    .getElementById("newComplaint")
    .addEventListener("click", function () {

        // Clear all form fields
        document.getElementById("customerName").value = "";
        document.getElementById("phoneNumber").value = "";
        document.getElementById("preferenceId").value = "";

        document.getElementById("gender").value = "";
        document.getElementById("age").value = "";

        document.getElementById("product_category").value = "";

        document.getElementById("complaint_text").value = "";


        // Reset character counter
        const count =
            document.getElementById("count");

        if (count) {
            count.textContent = "0";
        }


        // Hide previous result
        resultSection.classList.add("hidden");


        // Clear error
        errorBox.textContent = "";


        // Scroll to top
        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });

    });
