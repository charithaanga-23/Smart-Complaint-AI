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

    const data = {
        gender: document.getElementById("gender").value,
        age: Number(document.getElementById("age").value),
        product_category: document.getElementById("product_category").value,
        complaint_text: document.getElementById("complaint_text").value.trim()
    };

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Backend error");
        }

        document.getElementById("category").textContent =
            result.complaint_category;

        document.getElementById("priority").textContent =
            result.priority;

        document.getElementById("department").textContent =
            result.department;

        document.getElementById("response").textContent =
            result.suggested_response;

        resultSection.classList.remove("hidden");

    } catch (error) {
        errorBox.textContent = "Error: " + error.message;
    }

    submitBtn.disabled = false;
    submitBtn.textContent = "Analyze Complaint";
});