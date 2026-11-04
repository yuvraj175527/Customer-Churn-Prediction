
const form = document.getElementById("predictionForm");

form.addEventListener("submit", async function(e) {
    e.preventDefault();

    const button = document.getElementById("predictBtn");
    const result = document.getElementById("result");

    button.disabled = true;
    button.textContent = "Predicting...";

    const data = Object.fromEntries(new FormData(form));

    data.SeniorCitizen = Number(data.SeniorCitizen);
    data.tenure = Number(data.tenure);
    data.MonthlyCharges = Number(data.MonthlyCharges);
    data.TotalCharges = Number(data.TotalCharges);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const output = await response.json();

        if (!response.ok) {
            throw new Error(
                typeof output.detail === "string"
                    ? output.detail
                    : JSON.stringify(output.detail)
            );
        }

        document.getElementById("prediction").textContent =
            "Prediction: " + output.prediction;

        document.getElementById("probability").textContent =
            "Churn Probability: " + output.churn_probability + "%";

        document.getElementById("risk").textContent =
            "Risk Level: " + output.risk_level;

        result.classList.remove("hidden");

        result.scrollIntoView({ behavior: "smooth" });

    } catch (error) {
        alert("Prediction failed: " + error.message);
    } finally {
        button.disabled = false;
        button.textContent = "Predict Customer Churn";
    }
});
