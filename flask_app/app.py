from flask import Flask, render_template, request
import requests
import time
import logging
import os

app = Flask(__name__)

# Define FastAPI endpoint
FASTAPI_URL = "http://fastapi:8000/predict/"


# Ensure log directory exists
log_dir = "flask_app/logs"
os.makedirs(log_dir, exist_ok=True)

# Configure Logging
logging.basicConfig(
    filename=os.path.join(log_dir, "app_logs.log"),  # Store logs in a proper directory
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Flask app started successfully!")


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        start_time = time.time()  # Track request time

        try:
            # Collect user input data
            data = {
                "Bedroom": float(request.form["Bedroom"]),
                "Space": float(request.form["Space"]),
                "Room": float(request.form["Room"]),
                "Lot": float(request.form["Lot"]),
                "Tax": float(request.form["Tax"]),
                "Bathroom": float(request.form["Bathroom"]),
                "Garage": float(request.form["Garage"]),
                "Condition": float(request.form["Condition"]),
            }

            # Send request to FastAPI
            response = requests.post(FASTAPI_URL, json=data)
            prediction = response.json()["Predicted Price"]

            # Format prediction
            formatted_prediction = f"{prediction}M"

            # Log the request, response, and execution time
            execution_time = round(time.time() - start_time, 3)
            logging.info(
                f"User Input: {data} | Prediction: {formatted_prediction} | Response Time: {execution_time}s"
            )

            return render_template("index.html", prediction=formatted_prediction)

        except Exception as e:
            error_message = str(e)
            logging.error(f"Error processing request: {error_message}")

            return render_template("index.html", error=error_message)

    return render_template("index.html", prediction=None, error=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
