from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import tensorflow as tf

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("machine_failure_model.keras")

# Load scaler
scaler = joblib.load("machine_scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get sensor values from frontend
        temperature = float(request.form["temperature"])
        vibration = float(request.form["vibration"])
        pressure = float(request.form["pressure"])
        rpm = float(request.form["rpm"])
        current = float(request.form["current"])

        # Create input array
        sensor_data = np.array([
            [temperature, vibration, pressure, rpm, current]
        ])

        # Scale input using the same scaler used during training
        sensor_scaled = scaler.transform(sensor_data)

        # LSTM requires 20 time steps
        sequence_length = 20

        sequence = np.repeat(
            sensor_scaled[np.newaxis, :, :],
            sequence_length,
            axis=1
        )

        # Make prediction
        prediction = model.predict(sequence, verbose=0)[0][0]

        failure_probability = float(prediction * 100)

        # Calculate machine health
        health_score = max(0, 100 - failure_probability)

        # Determine risk
        if failure_probability >= 70:
            risk_level = "HIGH"
            recommendation = (
                "Immediate maintenance recommended. "
                "Inspect the machine before continued operation."
            )

        elif failure_probability >= 40:
            risk_level = "MODERATE"
            recommendation = (
                "Monitor the machine closely and schedule maintenance."
            )

        else:
            risk_level = "LOW"
            recommendation = (
                "Machine condition appears normal. "
                "Continue regular monitoring."
            )

        return render_template(
            "index.html",
            prediction=round(failure_probability, 2),
            health=round(health_score, 2),
            risk=risk_level,
            recommendation=recommendation,
            temperature=temperature,
            vibration=vibration,
            pressure=pressure,
            rpm=rpm,
            current=current
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)