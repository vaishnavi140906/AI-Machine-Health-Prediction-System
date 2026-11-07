from flask import Flask, render_template, request
import numpy as np
import joblib
import tensorflow as tf
import json
import os
from datetime import datetime


app = Flask(__name__)


# ============================================================
# LOAD V2 MODEL
# ============================================================

model = tf.keras.models.load_model(
    "machine_failure_model_v2.keras"
)


# ============================================================
# LOAD V2 SCALER
# ============================================================

scaler = joblib.load(
    "machine_scaler_v2.pkl"
)


# ============================================================
# LOAD OPTIMIZED FAILURE THRESHOLD
# ============================================================

threshold_data = joblib.load(
    "failure_threshold.pkl"
)

failure_threshold = threshold_data["threshold"]


# ============================================================
# PREDICTION HISTORY FILE
# ============================================================

HISTORY_FILE = "prediction_history.json"


print("V2 model loaded successfully.")
print("V2 scaler loaded successfully.")
print("Failure threshold:", failure_threshold)


# ============================================================
# LOAD HISTORY
# ============================================================

def load_history():

    if not os.path.exists(HISTORY_FILE):
        return []

    try:

        with open(HISTORY_FILE, "r") as file:
            return json.load(file)

    except Exception:

        return []


# ============================================================
# SAVE HISTORY
# ============================================================

def save_history(record):

    history = load_history()

    history.append(record)

    with open(HISTORY_FILE, "w") as file:

        json.dump(
            history,
            file,
            indent=4
        )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    history = load_history()

    return render_template(
        "index.html",
        history=history
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # MACHINE INFORMATION
        # ----------------------------------------------------

        machine_name = request.form[
            "machine_name"
        ]

        machine_id = request.form[
            "machine_id"
        ]


        # ----------------------------------------------------
        # MACHINE PARAMETERS
        # ----------------------------------------------------

        air_temperature = float(
            request.form[
                "air_temperature"
            ]
        )

        process_temperature = float(
            request.form[
                "process_temperature"
            ]
        )

        rpm = float(
            request.form[
                "rpm"
            ]
        )

        torque = float(
            request.form[
                "torque"
            ]
        )

        tool_wear = float(
            request.form[
                "tool_wear"
            ]
        )


        # ----------------------------------------------------
        # CREATE INPUT DATA
        # ----------------------------------------------------

        sensor_data = np.array([

            [
                air_temperature,
                process_temperature,
                rpm,
                torque,
                tool_wear
            ]

        ])


        # ----------------------------------------------------
        # SCALE INPUT
        # ----------------------------------------------------

        sensor_scaled = scaler.transform(
            sensor_data
        )


        # ----------------------------------------------------
        # CREATE 20-TIME-STEP SEQUENCE
        # ----------------------------------------------------

        sequence_length = 20

        sequence = np.repeat(

            sensor_scaled[
                np.newaxis,
                :,
                :
            ],

            sequence_length,

            axis=1

        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            sequence,
            verbose=0
        )[0][0]


        # Convert probability to percentage

        failure_probability = float(
            prediction * 100
        )


        # ----------------------------------------------------
        # FAILURE DECISION
        # ----------------------------------------------------

        if prediction >= failure_threshold:

            prediction_result = "FAILURE"

        else:

            prediction_result = "NO FAILURE"


        # ----------------------------------------------------
        # MACHINE HEALTH
        # ----------------------------------------------------

        health_score = max(
            0,
            100 - failure_probability
        )


        # ----------------------------------------------------
        # RISK LEVEL + AUTOMATIC ALERT
        # ----------------------------------------------------

        if prediction >= failure_threshold:

            risk_level = "HIGH"

            alert_status = "URGENT"

            alert_message = (
                "⚠️ HIGH FAILURE RISK! "
                "Immediate maintenance is required."
            )

            recommendation = (
                "Failure risk detected. "
                "Stop or inspect the machine immediately "
                "and perform maintenance before continued operation."
            )


        elif failure_probability >= 40:

            risk_level = "MODERATE"

            alert_status = "WARNING"

            alert_message = (
                "⚠️ Moderate risk detected. "
                "Maintenance should be scheduled."
            )

            recommendation = (
                "Moderate machine risk detected. "
                "Monitor the machine closely and "
                "schedule maintenance."
            )


        else:

            risk_level = "LOW"

            alert_status = "NORMAL"

            alert_message = (
                "✅ Machine operating normally. "
                "No immediate maintenance required."
            )

            recommendation = (
                "Machine condition appears normal. "
                "Continue regular monitoring."
            )


        # ----------------------------------------------------
        # CREATE HISTORY RECORD
        # ----------------------------------------------------

        record = {

            "date": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "machine_name": machine_name,

            "machine_id": machine_id,

            "air_temperature": air_temperature,

            "process_temperature": process_temperature,

            "rpm": rpm,

            "torque": torque,

            "tool_wear": tool_wear,

            "failure_risk": round(
                failure_probability,
                2
            ),

            "machine_health": round(
                health_score,
                2
            ),

            "risk_level": risk_level,

            "prediction": prediction_result,

            "alert_status": alert_status,

            "alert_message": alert_message

        }


        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        save_history(record)


        # ----------------------------------------------------
        # GET UPDATED HISTORY
        # ----------------------------------------------------

        history = load_history()


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        return render_template(

            "index.html",

            prediction=round(
                failure_probability,
                2
            ),

            prediction_result=prediction_result,

            health=round(
                health_score,
                2
            ),

            risk=round(
                failure_probability,
                2
            ),

            risk_level=risk_level,

            recommendation=recommendation,

            alert_status=alert_status,

            alert_message=alert_message,

            machine_name=machine_name,

            machine_id=machine_id,

            air_temperature=air_temperature,

            process_temperature=process_temperature,

            rpm=rpm,

            torque=torque,

            tool_wear=tool_wear,

            threshold=round(
                failure_threshold * 100,
                2
            ),

            history=history

        )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        print(
            "ERROR:",
            str(e)
        )

        return render_template(

            "index.html",

            error=str(e),

            history=load_history()

        )


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )