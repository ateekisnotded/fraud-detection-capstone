"""Streamlit interface for the saved fraud-detection model."""

import math

import streamlit as st

from src.predict import load_prediction_artifact, predict_with_artifact


st.set_page_config(
    page_title="Fraud Detection Capstone",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading fraud detection model...")
def get_model_artifact():
    """Load the saved Random Forest once per Streamlit server process."""
    return load_prediction_artifact()


def validate_transaction(transaction):
    """Return validation messages for values that cannot be scored safely."""
    errors = []
    if transaction["step"] < 1:
        errors.append("Step must be at least 1.")

    numeric_fields = {
        "Transaction amount": transaction["amount"],
        "Old origin balance": transaction["oldbalanceOrg"],
        "New origin balance": transaction["newbalanceOrig"],
        "Old destination balance": transaction["oldbalanceDest"],
        "New destination balance": transaction["newbalanceDest"],
    }
    for label, value in numeric_fields.items():
        if not math.isfinite(value):
            errors.append(f"{label} must be a finite number.")
        elif value < 0:
            errors.append(f"{label} cannot be negative.")
    return errors


def show_prediction(result):
    """Render the model result in presentation-friendly language."""
    predicted_fraud = result["isFraud"] == 1
    probability = result["fraud_probability"]

    if predicted_fraud:
        st.error("Potential fraudulent transaction detected")
        explanation = (
            "The saved Random Forest classified this transaction as fraudulent. "
            "It should be reviewed using the organisation's normal investigation process."
        )
        class_label = "Fraud (1)"
    else:
        st.success("Transaction classified as likely legitimate")
        explanation = (
            "The saved Random Forest classified this transaction as legitimate. "
            "This result is a model estimate and does not guarantee that the transaction is risk-free."
        )
        class_label = "Legitimate (0)"

    result_col, probability_col = st.columns(2)
    result_col.metric("Predicted class", class_label)
    probability_col.metric("Fraud probability", f"{probability:.2%}")
    st.write(explanation)
    st.caption(
        "The probability is the model's class-1 score. It is not a legal finding or a guaranteed real-world probability."
    )


st.markdown(
    """
    <style>
        .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1120px;}
        div[data-testid="stMetric"] {
            border: 1px solid #d8e2ea;
            border-radius: 0.75rem;
            padding: 1rem;
            background: #f8fafc;
        }
        div[data-testid="stForm"] {
            border: 1px solid #d8e2ea;
            border-radius: 0.9rem;
            padding: 1.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Fraud Detection Capstone")
st.write(
    "Enter the transaction details below to generate a prediction from the trained Random Forest model."
)

try:
    artifact = get_model_artifact()
except Exception as error:
    st.error("The saved fraud detection model could not be loaded.")
    st.exception(error)
    st.stop()

model = artifact["model"]
schema = artifact["preprocessing"]

with st.sidebar:
    st.header("Model information")
    st.write("**Model:** Random Forest")
    st.write(f"**Trees:** {model.n_estimators}")
    st.write(f"**Model features:** {len(schema['feature_columns'])}")
    st.success("Saved model loaded")
    st.caption("The application uses the Phase 1 model artifact and does not retrain it.")

st.subheader("Transaction details")
st.caption("All balance and amount fields must be zero or greater.")

with st.form("transaction_form", clear_on_submit=False):
    top_left, top_right = st.columns(2)
    with top_left:
        step = st.number_input(
            "Step (simulated hour)",
            min_value=1,
            value=1,
            step=1,
            help="The notebook defines one step as one simulated hour.",
        )
    with top_right:
        transaction_type = st.selectbox(
            "Transaction type",
            options=schema["type_categories"],
            index=schema["type_categories"].index("PAYMENT"),
        )

    amount = st.number_input(
        "Transaction amount",
        min_value=0.0,
        value=0.0,
        step=100.0,
        format="%.2f",
    )

    st.markdown("**Origin account balances**")
    origin_left, origin_right = st.columns(2)
    with origin_left:
        old_origin = st.number_input(
            "Old balance of origin account",
            min_value=0.0,
            value=0.0,
            step=100.0,
            format="%.2f",
        )
    with origin_right:
        new_origin = st.number_input(
            "New balance of origin account",
            min_value=0.0,
            value=0.0,
            step=100.0,
            format="%.2f",
        )

    st.markdown("**Destination account balances**")
    destination_left, destination_right = st.columns(2)
    with destination_left:
        old_destination = st.number_input(
            "Old balance of destination account",
            min_value=0.0,
            value=0.0,
            step=100.0,
            format="%.2f",
        )
    with destination_right:
        new_destination = st.number_input(
            "New balance of destination account",
            min_value=0.0,
            value=0.0,
            step=100.0,
            format="%.2f",
        )

    submitted = st.form_submit_button(
        "Predict Transaction", type="primary", use_container_width=True
    )

if submitted:
    transaction = {
        "step": int(step),
        "type": transaction_type,
        "amount": float(amount),
        "oldbalanceOrg": float(old_origin),
        "newbalanceOrig": float(new_origin),
        "oldbalanceDest": float(old_destination),
        "newbalanceDest": float(new_destination),
    }
    validation_errors = validate_transaction(transaction)
    if validation_errors:
        for message in validation_errors:
            st.error(message)
    else:
        try:
            prediction = predict_with_artifact(transaction, artifact)
            st.divider()
            st.subheader("Prediction result")
            show_prediction(prediction)
        except (TypeError, ValueError) as error:
            st.error(f"The transaction could not be processed: {error}")
        except Exception:
            st.error("Prediction failed. Confirm that the saved model artifact is available and compatible.")

st.divider()
with st.expander("About this project"):
    st.write(
        "This application is the deployment interface for a BIA Data Science & AI fraud detection "
        "capstone. The underlying analysis compares Logistic Regression, Random Forest, and Gradient "
        "Boosting. This interface uses the saved Random Forest selected in Phase 1."
    )
    st.write(
        "The application applies the same four balance-based engineered features, transaction-type "
        "encoding, and feature alignment used during training. It does not retrain the model."
    )
    st.caption(
        "Capstone limitation: results are specific to the supplied dataset, and the notebook's investigation "
        "of the step feature remains unresolved."
    )
