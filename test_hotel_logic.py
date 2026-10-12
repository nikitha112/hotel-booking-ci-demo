
def predict_booking_status(lead_time, special_requests):
    """
    Demonstration rule for the hotel booking CI practical.
    This is a rule-based example, not a trained ML model.
    """
    if lead_time < 0 or special_requests < 0:
        raise ValueError("Booking features cannot be negative")

    if lead_time <= 7 and special_requests > 0:
        return "Not_Canceled"
    else:
        return "Canceled"


if __name__ == "__main__":
    result = predict_booking_status(5, 2)
    print("Predicted Booking Status:", result)
