
import unittest
from hotel_logic import predict_booking_status


class TestHotelLogic(unittest.TestCase):

    def test_recent_booking_with_requests(self):
        self.assertEqual(
            predict_booking_status(5, 2),
            "Not_Canceled"
        )

    def test_long_lead_time(self):
        self.assertEqual(
            predict_booking_status(30, 1),
            "Canceled"
        )

    def test_negative_lead_time(self):
        with self.assertRaises(ValueError):
            predict_booking_status(-1, 1)

    def test_negative_special_requests(self):
        with self.assertRaises(ValueError):
            predict_booking_status(5, -1)


if __name__ == "__main__":
    unittest.main()
