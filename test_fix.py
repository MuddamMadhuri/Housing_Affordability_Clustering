import unittest
import json
import app

class TestSimulatorFix(unittest.TestCase):
    def setUp(self):
        app.app.testing = True
        self.client = app.app.test_client()
        # Ensure resources are loaded
        app.load_resources()

    def test_case_a_high_income(self):
        """
        CASE A: High Income -> Cluster 2 (High-Income Secure)
        """
        payload = {"income": 90000, "cost": 1800, "age": 45, "bedrooms": 3}
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        print(f"\n[CASE A] High Income: {data['primary_result']['predicted_cluster']['label']}")
        self.assertEqual(data['primary_result']['predicted_cluster']['id'], 2)
        self.assertFalse(data['warning']['anomaly_flag'])

    def test_case_b_middle_income(self):
        """
        CASE B: Middle Income -> Cluster 0 (Middle-Income Stable)
        """
        payload = {"income": 45000, "cost": 1200}
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        print(f"\n[CASE B] Middle Income: {data['primary_result']['predicted_cluster']['label']}")
        self.assertEqual(data['primary_result']['predicted_cluster']['id'], 0)
        # Age should be defaulted if missing
        self.assertEqual(data['primary_result']['age_used'], 44.0)

    def test_case_c_monthly_mistake(self):
        """
        CASE C: Monthly Mistake -> Low Income (Cluster 3) + Anomaly + Alternative
        """
        payload = {"income": 1300, "cost": 800}
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        print(f"\n[CASE C] Mistake: Anomaly={data['warning']['anomaly_flag']}, Alt={data['alternative_interpretation']['adjusted_annual_income']}")
        
        # Primary is Low Income
        self.assertIn(data['primary_result']['predicted_cluster']['id'], [1, 3])
        # Anomaly Flag True
        self.assertTrue(data['warning']['anomaly_flag'])
        self.assertIn("Potential monthly income entry detected.", data['warning']['messages'])
        # Alternative exists (1300 * 12 = 15600)
        self.assertIsNotNone(data['alternative_interpretation'])
        self.assertEqual(data['alternative_interpretation']['adjusted_annual_income'], 15600)

    def test_edge_case_10_missing_age(self):
        """
        EDGE CASE 10: Missing Age -> Default Age, No Anomaly
        """
        payload = {"income": 45000, "cost": 1200, "age": None, "bedrooms": 2}
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        print(f"\n[CASE 10] Missing Age: Used {data['primary_result']['age_used']}")
        self.assertEqual(data['primary_result']['age_used'], 44.0)
        self.assertFalse(data['warning']['anomaly_flag'])
        self.assertIn("age_defaulted_median", data['primary_result']['notes'])

    def test_edge_case_11_invalid_age(self):
        """
        EDGE CASE 11: Invalid Age (140) -> Default Age, Anomaly True
        """
        payload = {"income": 35000, "cost": 900, "age": 140, "bedrooms": 2}
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        print(f"\n[CASE 11] Invalid Age: Anomaly={data['warning']['anomaly_flag']}")
        self.assertEqual(data['primary_result']['age_used'], 44.0)
        self.assertTrue(data['warning']['anomaly_flag'])
        self.assertIn("age_out_of_valid_range: 140.0", data['primary_result']['notes'][0])

if __name__ == '__main__':
    unittest.main()
