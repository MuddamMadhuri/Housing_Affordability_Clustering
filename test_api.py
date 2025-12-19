import app
import json
import unittest

class TestDashboardAPI(unittest.TestCase):
    def setUp(self):
        app.app.testing = True
        self.client = app.app.test_client()
        # Ensure data is loaded
        app.load_resources()

    def test_health(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertGreater(data['rows'], 0)

    def test_get_data(self):
        response = self.client.get('/api/data')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertIn('scatter', data)
        self.assertIn('stats', data)
        self.assertGreater(len(data['scatter']), 0)
        
        # Check stats structure
        first_cluster = list(data['stats'].values())[0]
        self.assertIn('Count', first_cluster)
        self.assertIn('cost_burden_ratio', first_cluster)

    def test_predict_normal(self):
        payload = {"income": 45000, "cost": 1200, "age": 35, "bedrooms": 2}
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertEqual(data['cluster'], 0)
        self.assertEqual(data['anomaly_flag'], False)

    def test_predict_extreme(self):
        payload = {"income": 1300, "cost": 800}
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertEqual(data['anomaly_flag'], True)
        self.assertIn("Extremely high cost burden", data['anomaly_reasons'][0])

if __name__ == '__main__':
    unittest.main()
