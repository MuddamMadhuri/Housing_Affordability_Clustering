import unittest
import json
import app

class TestSimulatorRestore(unittest.TestCase):
    def setUp(self):
        app.app.testing = True
        self.client = app.app.test_client()
        app.load_resources()

    def test_validation_case(self):
        """
        VALIDATION CASE (MUST PASS)
        Input: income = 45000, housing_cost = 1200, age = 35, bedrooms = 3
        Expected: cost_burden = 0.32, predicted_cluster = Cluster 0, policy populated, NO NaN/undefined
        """
        payload = {"income": 45000, "cost": 1200, "age": 35, "bedrooms": 3}
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        print(f"\n[VALIDATION] Cluster: {data.get('cluster')}, Burden: {data.get('cost_burden_ratio')}")
        
        # Check flat structure
        self.assertIn('cluster', data, "Response must have 'cluster' key")
        self.assertIn('cost_burden_ratio', data, "Response must have 'cost_burden_ratio' key")
        self.assertIn('recommendation', data, "Response must have 'recommendation' key")
        
        # Check values
        self.assertEqual(data['cluster'], 0)
        self.assertAlmostEqual(data['cost_burden_ratio'], 0.32, places=2)
        self.assertIsNotNone(data['recommendation'])
        
        # Ensure no logical NaNs
        self.assertNotEqual(str(data['cluster']), "nan")
        self.assertNotEqual(str(data['cluster']), "undefined")

    def test_missing_age(self):
        """
        Test Missing Age -> Default 35, Logged (stdout checked manually or via anomaly if intended)
        """
        payload = {"income": 45000, "cost": 1200, "age": None}
        response = self.client.post('/api/predict', json=payload)
        data = json.loads(response.data)
        
        # Should work safely
        self.assertEqual(data['cluster'], 0)
        # We can't easily check 'age used' in flat response unless returned. 
        # But we ensure it works.

if __name__ == '__main__':
    unittest.main()
