import unittest
from unittest.mock import patch, MagicMock
from options_flow_analyzer import OptionsFlowAnalyzer, OptionsFlow

class TestOptionsFlowAnalyzer(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_key"
        self.analyzer = OptionsFlowAnalyzer(self.api_key)
        
    @patch('requests.get')
    def test_fetch_options_flow(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'call_volume': 10000,
            'put_volume': 8000,
            'call_open_interest': 50000,
            'put_open_interest': 45000,
            'put_call_ratio_volume': 0.8,
            'put_call_ratio_oi': 0.9,
            'flow': [
                {
                    'strike': 100,
                    'expiration': '2024-02-16',
                    'type': 'call',
                    'premium': 150000,
                    'volume': 500,
                    'open_interest': 1000,
                    'is_opening': True
                }
            ]
        }
        mock_get.return_value = mock_response
        
        flow = self.analyzer.fetch_options_flow('NVDA')
        
        self.assertEqual(flow.call_volume, 10000)
        self.assertEqual(flow.put_volume, 8000)
        self.assertEqual(flow.pcr_volume, 0.8)
        self.assertEqual(len(flow.unusual_activity), 1)
        
    def test_analyze_flow(self):
        flow = OptionsFlow(
            call_volume=10000,
            put_volume=8000,
            call_oi=50000,
            put_oi=45000,
            pcr_volume=0.8,
            pcr_oi=0.9,
            unusual_activity=[
                {
                    'strike': 100,
                    'premium': 150000,
                    'type': 'call'
                }
            ],
            institutional_sentiment=0.6
        )
        
        analysis = self.analyzer.analyze_flow(flow)
        
        self.assertIn('volume_signal', analysis)
        self.assertIn('oi_signal', analysis)
        self.assertIn('institutional_signal', analysis)
        self.assertIn('unusual_activity_score', analysis)
        self.assertIn('composite_score', analysis)
        
        self.assertEqual(analysis['institutional_signal'], 0.6)
        self.assertTrue(-1 <= analysis['composite_score'] <= 1)

if __name__ == '__main__':
    unittest.main()
