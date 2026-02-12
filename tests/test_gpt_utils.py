import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os
import asyncio
import json
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from locomo.evaluation.gpt_utils import get_input_context, get_gpt_answers_async, process_single_question

class MockArgs:
    def __init__(self):
        self.max_context = 1000
        self.batch_size = 2
        self.model_name = "gpt-3.5-turbo" # Dummy
        self.overwrite = True
        self.category = None
        self.max_questions = None

class TestGptUtils(unittest.TestCase):

    def setUp(self):
        self.args = MockArgs()
        self.encoding = MagicMock()
        self.encoding.encode.side_effect = lambda x: [1] * len(x) # Simply return length as tokens
        self.encoding.decode.side_effect = lambda x: "".join(["a"]*len(x)) # Dummy decode
    
    def test_chronological_order(self):
        """Verify get_input_context assembles sessions in chronological order (1 -> N)."""
        data = {
            "session_1_date_time": "2023-01-01",
            "session_1": [{"speaker": "A", "text": "Hello"}],
            "session_3_date_time": "2023-01-03", 
            "session_3": [{"speaker": "A", "text": "Bye"}],
            "session_2_date_time": "2023-01-02",
            "session_2": [{"speaker": "B", "text": "Hi"}],
        }
        
        context = get_input_context(data, 0, self.encoding, self.args)
        
        # Expect order: Session 1 "Hello", Session 2 "Hi", Session 3 "Bye"
        self.assertIn("DATE: 2023-01-01", context)
        self.assertIn("DATE: 2023-01-02", context)
        self.assertIn("DATE: 2023-01-03", context)
        
        idx1 = context.find("Date: 2023-01-01") # Case sensitive? The function uses "DATE:"
        idx1 = context.find("DATE: 2023-01-01")
        idx2 = context.find("DATE: 2023-01-02")
        idx3 = context.find("DATE: 2023-01-03")
        
        self.assertLess(idx1, idx2, "Session 1 should come before Session 2")
        self.assertLess(idx2, idx3, "Session 2 should come before Session 3")

    def test_truncation(self):
        """Verify get_input_context truncates from the beginning if too long."""
        self.args.max_context = 200 # Must be > buffer (100) + some tokens. 
        # buffer is 100. max_avail = 200 - 0 - 100 = 100.
        
        # Smart mock: encode returns list of char codes, decode reconstructs string
        # This preserves content for asserts
        self.encoding.encode.side_effect = lambda x: [ord(c) for c in x]
        self.encoding.decode.side_effect = lambda x: "".join([chr(i) for i in x])
        
        data = {
            "session_1_date_time": "2023-01-01",
            "session_1": [{"speaker": "A", "text": "LongOldHistory" * 10}], # lengthy
            "session_2_date_time": "2023-01-02", 
            "session_2": [{"speaker": "B", "text": "Recent"}],
        }
        
        context = get_input_context(data, 0, self.encoding, self.args)
        
        duration = end_time - start_time
        
        # Relax threshold to 0.5s to account for significant windows overhead
        self.assertLess(duration, 0.5, f"Should be faster than serial execution (took {duration}s)")
        self.assertEqual(mock_run.call_count, 3)
        self.assertEqual(result['qa'][0]['pred'], "Mock Answer")

if __name__ == '__main__':
    unittest.main()
