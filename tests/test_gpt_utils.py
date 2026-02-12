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
        
        idx1 = context.find("DATE: 2023-01-01")
        idx2 = context.find("DATE: 2023-01-02")
        idx3 = context.find("DATE: 2023-01-03")
        
        self.assertLess(idx1, idx2, "Session 1 should come before Session 2")
        self.assertLess(idx2, idx3, "Session 2 should come before Session 3")

    @patch('locomo.evaluation.gpt_utils.run_chatgpt_async', new_callable=AsyncMock)
    async def test_cat5_stability(self, mock_run):
        """Verify Category 5 prompt swapping is deterministic."""
        mock_run.return_value = "Answer"
        in_data = {
            "sample_id": "stable_sample",
            "person1": "A", "person2": "B",
            "conversation": {"session_1": [{"speaker": "A", "text": "X"}]},
            "qa": [{"question": "Is X true?", "category": 5, "answer": "Yes"}]
        }
        out_data = {"qa": [{"question": "Is X true?"}]}
        
        # Run 1
        await get_gpt_answers_async(in_data, out_data, "pred", self.args)
        first_prompt = mock_run.call_args[0][0]
        
        # Run 2
        mock_run.reset_mock()
        await get_gpt_answers_async(in_data, out_data, "pred", self.args)
        second_prompt = mock_run.call_args[0][0]
        
        self.assertEqual(first_prompt, second_prompt, "Adversarial prompt should be identical across runs")

    def test_truncation(self):
        """Verify get_input_context truncates from the beginning if too long."""
        self.args.max_context = 200 # Must be > buffer (100) + some tokens. 
        
        self.encoding.encode.side_effect = lambda x: [ord(c) for c in x]
        self.encoding.decode.side_effect = lambda x: "".join([chr(i) for i in x])
        
        data = {
            "session_1_date_time": "2023-01-01",
            "session_1": [{"speaker": "A", "text": "LongOldHistory" * 10}],
            "session_2_date_time": "2023-01-02", 
            "session_2": [{"speaker": "B", "text": "Recent"}],
        }
        
        context = get_input_context(data, 0, self.encoding, self.args)
        
        self.assertIn("Recent", context)
        self.assertLessEqual(len(context), self.args.max_context)


class TestAsyncGpt(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.args = MockArgs()
        self.args.batch_size = 5
        self.encoding = MagicMock()
        self.encoding.encode.side_effect = lambda x: [1] * len(x)
        
    @patch('locomo.evaluation.gpt_utils.run_chatgpt_async', new_callable=AsyncMock)
    async def test_async_concurrency(self, mock_run):
        """Verify that multiple questions are processed concurrently."""
        mock_run.return_value = "Mock Answer"
        
        in_data = {
            "sample_id": "test_1",
            "person1": "A", "person2": "B",
            "conversation": {},
            "qa": [
                {"question": "Q1", "category": 1},
                {"question": "Q2", "category": 1},
                {"question": "Q3", "category": 1},
            ]
        }
        out_data = {"qa": [{"question": "Q1"}, {"question": "Q2"}, {"question": "Q3"}]}
        
        async def delayed_answer(*args, **kwargs):
            await asyncio.sleep(0.1)
            return "Mock Answer"
        
        mock_run.side_effect = delayed_answer
        
        start_time = time.time()
        result = await get_gpt_answers_async(in_data, out_data, "pred", self.args)
        end_time = time.time()
        
        duration = end_time - start_time
        self.assertLess(duration, 0.5, f"Should be faster than serial execution (took {duration}s)")
        self.assertEqual(mock_run.call_count, 3)
        self.assertEqual(result['qa'][0]['pred'], "Mock Answer")

if __name__ == '__main__':
    unittest.main()
