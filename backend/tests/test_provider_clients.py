import json
import unittest
from unittest.mock import Mock, patch

import requests

from backend.app.provider_clients import (
    TrackBox,
    _parse_responses_text,
    _parse_sam_boxes,
    _spark_cost,
    ProviderClients,
    ProviderError,
)


class ProviderResponseTests(unittest.TestCase):
    def test_empty_baseten_reply_still_records_returned_usage(self):
        response = Mock(status_code=200)
        response.json.return_value = {"usage": {"prompt_tokens": 10, "completion_tokens": 512},
                                      "choices": [{"message": {"content": ""}}]}
        tracker = Mock()
        with patch.dict("os.environ", {"META_API_KEY": "test-key", "BASETEN_API_KEY": "test-key"}), patch(
            "backend.app.provider_clients.requests.post", return_value=response
        ):
            client = ProviderClients(tracker)
            with self.assertRaisesRegex(ProviderError, "no user-facing text"):
                client.baseten_spark_open("test prompt", "test-run")
        tracker.record_call.assert_called_once()
        self.assertEqual(tracker.record_call.call_args.kwargs["units"], 522)

    def test_responses_api_reads_user_facing_text_and_usage(self):
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(
            {
                "status": "completed",
                "output": [
                    {"type": "reasoning", "summary": []},
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "ready"},
                        ],
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        ).encode()

        result = _parse_responses_text(response)

        self.assertEqual(result.text, "ready")
        self.assertEqual(result.input_tokens, 10)
        self.assertEqual(result.output_tokens, 5)
        self.assertEqual(result.response_status, "completed")

    def test_sam_parser_preserves_frame_and_object_ids(self):
        output = (
            "<0f>0<|box;x1=1;y1=2;x2=11;y2=22;w=640;h=480|>"
            "<|mask;x=0;y=0;data=1,2,~...|>,"
            "1<|box;x1=30;y1=40;x2=50;y2=70;w=640;h=480|>"
            "<|mask;x=0;y=0;data=1,2,~...|>"
            "<9f>0<|box;x1=3;y1=4;x2=13;y2=24;w=640;h=480|>"
            "<|mask;x=0;y=0;data=1,2,~...|>"
        )

        boxes = _parse_sam_boxes(output)

        self.assertEqual(
            [(box.frame_index, box.object_index) for box in boxes],
            [(0, 0), (0, 1), (9, 0)],
        )
        self.assertIsInstance(boxes[0], TrackBox)
        self.assertEqual(boxes[0].center_x, 6)
        self.assertEqual(boxes[0].center_y, 12)

    def test_spark_cost_uses_standard_input_and_output_rates(self):
        self.assertAlmostEqual(_spark_cost(1_000_000, 1_000_000), 5.5)


if __name__ == "__main__":
    unittest.main()
