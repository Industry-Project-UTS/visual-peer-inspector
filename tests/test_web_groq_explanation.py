"""The optional language explanation cannot consume private case content or alter ranking."""

import json
import unittest
from unittest.mock import patch

from visual_peer_inspector.web.core import InspectionResult, RankedObject, Session
from visual_peer_inspector.web.groq_explanation import (
    ExplanationUnavailable,
    explain_result,
    ranking_evidence,
)


def session_with_result():
    result = InspectionResult(
        ranking=(
            RankedObject("private-item-a", 0.81, ("view-a", "view-b")),
            RankedObject("private-item-b", 0.63, ("view-b",)),
        ),
        observations=(),
        review_state="review_required",
        mode="automatic",
        selected_identity_id="private-item-a",
        cache_token="private-cache-token",
    )
    return Session(views=(object(), object()), result=result, case={"item": "private case note"})


class GroqExplanationTests(unittest.TestCase):
    def test_evidence_is_anonymous_and_aggregate_only(self):
        evidence = ranking_evidence(session_with_result())
        self.assertEqual(evidence["view_count"], 2)
        self.assertEqual(evidence["leading_candidates"][0], {
            "position": 1, "relative_score": 0.81, "views_with_support": 2,
        })
        text = json.dumps(evidence)
        for private in ("private-item", "private case note", "private-cache-token"):
            self.assertNotIn(private, text)

    def test_missing_result_or_key_never_calls_groq(self):
        with patch("visual_peer_inspector.web.groq_explanation.urlopen") as send:
            with self.assertRaisesRegex(ExplanationUnavailable, "Run an inspection"):
                explain_result(Session(), api_key="test-key")
            with self.assertRaisesRegex(ExplanationUnavailable, "unavailable"):
                explain_result(session_with_result(), api_key="")
            send.assert_not_called()

    def test_request_uses_aggregate_scores_and_preserves_result(self):
        session = session_with_result()
        original = session.result

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self, _):
                return json.dumps({"choices": [{"message": {
                    "content": "The leading object has support in both views."
                }}]}).encode()

        with patch("visual_peer_inspector.web.groq_explanation.urlopen", return_value=Response()) as send:
            self.assertIn("both views", explain_result(session, api_key="test-key"))
        request = send.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(send.call_args.kwargs["timeout"], 8)
        for private in ("private-item", "private case note", "private-cache-token"):
            self.assertNotIn(private, json.dumps(payload))
        self.assertIs(session.result, original)

    def test_service_failure_is_contained_without_leaking_details(self):
        with patch("visual_peer_inspector.web.groq_explanation.urlopen", side_effect=RuntimeError("private failure")):
            with self.assertRaises(ExplanationUnavailable) as error:
                explain_result(session_with_result(), api_key="private-key")
        self.assertNotIn("private", str(error.exception))


if __name__ == "__main__":
    unittest.main()
