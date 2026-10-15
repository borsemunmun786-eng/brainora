"""Flask workflow tests using isolated temporary storage."""
import os
import tempfile
import unittest
from io import BytesIO
from unittest.mock import patch

_test_data = tempfile.TemporaryDirectory()
os.environ["APP_ENV"] = "development"
os.environ["APP_DATA_DIR"] = _test_data.name
for _variable in ("DATABASE_PATH", "SESSION_FILE_DIR", "UPLOAD_FOLDER"):
    os.environ.pop(_variable, None)

import app as application


class QuizWorkflowTests(unittest.TestCase):
    def setUp(self):
        application.app.config["TESTING"] = True
        self.client = application.app.test_client()

    def test_production_requires_a_secret(self):
        with self.assertRaises(RuntimeError):
            application._secret_key_for_environment("production", None)

    def test_upload_generate_submit_and_persist_history(self):
        study_text = "Networking concepts and protocols. " * 10
        questions = [
            {
                "question": f"Question {index}?",
                "options": {"A": "Correct", "B": "Option B", "C": "Option C", "D": "Option D"},
                "answer": "A",
                "answer_text": "Correct",
                "topic": "Networks",
                "source": study_text,
            }
            for index in range(5)
        ]

        with patch.object(application, "extract_text", return_value=study_text), \
             patch.object(application, "detect_topics", return_value={"Networks": study_text}):
            upload = self.client.post(
                "/api/upload",
                data={"pdf": (BytesIO(b"test pdf"), "notes.pdf")},
                content_type="multipart/form-data",
            )
        self.assertEqual(upload.status_code, 200)
        self.assertTrue(upload.get_json()["success"])

        with patch.object(application, "generate_quiz_from_topics", return_value=questions):
            generated = self.client.post(
                "/api/generate",
                json={"num_questions": 5, "questions_per_topic": 3},
            )
        self.assertEqual(generated.status_code, 200)
        self.assertEqual(generated.get_json()["total"], 5)

        with patch.object(application.coach, "run_coach", return_value={"action": {}, "insights": []}), \
             patch.object(application.coach, "update_plan"):
            submitted = self.client.post(
                "/api/submit",
                json={"answers": {str(index): "A" for index in range(5)}},
            )
        self.assertEqual(submitted.status_code, 200)
        self.assertEqual(submitted.get_json()["score_pct"], 100)

        history = self.client.get("/api/history").get_json()
        self.assertTrue(history["attempts"])
        self.assertEqual(history["attempts"][0]["pdf_name"], "notes.pdf")


if __name__ == "__main__":
    unittest.main()