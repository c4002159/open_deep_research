import re
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class ExecutionAccuracyMetric(BaseMetric):
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.name = "Execution Accuracy"

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[,\s]", "", str(text).lower())

    def measure(self, test_case: LLMTestCase) -> float:
        expected_values = [v for v in test_case.expected_output.split("|") if v]
        actual_norm = self._normalize(test_case.actual_output)
        hits = sum(1 for v in expected_values if self._normalize(v) in actual_norm)
        metadata = getattr(test_case, "metadata", None) or {}
        match_mode = metadata.get("match_mode", "all")

        if match_mode == "any":
            self.score = 1.0 if hits > 0 else 0.0
        else:
            self.score = hits / len(expected_values) if expected_values else 0.0

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success