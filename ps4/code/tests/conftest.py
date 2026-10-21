"""PS4 autograder bootstrap. Run via scripts/grader/run.py (staff) or
starter/autograder.py (students, public tier); both set
GRADER_SUBMISSION and put sim/ and scripts/grader/ on PYTHONPATH."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "public: student-facing tier (run by starter/autograder.py)")
