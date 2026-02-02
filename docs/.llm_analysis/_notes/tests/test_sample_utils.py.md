### `tests/test_sample_utils.py`

#### Purpose
This file provides unit tests for the core sampling functions located in `mlx_lm.sample_utils`. These functions (`apply_top_p`, `apply_min_p`, `apply_top_k`, `apply_xtc`) are critical for controlling the diversity and quality of the generated text by manipulating the model's output logits before a token is sampled.

#### Structure
- The tests are written using Python's standard `unittest` framework.
- A single class, `TestSampleUtils`, contains all the test cases.
- No `setUp` or `tearDown` methods are needed, as each test is self-contained and operates on small, manually defined `mlx.core.array` instances. This keeps the tests simple and fast.
- Each sampling utility has a corresponding, clearly named test method (`test_apply_top_p`, `test_apply_top_k`, etc.).

#### Code Quality
- **Clarity and Readability**: The tests are exceptionally clear and easy to follow. Each test case defines a simple probability distribution, applies the function being tested with specific parameters, and then verifies that the output distribution is modified as expected.
- **Correctness**: The assertions are logically sound and cover various scenarios for each function. The tests correctly convert probabilities to logits (the functions' input) and then convert the output logits back to probabilities for easier verification.
- **Test Coverage**: The file demonstrates good test coverage:
  - **Batching**: It explicitly checks that `apply_top_p`, `apply_min_p`, and `apply_top_k` work correctly on batched inputs, which is essential for efficient generation.
  - **Edge Cases**: The tests for `apply_top_p` and `apply_top_k` use different thresholds to ensure that the correct number of tokens are selected (e.g., selecting only the top token, multiple tokens, or all tokens).
  - **`apply_xtc`**: The test for the "Extremely Timely Constrained" sampling method correctly validates its unique thresholding logic and its handling of special tokens.
- **Conventions**: The code adheres to standard Python and `unittest` conventions.

#### Potential Issues/Improvements
- **Floating-Point Comparisons**: The tests occasionally use `round()` followed by `assertEqual` for comparing floating-point numbers. While effective here, a more conventional and robust approach would be to consistently use `mx.allclose` or `unittest.TestCase.assertAlmostEqual` with a defined tolerance. This is a minor stylistic point, as the current implementation is functionally correct.

#### Overall Assessment
This is a high-quality, focused, and effective test file. It provides strong validation for the critical text generation sampling logic. The code is clean, readable, and easy to maintain. No bugs or significant issues were found.
