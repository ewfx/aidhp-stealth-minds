To run this script, run the following command
pytest -v

**TEST Results**
----------------------------
======================================= test session starts ========================================
platform linux -- Python 3.11.11, pytest-8.3.5, pluggy-1.5.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /content
plugins: langsmith-0.3.18, anyio-4.9.0, typeguard-4.4.2
collected 4 items                                                                                  

test_app.py::test_load_data PASSED                                                           [ 25%]
test_app.py::test_check_bias PASSED                                                          [ 50%]
test_app.py::test_run_benchmarking PASSED                                                    [ 75%]
test_app.py::test_generate_recommendation PASSED                                             [100%]

======================================== 4 passed in 12.28s ========================================
