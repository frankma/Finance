[![tests](https://github.com/frankma/Finance/actions/workflows/tests.yml/badge.svg)](https://github.com/frankma/Finance/actions/workflows/tests.yml)

Quantitative finance and related topics in implementation.

## Running the tests

```bash
pip install -r requirements.txt
MPLBACKEND=Agg python -m pytest test
```

`MPLBACKEND=Agg` keeps the plotting tests from opening windows. CI runs the same suite on Python 3.11, 3.12 and 3.13 for every push to `master` and every pull request.
