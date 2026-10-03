# CI Python version note (week-38)

## Problem

`rl/requirements.txt` pins `numpy==2.5.3` / `scipy==1.18.1` / `shap==0.52.0`,
which require **Python >= 3.12**. GitHub Actions `.github/workflows/ci.yml` was
still on `python-version: "3.11"`, so Install dependencies failed with
`No matching distribution found for numpy==2.5.3`.

## Preferred fix (blocked on push)

```yaml
# .github/workflows/ci.yml
python-version: "3.12"
```

Pushing that one-line change is refused by GitHub when the OAuth app token lacks
the `workflow` scope (`refusing to allow an OAuth App to create or update
workflow … without workflow scope`). Dockerfile **is** on `python:3.12-slim`.

## Interim workaround (in tree)

Environment markers in `rl/requirements.txt`:

| Package | Python >= 3.12 (Colab / Docker) | Python < 3.12 (GHA 3.11) |
|---|---|---|
| numpy | `==2.5.3` | `==2.4.6` |
| scipy | `==1.18.1` | `==1.17.1` |
| shap | `==0.52.0` | `==0.51.0` |

Colab/training pins stay exact on 3.12+. When a token with `workflow` scope is
available, bump `ci.yml` to 3.12 and (optionally) drop the `<3.12` marker lines.

## Verify

```bash
pip install -r rl/requirements.txt -r requirements-api.txt
PYTHONPATH=. pytest rl/tests tests -q
```
