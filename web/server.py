"""Spec-named entry point: `uvicorn server:app`.

The implementation lives in api.py; this exists so both the command in
API_SPEC.md (`server:app`) and the module the backend was written as
(`api:app`) reach the same application object.
"""

from api import app          # noqa: F401
