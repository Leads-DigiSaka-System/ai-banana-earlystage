# Pytest fixtures for feedback enhancement and API tests.

import uuid
from unittest.mock import MagicMock

import pytest


def make_query_chain(*, all_items=None, count=0, first_result=None):
    """Build a MagicMock chain for db.query(X).filter().order_by().offset().limit().all() / .count() / .first()."""
    chain = MagicMock()
    chain.filter.return_value = chain
    chain.order_by.return_value = chain
    chain.offset.return_value = chain
    chain.limit.return_value = chain
    chain.all.return_value = all_items if all_items is not None else []
    chain.count.return_value = count
    chain.first.return_value = first_result
    return chain


def mock_row(**kwargs):
    """Build a mock ORM row so _row_to_dict(row) can read attributes. Use metadata_= for 'metadata' column."""
    m = MagicMock()
    for k, v in kwargs.items():
        setattr(m, k, v)
    cols = [MagicMock(key="metadata" if k == "metadata_" else k) for k in kwargs.keys()]
    table = MagicMock()
    table.columns = cols
    setattr(m, "__table__", table)
    return m


@pytest.fixture
def sample_image_bytes():
    """Minimal valid JPEG bytes (1x1 pixel)."""
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' \",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00\x00?\x00\xfe\x02\x1a\x7f\xff\xd9"
    )


@pytest.fixture
def sample_prediction_result():
    """Minimal prediction result dict as returned by inference."""
    return {
        "class_id": 2,
        "class_name": "Stage2",
        "confidence": 0.85,
        "bbox": {"x1": 10.0, "y1": 20.0, "x2": 100.0, "y2": 120.0},
    }


@pytest.fixture
def mock_db():
    """Mock SQLAlchemy Session for unit tests."""
    db = MagicMock()
    db.add = MagicMock()
    db.commit = MagicMock()
    db.query.return_value.filter.return_value.first = MagicMock(return_value=None)
    db.query.return_value.filter.return_value.all = MagicMock(return_value=[])
    return db


@pytest.fixture
def prediction_id_str():
    return "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"


@pytest.fixture
def mock_prediction(prediction_id_str):
    """Mock Prediction model instance."""
    pred = MagicMock()
    pred.id = uuid.UUID(prediction_id_str)
    pred.image_path = "data/uploads/predictions/abc.jpg"
    pred.image_hash = "abc123"
    pred.bbox_data = None
    pred.predicted_class_name = "Stage2"
    return pred


@pytest.fixture
def db_crud_client(mock_db):
    """TestClient for /api/v1/db with get_db overridden to use mock_db. Patches load_model so startup does not fail."""
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from main import app
    from database.connection import get_db

    def override_get_db():
        yield mock_db

    with patch("main.load_model"):
        app.dependency_overrides[get_db] = override_get_db
        with TestClient(app) as c:
            yield c
        app.dependency_overrides.pop(get_db, None)
