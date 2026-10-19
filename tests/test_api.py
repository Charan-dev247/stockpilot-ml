
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_inventory_summary():
    response = client.get("/api/summary")

    assert response.status_code == 200

    data = response.json()
    assert data["total_products"] > 0
    assert data["reorder_now"] >= 0
    assert data["total_proposed_units"] >= 0
    assert data["test_data_used_for_planning"] is False


def test_inventory_list_pagination():
    response = client.get(
        "/api/inventory",
        params={"page": 1, "page_size": 10},
    )

    assert response.status_code == 200

    data = response.json()
    assert len(data["items"]) == 10
    assert data["page"] == 1
    assert data["page_size"] == 10
    assert data["total"] > 0


def test_inventory_search():
    response = client.get(
        "/api/inventory",
        params={"search": "17003"},
    )

    assert response.status_code == 200

    items = response.json()["items"]
    assert any(item["StockCode"] == "17003" for item in items)


def test_inventory_recommendation_filter():
    response = client.get(
        "/api/inventory",
        params={
            "recommendation": "Reorder now",
            "page_size": 10,
        },
    )

    assert response.status_code == 200

    items = response.json()["items"]
    assert len(items) == 10
    assert all(
        item["recommendation"] == "Reorder now"
        for item in items
    )


def test_missing_product_returns_404():
    response = client.get("/api/inventory/UNKNOWN_PRODUCT_CODE")

    assert response.status_code == 404


def test_invalid_page_size_returns_422():
    response = client.get(
        "/api/inventory",
        params={"page_size": 1000},
    )

    assert response.status_code == 422
