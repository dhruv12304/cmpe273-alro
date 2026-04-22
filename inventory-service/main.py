"""Inventory service FastAPI application."""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from state import WarehouseState

GRID_SIZE = int(os.getenv("GRID_SIZE", "5"))

_warehouse: WarehouseState | None = None


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line to stdout."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _configure_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers.clear()
    root.addHandler(handler)


_configure_logging()
log = logging.getLogger("inventory")


def round_congestion_map(data: dict[str, float]) -> dict[str, float]:
    """Round monetary/weight floats to 2 decimal places for API responses."""
    return {k: round(float(v), 2) for k, v in data.items()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _warehouse
    _warehouse = WarehouseState(GRID_SIZE)
    log.info("WarehouseState initialised grid_size=%s", GRID_SIZE)
    yield
    _warehouse = None


app = FastAPI(title="ALRO Inventory Service", lifespan=lifespan)


def get_warehouse() -> WarehouseState:
    if _warehouse is None:
        raise RuntimeError("Warehouse not initialised")
    return _warehouse


WarehouseDep = Annotated[WarehouseState, Depends(get_warehouse)]


class HealthResponse(BaseModel):
    status: str
    service: str


class DeductBody(BaseModel):
    deduct: int = Field(..., ge=0)


class DeductResponse(BaseModel):
    node_id: int
    new_stock: int


class StateResetResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="inventory")


@app.get("/inventory")
def get_inventory(wh: WarehouseDep) -> dict[int, int]:
    return wh.get_all_inventory()


@app.get("/inventory/{node_id}")
def get_inventory_node(node_id: int, wh: WarehouseDep) -> dict:
    try:
        return wh.get_node(node_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Not a warehouse node") from None


@app.put("/inventory/{node_id}", response_model=DeductResponse)
def put_inventory_deduct(
    node_id: int, body: DeductBody, wh: WarehouseDep
) -> DeductResponse:
    try:
        new_stock = wh.deduct_stock(node_id, body.deduct)
    except KeyError:
        raise HTTPException(status_code=404, detail="Not a warehouse node") from None
    return DeductResponse(node_id=node_id, new_stock=new_stock)


@app.get("/congestion")
def get_congestion(wh: WarehouseDep) -> dict[str, float]:
    return round_congestion_map(wh.get_congestion())


@app.post("/congestion/reset")
def post_congestion_reset(wh: WarehouseDep) -> dict[str, float]:
    return round_congestion_map(wh.reset_congestion())


@app.post("/state/reset", response_model=StateResetResponse)
def post_state_reset(wh: WarehouseDep) -> StateResetResponse:
    wh.reset_stock()
    wh.reset_congestion()
    return StateResetResponse(status="reset")
