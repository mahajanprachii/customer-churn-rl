# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Customer Churn Env Environment.

This module creates an HTTP server that exposes the CustomerChurnEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import CustomerChurnAction, CustomerChurnObservation
    from .customer_churn_env_environment import CustomerChurnEnvironment
except (ModuleNotFoundError, ImportError):
    from customer_churn_env.models import CustomerChurnAction, CustomerChurnObservation
    from customer_churn_env.server.customer_churn_env_environment import CustomerChurnEnvironment

# Create the app with web interface and README integration
app = create_app(
    CustomerChurnEnvironment,
    CustomerChurnAction,
    CustomerChurnObservation,
    env_name="customer_churn_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)
from fastapi.responses import JSONResponse

@app.get("/tasks")
def get_tasks():
    return JSONResponse({
        "tasks": [
            {
                "name": "easy",
                "description": "New customer with high complaints on monthly contract",
                "difficulty": "easy",
                "action_schema": {"action_type": "free_upgrade | offer_discount | personal_call | do_nothing"}
            },
            {
                "name": "medium",
                "description": "Mid-tenure customer with some churn signals",
                "difficulty": "medium",
                "action_schema": {"action_type": "free_upgrade | offer_discount | personal_call | do_nothing"}
            },
            {
                "name": "hard",
                "description": "High value customer with many complaints",
                "difficulty": "hard",
                "action_schema": {"action_type": "free_upgrade | offer_discount | personal_call | do_nothing"}
            }
        ]
    })

@app.get("/grader")
def get_grader():
    return JSONResponse({
        "grader": {
            "easy": {"correct_action": "free_upgrade", "max_score": 1.0},
            "medium": {"correct_action": "offer_discount", "max_score": 1.0},
            "hard": {"correct_action": "personal_call", "max_score": 1.0, "partial_action": "offer_discount", "partial_score": 0.5}
        }
    })

@app.get("/baseline")
def get_baseline():
    return JSONResponse({
        "baseline_scores": {
            "easy": 1.0,
            "medium": 1.0,
            "hard": 0.5
        },
        "average_score": 0.833
    })

def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m customer_churn_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn customer_churn_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
