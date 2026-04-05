try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import CustomerChurnAction, CustomerChurnObservation
    from .customer_churn_env_environment import CustomerChurnEnvironment
except (ModuleNotFoundError, ImportError):
    from customer_churn_env.models import CustomerChurnAction, CustomerChurnObservation
    from customer_churn_env.server.customer_churn_env_environment import CustomerChurnEnvironment

from fastapi.responses import JSONResponse

app = create_app(
    CustomerChurnEnvironment,
    CustomerChurnAction,
    CustomerChurnObservation,
    env_name="customer_churn_env",
    max_concurrent_envs=1,
)

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
            "hard": {
                "correct_action": "personal_call",
                "max_score": 1.0,
                "partial_action": "offer_discount",
                "partial_score": 0.5
            }
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
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()