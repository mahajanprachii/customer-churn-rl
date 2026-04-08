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