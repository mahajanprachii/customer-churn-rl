from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field

class CustomerChurnObservation(Observation):
    monthly_charges: float = Field(..., description="Customer's monthly bill amount")
    tenure_months: int = Field(..., description="How long the customer has been with us")
    complaint_count: int = Field(..., description="Number of complaints filed")
    contract_type: str = Field(..., description="Month-to-month, One year, Two year")

class CustomerChurnAction(Action):
    action_type: str = Field(..., description="offer_discount | free_upgrade | personal_call | do_nothing")

class CustomerChurnState(State):
    current_step: int = Field(default=0, description="Step number within the episode")
    current_task: str = Field(default="easy", description="easy | medium | hard")
    is_done: bool = Field(default=False, description="Whether the episode has ended")