from uuid import uuid4
import random
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CustomerChurnAction, CustomerChurnObservation, CustomerChurnState
except ImportError:
    from models import CustomerChurnAction, CustomerChurnObservation, CustomerChurnState

class CustomerChurnEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = CustomerChurnState(
            episode_id=str(uuid4()),
            step_count=0,
            current_task="easy",
            is_done=False
        )
        self._current_customer = None
        self._task_list = ["easy", "medium", "hard"]
        self._task_index = 0

    def _generate_customer(self, task: str) -> CustomerChurnObservation:
        if task == "easy":
            return CustomerChurnObservation(
                monthly_charges=random.uniform(80, 120),
                tenure_months=random.randint(0, 3),
                complaint_count=random.randint(2, 4),
                contract_type="monthly",
                done=False,
                reward=0.0
            )
        elif task == "medium":
            return CustomerChurnObservation(
                monthly_charges=random.uniform(60, 100),
                tenure_months=random.randint(6, 18),
                complaint_count=random.randint(1, 2),
                contract_type="monthly",
                done=False,
                reward=0.0
            )
        else:
            return CustomerChurnObservation(
                monthly_charges=random.uniform(150, 200),
                tenure_months=random.randint(12, 36),
                complaint_count=random.randint(3, 5),
                contract_type="monthly",
                done=False,
                reward=0.0
            )

    def _grade(self, task: str, action: str) -> float:
        if task == "easy":
            return 1.0 if action == "free_upgrade" else 0.0
        elif task == "medium":
            return 1.0 if action == "offer_discount" else 0.0
        else:
            return 1.0 if action == "personal_call" else (0.5 if action == "offer_discount" else 0.0)

    def reset(self) -> CustomerChurnObservation:
        self._task_index = 0
        self._state = CustomerChurnState(
            episode_id=str(uuid4()),
            step_count=0,
            current_task="easy",
            is_done=False
        )
        self._current_customer = self._generate_customer("easy")
        return self._current_customer

    def step(self, action: CustomerChurnAction) -> CustomerChurnObservation:
        self._state.step_count += 1
        task = self._task_list[self._task_index]
        reward = self._grade(task, action.action_type)

        self._task_index += 1
        is_done = self._task_index >= len(self._task_list)
        self._state.is_done = is_done

        if not is_done:
            next_task = self._task_list[self._task_index]
            self._state.current_task = next_task
            next_customer = self._generate_customer(next_task)
            next_customer.reward = reward
            next_customer.done = is_done
            return next_customer
        else:
            return CustomerChurnObservation(
                monthly_charges=0,
                tenure_months=0,
                complaint_count=0,
                contract_type="done",
                done=True,
                reward=reward
            )

    @property
    def state(self) -> State:
        return self._state