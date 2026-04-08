from uuid import uuid4
import random
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from openenv.core.rubrics.base import Rubric

try:
    from ..models import CustomerChurnAction, CustomerChurnObservation, CustomerChurnState
except ImportError:
    from models import CustomerChurnAction, CustomerChurnObservation, CustomerChurnState


class TaskGrader(Rubric):
    def __init__(self, expected_action: str, partial_action: str = None, partial_score: float = 0.0):
        super().__init__()
        self.expected_action = expected_action
        self.partial_action = partial_action
        self.partial_score = partial_score

    def forward(self, action: CustomerChurnAction, observation: CustomerChurnObservation) -> float:
        if action.action_type == self.expected_action:
            return 1.0
        if self.partial_action and action.action_type == self.partial_action:
            return self.partial_score
        return 0.0


# ✅ ADD THESE FUNCTIONS (CRITICAL FIX)
def easy_grader(action, observation):
    return TaskGrader("free_upgrade")(action, observation)

def medium_grader(action, observation):
    return TaskGrader("offer_discount")(action, observation)

def hard_grader(action, observation):
    return TaskGrader("personal_call", "offer_discount", 0.5)(action, observation)


class CustomerChurnRubric(Rubric):
    def __init__(self, env: 'CustomerChurnEnvironment'):
        super().__init__()
        self.env = env
        self.easy = TaskGrader("free_upgrade")
        self.medium = TaskGrader("offer_discount")
        self.hard = TaskGrader("personal_call", "offer_discount", 0.5)

    def forward(self, action: CustomerChurnAction, observation: CustomerChurnObservation) -> float:
        task = self.env._task_list[self.env._task_index]
        if task == "easy":
            return self.easy(action, observation)
        elif task == "medium":
            return self.medium(action, observation)
        elif task == "hard":
            return self.hard(action, observation)
        return 0.0


class CustomerChurnEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__(rubric=CustomerChurnRubric(self))
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

    def reset(self, seed=None, episode_id=None, **kwargs) -> CustomerChurnObservation:
        self._reset_rubric()
        
        task = kwargs.get('task') or kwargs.get('task_id')
        if task in self._task_list:
            self._task_index = self._task_list.index(task)
        else:
            self._task_index = 0
            
        self._state = CustomerChurnState(
            episode_id=str(uuid4()),
            step_count=0,
            current_task=self._task_list[self._task_index],
            is_done=False
        )
        self._current_customer = self._generate_customer(self._state.current_task)
        return self._current_customer

    def step(self, action: CustomerChurnAction, timeout_s=None, **kwargs) -> CustomerChurnObservation:
        self._state.step_count += 1
        
        reward = self._apply_rubric(action, self._current_customer)

        self._task_index += 1
        is_done = self._task_index >= len(self._task_list)
        self._state.is_done = is_done

        if not is_done:
            next_task = self._task_list[self._task_index]
            self._state.current_task = next_task
            next_customer = self._generate_customer(next_task)
            next_customer.reward = reward
            next_customer.done = is_done
            self._current_customer = next_customer
            return next_customer
        else:
            final_customer = CustomerChurnObservation(
                monthly_charges=0,
                tenure_months=0,
                complaint_count=0,
                contract_type="done",
                done=True,
                reward=reward
            )
            self._current_customer = final_customer
            return final_customer

    @property
    def state(self) -> State:
        return self._state