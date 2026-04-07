# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Customer Churn Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CustomerChurnAction, CustomerChurnObservation, CustomerChurnState


class CustomerChurnEnv(
    EnvClient[CustomerChurnAction, CustomerChurnObservation, CustomerChurnState]
):
    """
    Client for the Customer Churn Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with CustomerChurnEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(CustomerChurnAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = CustomerChurnEnv.from_docker_image("customer_churn_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(CustomerChurnAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: CustomerChurnAction) -> Dict:
        """
        Convert CustomerChurnAction to JSON payload for step message.

        Args:
            action: CustomerChurnAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_type": action.action_type,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CustomerChurnObservation]:
        """
        Parse server response into StepResult[CustomerChurnObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with CustomerChurnObservation
        """
        obs_data = payload.get("observation", {})
        observation = CustomerChurnObservation(
            monthly_charges=obs_data.get("monthly_charges", 0.0),
            tenure_months=obs_data.get("tenure_months", 0),
            complaint_count=obs_data.get("complaint_count", 0),
            contract_type=obs_data.get("contract_type", "monthly"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> CustomerChurnState:
        """
        Parse server response into CustomerChurnState object.

        Args:
            payload: JSON response from state request

        Returns:
            CustomerChurnState object
        """
        return CustomerChurnState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            current_step=payload.get("current_step", 0),
            current_task=payload.get("current_task", "easy"),
            is_done=payload.get("is_done", False),
        )
