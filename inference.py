import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from customer_churn_env import CustomerChurnAction, CustomerChurnEnv

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("CUSTOMER_CHURN_TASK", "customer-churn")
BENCHMARK = os.getenv("CUSTOMER_CHURN_BENCHMARK", "customer_churn_env")
MAX_STEPS = 3
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI retention specialist for a subscription company.
    You will see a customer profile and must choose the best retention action.
    
    Available actions:
    - free_upgrade: Give a free plan upgrade (best for new customers with complaints)
    - offer_discount: Offer a 10% discount (best for medium risk customers)
    - personal_call: Call the customer directly (best for high risk, high value customers)
    - do_nothing: Take no action (best for low risk customers)
    
    Reply with exactly one action string — no quotes, no explanation, just the action.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, observation: dict, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Customer Profile:
          Monthly Charges: {observation.get('monthly_charges', 0):.2f}
          Tenure Months: {observation.get('tenure_months', 0)}
          Complaint Count: {observation.get('complaint_count', 0)}
          Contract Type: {observation.get('contract_type', 'unknown')}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Choose your action:
        """
    ).strip()


def get_model_action(client: OpenAI, step: int, observation: dict, last_reward: float, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, observation, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        valid_actions = ["free_upgrade", "offer_discount", "personal_call", "do_nothing"]
        for action in valid_actions:
            if action in text:
                return action
        return "do_nothing"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "do_nothing"


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = CustomerChurnEnv(base_url=f"https://p1108-customer-churn-env.hf.space")

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with env:
            result = await env.reset()
            obs = result.observation
            last_reward = 0.0

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                obs_dict = {
                    "monthly_charges": obs.monthly_charges,
                    "tenure_months": obs.tenure_months,
                    "complaint_count": obs.complaint_count,
                    "contract_type": obs.contract_type,
                }

                action = get_model_action(client, step, obs_dict, last_reward, history)
                result = await env.step(CustomerChurnAction(action_type=action))

                reward = result.reward or 0.0
                done = result.done
                error = None

                rewards.append(reward)
                steps_taken = step
                last_reward = reward
                obs = result.observation

                history.append(f"Step {step}: action={action} reward={reward:.2f}")
                log_step(step=step, action=action, reward=reward, done=done, error=error)

                if done:
                    break

        score = sum(rewards) / MAX_STEPS if MAX_STEPS > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())