import asyncio
import os
import textwrap
import httpx
from typing import List, Optional
from openai import OpenAI

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("CUSTOMER_CHURN_TASK", "customer-churn")
BENCHMARK = os.getenv("CUSTOMER_CHURN_BENCHMARK", "customer_churn_env")
ENV_URL = "https://p1108-customer-churn-env.hf.space"
MAX_STEPS = 3
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI retention specialist for a subscription company.
    You will see a customer profile and must choose the best retention action.
    Available actions:
    - free_upgrade: Give a free plan upgrade (best for new customers with complaints)
    - offer_discount: Offer a 10% discount (best for medium risk customers)
    - personal_call: Call the customer directly (best for high risk, high value customers)
    - do_nothing: Take no action (best for low risk customers)
    Reply with exactly one action string — no quotes, no explanation, just the action.
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, obs: dict, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(f"""
        Step: {step}
        Customer Profile:
          Monthly Charges: {obs.get('monthly_charges', 0)}
          Tenure Months: {obs.get('tenure_months', 0)}
          Complaint Count: {obs.get('complaint_count', 0)}
          Contract Type: {obs.get('contract_type', 'monthly')}
        Last reward: {last_reward:.2f}
        Previous steps: {history_block}
        Choose your action:
    """).strip()


def get_model_action(client: OpenAI, step: int, obs: dict, last_reward: float, history: List[str]) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(step, obs, last_reward, history)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        for action in ["free_upgrade", "offer_discount", "personal_call", "do_nothing"]:
            if action in text:
                return action
        return "do_nothing"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "do_nothing"


async def main() -> None:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
        # Using a dummy API key if none is provided to avoid unhandled OpenAIError
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")

        async with httpx.AsyncClient(timeout=30) as http:
            # Reset
            reset_resp = await http.post(f"{ENV_URL}/reset")
            reset_resp.raise_for_status()
            reset_data = reset_resp.json()
            obs = reset_data.get("observation", reset_data)
            done = reset_data.get("done", False)
            last_reward = 0.0

            for step in range(1, MAX_STEPS + 1):
                try:
                    if done:
                        break

                    action = get_model_action(client, step, obs, last_reward, history)

                    step_resp = await http.post(
                        f"{ENV_URL}/step",
                        json={"action": {"action_type": action}},
                        headers={"Content-Type": "application/json"}
                    )
                    step_resp.raise_for_status()
                    step_data = step_resp.json()

                    obs = step_data.get("observation", obs)
                    reward = float(step_data.get("reward", 0.0))
                    done = step_data.get("done", False)
                    error = None

                except Exception as e:
                    error = str(e)
                    reward = 0.0
                    done = True
                    action = "do_nothing"
                    print(f"[DEBUG] Step error: {e}", flush=True)

                rewards.append(reward)
                steps_taken = step
                last_reward = reward
                history.append(f"Step {step}: action={action} reward={reward:.2f}")
                log_step(step=step, action=action, reward=reward, done=done, error=error)

                if done:
                    break

        score = sum(rewards) / MAX_STEPS if MAX_STEPS > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Main error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())