---
title: Customer Churn Retention Environment
emoji: 🎯 
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - customer-churn
  - retention
  - business
---

# Customer Churn Retention Environment

An OpenEnv reinforcement learning environment where an AI agent learns to retain at-risk customers by choosing optimal retention strategies across easy, medium, and hard scenarios.

## Problem Statement

Customer churn costs companies billions annually. This environment trains an AI agent to act as a retention specialist — analyzing customer profiles and deciding the best retention action before the customer leaves. Unlike traditional ML models that only predict churn, this environment teaches an agent WHAT TO DO about it.

## Quick Start
```python
from customer_churn_env import CustomerChurnAction, CustomerChurnEnv

with CustomerChurnEnv(base_url="https://qwerty1108-customer-churn-env.hf.space") as env:
    result = env.reset()
    print(f"Customer: {result.observation}")

    result = env.step(CustomerChurnAction(action_type="free_upgrade"))
    print(f"Reward: {result.reward}")
```

## Action Space

| Action | Description | Best For |
|--------|-------------|----------|
| `free_upgrade` | Give customer a free plan upgrade | New/first-time customers |
| `offer_discount` | Offer 10% discount | Medium risk customers |
| `personal_call` | Call the customer directly | High risk customers |
| `do_nothing` | Take no action | Low risk customers |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `monthly_charges` | float | Customer's monthly bill (50-200) |
| `tenure_months` | int | How long they've been a customer |
| `complaint_count` | int | Number of complaints filed (0-5) |
| `contract_type` | str | monthly or annual |

## Tasks

| Task | Difficulty | Customer Profile | Correct Action | Score |
|------|-----------|-----------------|----------------|-------|
| Easy | ⭐ | New customer, high complaints, monthly contract | free_upgrade | 0.0-1.0 |
| Medium | ⭐⭐ | Mid-tenure, some churn signals, moderate charges | offer_discount | 0.0-1.0 |
| Hard | ⭐⭐⭐ | Long tenure, high charges, many complaints | personal_call | 0.0-1.0 |

## Reward Function

- `1.0` → Agent picks the correct retention action
- `0.5` → Agent picks partially correct action (hard task only: offer_discount)
- `0.0` → Agent picks wrong action

Rewards are provided at each step, not just at episode end — enabling the agent to learn progressively.

## Baseline Scores

| Task | Random Agent | Optimal Agent |
|------|-------------|---------------|
| Easy | 0.25 | 1.0 |
| Medium | 0.25 | 1.0 |
| Hard | 0.25 | 1.0 |

## Real World Utility

This environment models a genuine business problem. Retention specialists at telecom, SaaS, and subscription companies face exactly these decisions daily. An agent trained here could:
- Automate retention decisions at scale
- Reduce human error in customer handling
- Optimize retention budgets by avoiding unnecessary discounts

## Project Structure

customer_churn_env/
├── __init__.py
├── README.md
├── openenv.yaml
├── models.py              # Action, Observation, State models
├── client.py              # CustomerChurnEnv client
└── server/
    ├── app.py             # FastAPI application
    ├── customer_churn_env_environment.py  # Core environment logic
    └── Dockerfile
