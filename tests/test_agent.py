import matplotlib
import pandas as pd

matplotlib.use("Agg")

from packages.agent import RLAgent


def _agent(policy="q_learning", action_policy="epsilon_greedy") -> RLAgent:
    return RLAgent(
        stock_no="TEST",
        len_avg_days=2,
        policy=policy,
        action_policy=action_policy,
        epsilon=0,
        episodes=1,
    )


def test_q_learning_update_uses_best_next_action():
    agent = _agent(policy="q_learning")
    state = agent.STATES[0]
    next_state = agent.STATES[1]
    agent.Q_table[next_state]["buy"] = 2.0
    agent.Q_table[next_state]["sell"] = 1.0

    agent._update_Q_table(state, "hold", reward=1.0, next_state=next_state, next_action=None)

    assert agent.Q_table[state]["hold"] == agent.alpha * (1.0 + agent.gamma * 2.0)


def test_sarsa_requires_next_action_when_not_done():
    agent = _agent(policy="sarsa")
    state = agent.STATES[0]

    try:
        agent._update_Q_table(state, "hold", reward=1.0, next_state=state, next_action=None)
    except ValueError as exc:
        assert "SARSA" in str(exc)
    else:
        raise AssertionError("SARSA update should require next_action when not done")


def test_epsilon_greedy_evaluation_chooses_highest_q_value():
    agent = _agent(action_policy="epsilon_greedy")
    state = agent.STATES[0]
    agent.Q_table[state].update({"buy": 0.1, "sell": 0.9, "hold": 0.2})

    assert agent._choose_action(state, evaluate=True) == "sell"


def test_evaluate_learning_keeps_cash_when_policy_always_holds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = _agent(action_policy="epsilon_greedy")
    for state in agent.Q_table:
        agent.Q_table[state].update({"buy": 0.0, "sell": 0.0, "hold": 1.0})

    df = pd.DataFrame(
        {
            "Open": [100.0, 110.0, 120.0],
            "Close": [110.0, 120.0, 130.0],
            "Trend_0": ["up", "up", "up"],
            "Trend_1": ["up", "up", "up"],
            "volume_state": ["normal", "normal", "normal"],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )

    final_value = agent.evaluate_learning(df, initial_cash=10_000)

    assert final_value == 10_000
    assert (tmp_path / "images" / "q_learning_epsilon_greedy.png").exists()
    assert (tmp_path / "model_weights" / "q_learning_epsilon_greedy.pkl").exists()
