import pandas as pd

from packages import preprocess


class _FakeTicker:
    splits = pd.Series(dtype=float)


def _sample_yfinance_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    columns = pd.MultiIndex.from_product(
        [["Close", "Open", "Volume"], ["AAPL"]],
        names=["Price", "Ticker"],
    )
    return pd.DataFrame(
        [
            [100.0, 99.0, 1_000],
            [101.0, 0.0, 1_100],
            [103.0, 102.0, 0],
            [102.0, 101.0, 1_300],
            [104.0, 103.0, 1_400],
        ],
        index=dates,
        columns=columns,
    )


def test_preprocess_builds_trend_and_volume_states(monkeypatch):
    monkeypatch.setattr(preprocess.yf, "Ticker", lambda _: _FakeTicker())

    result = preprocess.prerpocess("AAPL", _sample_yfinance_frame(), avg_days=[2, 3])

    assert list(result.columns) == ["Open", "Close", "Trend_0", "Trend_1", "volume_state"]
    assert len(result) == 3
    assert set(result["Trend_0"]).issubset({"up", "down", "stable"})
    assert set(result["Trend_1"]).issubset({"up", "down", "stable"})
    assert set(result["volume_state"]).issubset({"high", "low", "normal"})
    assert (result["Open"] != 0).all()


def test_deep_agent_preprocess_one_hot_encodes_categorical_states():
    df = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "Close": [101.0, 102.0],
            "Trend_0": ["up", "down"],
            "Trend_1": ["stable", "up"],
            "volume_state": ["high", "low"],
        }
    )

    result = preprocess.deep_agent_preprocess(df)

    assert result.columns[0:2].tolist() == ["Open", "Close"]
    assert result["Open"].tolist() == [100.0, 101.0]
    assert result["Close"].tolist() == [101.0, 102.0]
    encoded_cols = [col for col in result.columns if col not in {"Open", "Close"}]
    assert encoded_cols
    assert set(result[encoded_cols].to_numpy().ravel()).issubset({0, 1})
