import pandas as pd

from packages import preprocess


class _SplitTicker:
    splits = pd.Series(
        [4.0],
        index=pd.DatetimeIndex(["2020-08-31"], name="Date"),
    )


def _split_sample_frame() -> pd.DataFrame:
    dates = pd.to_datetime(["2020-08-28", "2020-08-31", "2020-09-01"])
    columns = pd.MultiIndex.from_product(
        [["Close", "Open", "Volume"], ["AAPL"]],
        names=["Price", "Ticker"],
    )
    return pd.DataFrame(
        [
            [100.0, 99.0, 1_000],
            [25.0, 24.0, 1_100],
            [26.0, 25.0, 1_200],
        ],
        index=dates,
        columns=columns,
    )


def test_preprocess_does_not_reapply_splits_to_auto_adjusted_data_by_default(monkeypatch):
    monkeypatch.setattr(preprocess.yf, "Ticker", lambda _: _SplitTicker())

    result = preprocess.prerpocess("AAPL", _split_sample_frame(), avg_days=[1])

    assert result.loc[pd.Timestamp("2020-09-01"), "Open"] == 25.0
    assert result.loc[pd.Timestamp("2020-09-01"), "Close"] == 26.0


def test_preprocess_can_optionally_adjust_raw_post_split_prices(monkeypatch):
    monkeypatch.setattr(preprocess.yf, "Ticker", lambda _: _SplitTicker())

    result = preprocess.prerpocess(
        "AAPL",
        _split_sample_frame(),
        avg_days=[1],
        adjust_splits=True,
    )

    assert result.loc[pd.Timestamp("2020-08-31"), "Open"] == 96.0
    assert result.loc[pd.Timestamp("2020-08-31"), "Close"] == 100.0
    assert result.loc[pd.Timestamp("2020-09-01"), "Open"] == 100.0
    assert result.loc[pd.Timestamp("2020-09-01"), "Close"] == 104.0
