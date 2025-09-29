from methods.credit_rating.baseline.lstm_baseline import run_walkforward_lstm

if __name__ == "__main__":
    run_walkforward_lstm(
        start_quarter="2017Q4",
        end_quarter=None,
        show_cm=True,
        runs=4,
        base_seed=42
    )
