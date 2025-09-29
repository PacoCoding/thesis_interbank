import argparse
from methods.credit_rating.baseline.lstm_baseline import run_walkforward_lstm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_quarter", default = "2017Q4")
    parser.add_argument("--end_quarter", default = None)
    parser.add_argument("--show_cm", default = True)
    parser.add_argument("--runs", default = 4)
    parser.add_argument("--base_seed", default = 42)
    run_walkforward_lstm(
        start_quarter=args.start_quarter,
        end_quarter=args.end_quarter,
        show_cm=args.show_cm,
        runs=args.runs,
        base_seed=arge.base_seed
    )
