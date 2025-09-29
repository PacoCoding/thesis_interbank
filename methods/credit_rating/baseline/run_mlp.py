import argparse
from methods.credit_rating.baseline.mlp_baseline import run_walkforward_mlp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_quarter", default="2017Q4")
    parser.add_argument("--end_quarter", default=None)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    run_walkforward_mlp(
        start_quarter=args.start_quarter,
        end_quarter=args.end_quarter,
        runs=args.runs,
        base_seed=42,
        show_cm=True,
    )
