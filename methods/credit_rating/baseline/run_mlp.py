from methods.credit_rating.baseline.mlp_baseline import run_walkforward_mlp

if __name__ == "__main__":
    run_walkforward_mlp(start_quarter="2017Q4", end_quarter=None, runs=4, base_seed=42)

