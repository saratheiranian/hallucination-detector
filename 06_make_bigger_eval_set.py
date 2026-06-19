import pandas as pd

test_df = pd.read_csv("test_clean.csv")

# 150 hallucinated + 150 clean = 300 total
hallucinated = test_df[test_df["is_hallucination"] == 1].sample(150, random_state=42)
clean = test_df[test_df["is_hallucination"] == 0].sample(150, random_state=42)

eval_set = pd.concat([hallucinated, clean]).sample(frac=1, random_state=42).reset_index(drop=True)

eval_set.to_csv("eval_set_300.csv", index=False)
print(f"Eval set created: {len(eval_set)} examples")
print(eval_set["is_hallucination"].value_counts())
