import pandas as pd

test_df = pd.read_csv("test_clean.csv")

# Take a small, balanced sample: 15 hallucinated + 15 clean
hallucinated = test_df[test_df["is_hallucination"] == 1].sample(15, random_state=42)
clean = test_df[test_df["is_hallucination"] == 0].sample(15, random_state=42)

dev_set = pd.concat([hallucinated, clean]).sample(frac=1, random_state=42).reset_index(drop=True)

dev_set.to_csv("dev_set.csv", index=False)
print(f"Dev set created: {len(dev_set)} examples")
print(dev_set["is_hallucination"].value_counts())
