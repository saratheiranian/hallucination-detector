from datasets import load_dataset
import pandas as pd

ds = load_dataset("wandb/RAGTruth-processed")

def process_split(split_name):
    df = ds[split_name].to_pandas()

    def get_label(row):
        labels = row["hallucination_labels_processed"]
        return 1 if (labels["evident_conflict"] == 1 or labels["baseless_info"] == 1) else 0

    df["is_hallucination"] = df.apply(get_label, axis=1)

    clean_df = df[["id", "context", "output", "query", "task_type", "is_hallucination"]].copy()
    return clean_df

train_df = process_split("train")
test_df = process_split("test")

print("=== TRAIN ===")
print(train_df["is_hallucination"].value_counts())
print(f"Hallucination rate: {train_df['is_hallucination'].mean():.2%}")

print("\n=== TEST ===")
print(test_df["is_hallucination"].value_counts())
print(f"Hallucination rate: {test_df['is_hallucination'].mean():.2%}")

train_df.to_csv("train_clean.csv", index=False)
test_df.to_csv("test_clean.csv", index=False)
print("\nSaved train_clean.csv and test_clean.csv")
