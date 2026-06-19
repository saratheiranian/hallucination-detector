from datasets import load_dataset
import pandas as pd

pd.set_option('display.max_colwidth', 300)

ds = load_dataset("wandb/RAGTruth-processed")
df = ds["train"].to_pandas()

# Look at one full example in detail
example = df.iloc[0]
print("=== QUERY ===")
print(example["query"]) 
print("\n=== CONTEXT (truncated) ===")
print(str(example["context"])[:500])
print("\n=== OUTPUT (the LLM's answer) ===")
print(example["output"])
print("\n=== TASK TYPE ===")
print(example["task_type"])
print("\n=== HALLUCINATION LABELS (raw) ===")
print(example["hallucination_labels"])
print("\n=== HALLUCINATION LABELS (processed) ===")
print(example["hallucination_labels_processed"])

# Check task type distribution
print("\n=== TASK TYPE COUNTS ===")
print(df["task_type"].value_counts())

# Check how labels are structured across a few rows
print("\n=== Sample of labels_processed across 5 rows ===")
for i in range(5):
    print(f"Row {i}: {df.iloc[i]['hallucination_labels_processed']}")