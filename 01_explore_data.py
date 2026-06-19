from datasets import load_dataset
import pandas as pd

ds = load_dataset("wandb/RAGTruth-processed")
print(ds)

split_name = list(ds.keys())[0]
df = ds[split_name].to_pandas()
print(df.columns.tolist())
print(df.head(3))
print(df.shape)