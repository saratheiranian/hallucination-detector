import pandas as pd
import os
import time
from dotenv import load_dotenv
from anthropic import Anthropic
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

eval_set = pd.read_csv("eval_set_300.csv")

JUDGE_PROMPT = """You are a strict fact-checker. You will be given a CONTEXT (a source document) and an OUTPUT (a generated answer based on that context).

Your job: determine if the OUTPUT contains any claims that are NOT supported by the CONTEXT, or that CONTRADICT the CONTEXT.

Respond with ONLY one word: "HALLUCINATION" if the output contains unsupported or contradictory claims, or "FAITHFUL" if everything in the output is fully supported by the context. Do not explain, just give the one word.

CONTEXT:
{context}

OUTPUT:
{output}
"""

def judge(context, output, max_retries=3):
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=10,
                messages=[{
                    "role": "user",
                    "content": JUDGE_PROMPT.format(context=context, output=output)
                }]
            )
            response_text = message.content[0].text.strip().upper()
            return 1 if "HALLUCINATION" in response_text else 0
        except Exception as e:
            print(f"  Retry {attempt+1}/{max_retries} due to: {e}")
            time.sleep(2 * (attempt + 1))
    print("  Failed after retries, defaulting prediction to None")
    return None

predictions = []
print(f"Running LLM judge on {len(eval_set)} examples...\n")

for i, row in eval_set.iterrows():
    pred = judge(row["context"], row["output"])
    predictions.append(pred)
    if (i + 1) % 20 == 0 or (i + 1) == len(eval_set):
        print(f"Progress: {i+1}/{len(eval_set)}")

eval_set["llm_judge_prediction"] = predictions
eval_set.to_csv("eval_set_300_with_predictions.csv", index=False)

scored_df = eval_set.dropna(subset=["llm_judge_prediction"])
if len(scored_df) < len(eval_set):
    print(f"\nWarning: {len(eval_set) - len(scored_df)} examples failed and were excluded from scoring.")

y_true = scored_df["is_hallucination"]
y_pred = scored_df["llm_judge_prediction"]

print("\n=== RESULTS (LLM-as-Judge, n={}) ===".format(len(scored_df)))
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.2%}")
print(f"Precision: {precision_score(y_true, y_pred):.2%}")
print(f"Recall:    {recall_score(y_true, y_pred):.2%}")
print(f"F1:        {f1_score(y_true, y_pred):.2%}")
print(f"\nConfusion matrix:\n{confusion_matrix(y_true, y_pred)}")
