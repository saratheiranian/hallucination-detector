import pandas as pd
import os
from dotenv import load_dotenv
from anthropic import Anthropic
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

dev_set = pd.read_csv("dev_set.csv")

JUDGE_PROMPT = """You are a strict fact-checker. You will be given a CONTEXT (a source document) and an OUTPUT (a generated answer based on that context).

Your job: determine if the OUTPUT contains any claims that are NOT supported by the CONTEXT, or that CONTRADICT the CONTEXT.

Respond with ONLY one word: "HALLUCINATION" if the output contains unsupported or contradictory claims, or "FAITHFUL" if everything in the output is fully supported by the context. Do not explain, just give the one word.

CONTEXT:
{context}

OUTPUT:
{output}
"""

def judge(context, output):
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

predictions = []
print("Running LLM judge on dev set...\n")
for i, row in dev_set.iterrows():
    pred = judge(row["context"], row["output"])
    predictions.append(pred)
    print(f"Example {i+1}/{len(dev_set)} | True: {row['is_hallucination']} | Predicted: {pred}")

dev_set["llm_judge_prediction"] = predictions
dev_set.to_csv("dev_set_with_predictions.csv", index=False)

y_true = dev_set["is_hallucination"]
y_pred = dev_set["llm_judge_prediction"]

print("\n=== RESULTS ===")
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.2%}")
print(f"Precision: {precision_score(y_true, y_pred):.2%}")
print(f"Recall:    {recall_score(y_true, y_pred):.2%}")
print(f"F1:        {f1_score(y_true, y_pred):.2%}")
print(f"\nConfusion matrix:\n{confusion_matrix(y_true, y_pred)}")
