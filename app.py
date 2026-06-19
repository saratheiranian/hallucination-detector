import streamlit as st
import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

JUDGE_PROMPT = """You are a strict fact-checker. You will be given a CONTEXT (a source document) and an OUTPUT (a generated answer based on that context).

Your job: determine if the OUTPUT contains any claims that are NOT supported by the CONTEXT, or that CONTRADICT the CONTEXT.

First, list each claim in the OUTPUT and whether it is supported, contradicted, or not mentioned in the CONTEXT.
Then on the final line, write ONLY one word: "HALLUCINATION" or "FAITHFUL".

CONTEXT:
{context}

OUTPUT:
{output}
"""

def judge(context, output):
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(context=context, output=output)
        }]
    )
    full_text = message.content[0].text.strip()
    last_line = full_text.strip().splitlines()[-1].upper()
    verdict = "HALLUCINATION" if "HALLUCINATION" in last_line else "FAITHFUL"
    return verdict, full_text

st.set_page_config(page_title="Hallucination Detector", page_icon="🔍", layout="centered")

st.title("🔍 Hallucination Detector")
st.caption("Paste a source context and an AI-generated answer. Find out if the answer is fully supported by the source.")

with st.form("detector_form"):
    context_input = st.text_area("Source context (the document / facts the answer should be based on)", height=200,
                                   placeholder="Paste the source document here...")
    output_input = st.text_area("Generated answer (to check for hallucinations)", height=150,
                                  placeholder="Paste the AI-generated answer here...")
    submitted = st.form_submit_button("Check for hallucinations", use_container_width=True)

if submitted:
    if not context_input.strip() or not output_input.strip():
        st.warning("Please fill in both fields.")
    else:
        with st.spinner("Checking against source..."):
            verdict, reasoning = judge(context_input, output_input)

        if verdict == "HALLUCINATION":
            st.error("⚠️ HALLUCINATION DETECTED — this answer contains claims not supported by the source.")
        else:
            st.success("✅ FAITHFUL — this answer is fully supported by the source.")

        with st.expander("See reasoning"):
            st.write(reasoning)

st.divider()
st.caption("Built by Sara Ghassemi · Powered by Claude (Anthropic) · Benchmarked on the RAGTruth dataset")
