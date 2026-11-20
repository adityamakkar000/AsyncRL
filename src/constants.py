import os

# GS_BUCKET = "gs://arl-us-east-5"
GS_BUCKET = "gs://arl-us-central-1"
RUN_PATH = f"{GS_BUCKET}/runs"

HF_CHECKPOINT_PATH = os.path.expanduser("~/hf_params")

INTERUPT_THINKING_PHARSE = "Okay, time is up. Let me stop thinking and formulate a final answer now. \n\n</think>"

SYSTEM_PROMPT = r"""Your task is to follow a systematic, thorough reasoning process before providing the final solution. 
This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. 
Structure your response into two sections: Thought and Solution. In the Thought section, present your reasoning using the format: \"<think>\n {thoughts} </think>\n\". 
Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas.
After \"</think>\n,\" in the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section."""
