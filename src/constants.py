import os

GS_BUCKET = "gs://arl-experiments"
CHECKPOINTS = "checkpoints"
DATA = "data"
PROFILE = "profile"

HF_CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hf_params")

INTERUPT_THINKING_PHARSE = "Okay, time is up. Let me stop thinking and formulate a final answer now. \n\n</think>"

SYSTEM_PROMPT = r"""Your task is to follow a systematic, thorough reasoning process before providing the final solution. 
This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. 
Structure your response into two sections: Thought and Solution. In the Thought section, present your reasoning using the format: \"<think>\n {thoughts} </think>\n\". 
Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas.
After \"</think>\n,\" in the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section."""
