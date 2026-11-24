SYSTEM_PROMPT = r"""Your task is to follow a systematic, thorough reasoning process before providing the final solution. 
This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. 
Structure your response into two sections: Thought and Solution. In the Thought section, present your reasoning using the format: \"<think>\n {thoughts} </think>\n\". 
Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas.
After \"</think>\n,\" in the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section."""


def apply_prompt_template(text: str) -> str:
    return f"""Solve the following math problem step by step. Put your answer inside \\boxed{{}}.

{text}

Remember to put your answer inside \\boxed{{}}."""
