"""

This file is here because the local completions from the lm harnesses doesn't works since it
sends a seed in the paylod which breaks vllm tpu. Thus this is just a copy paste that removes
the seed in the payload. Orginal code is here:
https://github.com/EleutherAI/lm-evaluation-harness/blob/c1c4bea3777f73e188395264083adcf454913344/lm_eval/models/openai_completions.py#L16-L138

"""

import os
from operator import itemgetter
from typing import Dict, List, Optional, Tuple, Union

from lm_eval.models.api_models import TemplateAPI
from lm_eval.models.utils import handle_stop_sequences

from src.constants import IP, PORT

URL = f"http://{IP}:{PORT}/v1/completions"


class LocalModelEval(TemplateAPI):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            base_url=URL,
            tokenizer_backend="huggingface",
            verify_certificate=True,
            ca_cert_path=None,
            auth_token=None,
            **kwargs,
        )

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        generate=False,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        eos=None,
        **kwargs,
    ) -> dict:
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
        return {
            "prompt": messages,
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop,
            **gen_kwargs,
        }

    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] | None = None,
        ctxlen: List[int] | None = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            for choice, ctxlen in zip(sorted(out["choices"], key=itemgetter("index")), ctxlen):
                assert ctxlen > 0, "Context length must be greater than 0"
                logprobs = sum(choice["logprobs"]["token_logprobs"][ctxlen:-1])
                tokens_logprobs = choice["logprobs"]["token_logprobs"][ctxlen:-1]
                top_logprobs = choice["logprobs"]["top_logprobs"][ctxlen:-1]
                is_greedy = True
                for tok, top in zip(tokens_logprobs, top_logprobs):
                    if tok != max(top.values()):
                        is_greedy = False
                        break
                res.append((logprobs, is_greedy))
        return res

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            tmp = [None] * len(out["choices"])
            for choices in out["choices"]:
                tmp[choices["index"]] = choices["text"]
            res = res + tmp
        return res

    @property
    def api_key(self):
        return os.environ.get("OPENAI_API_KEY", "")
