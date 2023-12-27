import os, requests, json
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

class OpenRouter(LLM):
    model: str

    @property
    def _llm_type(self) -> str:
        return f"openrouter:{self.model}"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
        headers = {
            'Authorization': f'Bearer {OPENROUTER_API_KEY}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': self.model,
            'messages': [
                {'role': 'user', 'content': prompt}
            ]
        }
        # Output example: {'choices': [{'message': {'role': 'assistant', 'content': "I am OpenAI's artificial intelligence model called GPT-3."}}], 'model': 'gpt-4-32k-0613', 'usage': {'prompt_tokens': 11, 'completion_tokens': 14, 'total_tokens': 25}, 'id': 'gen-e4MSuTT1v2wvrYFNFunhumsIawaI'}
        response = requests.post('https://openrouter.ai/api/v1/chat/completions', headers=headers, data=json.dumps(data))
        output = response.json()['choices'][0]['message']['content']

        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return output

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model": self.model}