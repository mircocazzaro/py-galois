
import os, json, time, random
from typing import List, Dict, Any, Tuple, Optional
from openai import OpenAI, AzureOpenAI
from openai import RateLimitError, APIError
import requests
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters


WATSONX_API_KEY=""
WATSONX_URL="https://us-south.ml.cloud.ibm.com"
WATSONX_PROJECT_ID="910d0796-88b3-4582-a8e6-5a2c72e25a98"
WATSONX_MODEL_ID="meta-llama/llama-3-3-70b-instruct"


#gianmaria 4o-mini
AZURE_OPENAI_API_KEY=''
AZURE_OPENAI_BASE_URL = "https://kbest.openai.azure.com/openai/v1/"
AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini-2" 
AZURE_OPENAI_API_VERSION='2024-12-01-preview'

#mirco llama 8b-70b
AZURE_INFERENCE_ENDPOINT="https://mirco-llama-3-1-70b-resource.services.ai.azure.com/openai/v1/"
AZURE_INFERENCE_KEY=""
AZURE_INFERENCE_MODEL="Meta-Llama-3.1-8B-Instruct"
LLM_SEED="7"  # optional but nice for determinism


# GROK SU AZURE
AZURE_GROK_API_KEY = ""
AZURE_GROK_TARGET_URI = (
    "https://2026-foundry.cognitiveservices.azure.com/"
    "openai/deployments/grok-4-1-fast-reasoning/"
    "chat/completions?api-version=2024-05-01-preview"
)
AZURE_GROK_DEPLOYMENT = "grok-4-1-fast-reasoning"


#gianmaria llama 8b - 70b
AZURE_INFERENCE_ENDPOINT="https://gianm-m6osfju4-eastus2.services.ai.azure.com/openai/v1/"
AZURE_INFERENCE_KEY=""
AZURE_INFERENCE_MODEL="Llama-3.3-70B-Instruct"
LLM_SEED="7"  # optional but nice for determinism

#openrouter
OPENROUTER_API_KEY=""
OPENROUTER_MODEL="meta-llama/llama-3.3-70b-instruct" 
#"openai/gpt-4o-mini" #example of an OpenAI model proxied via OpenRouter

class LLMResponse:
    def __init__(self, text: str, usage_tokens: int = 0, latency_s: float = 0.0):
        self.text = text
        self.usage_tokens = usage_tokens
        self.latency_s = latency_s

class BaseLLM:
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        raise NotImplementedError


class OllamaClient(BaseLLM):
    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://127.0.0.1:11434/v1",
        timeout: int = 18000,
    ):
        self.model = model
        self.base_url = os.getenv("OLLAMA_BASE_URL", base_url).rstrip("/")
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }

        # Local Ollama does not require auth, but many OpenAI-style clients
        # still send a dummy key such as "ollama".
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer ollama",
        }

        r = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()

        text = data["choices"][0]["message"]["content"]

        usage = data.get("usage", {})
        total_tokens = usage.get("total_tokens")

        return LLMResponse(text=text, usage_tokens=total_tokens)    

class OpenRouterClient(BaseLLM):
    """
    OpenRouter backend using the OpenAI Python SDK.

    Expects:
      - OPENROUTER_API_KEY         (required)
      - OPENROUTER_MODEL           (optional, default below)
      - OPENROUTER_SITE_URL        (optional, for HTTP-Referer)
      - OPENROUTER_SITE_NAME       (optional, for X-Title)
    """

    def __init__(self, model: Optional[str] = None):
        from openai import OpenAI  # type: ignore

        api_key = OPENROUTER_API_KEY
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is required to use OpenRouterClient."
            )

        # Pick your default OpenRouter model here
        # (example: an OpenAI model proxied via OpenRouter)
        self.model = OPENROUTER_MODEL

        # Optional but recommended headers for OpenRouter ranking/attribution
        site_url = os.environ.get("OPENROUTER_SITE_URL", "https://example.com")
        site_name = "GALOIS"

        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": site_url,
                "X-Title": site_name,
            },
        )

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        t0 = time.time()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,   # keep determinism aligned with the paper
            top_p=1.0,
            seed=int(LLM_SEED),
            max_tokens=10000,
            **kwargs,          # let callers add response_format etc.
        )

        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        total_tokens = 0
        if usage:
            total_tokens = (
                (getattr(usage, "prompt_tokens", 0) or 0)
                + (getattr(usage, "completion_tokens", 0) or 0)
            )

        return LLMResponse(
            text=text,
            usage_tokens=total_tokens,
            latency_s=(time.time() - t0),
        )


class AzureGrokClient(BaseLLM):
    def __init__(
        self,
        target_uri: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 300,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 2000,   # lower default
        seed: Optional[int] = None,
        max_retries: int = 5,
    ):
        self.target_uri = (target_uri or AZURE_GROK_TARGET_URI).strip()
        self.api_key = api_key or AZURE_GROK_API_KEY
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = int(seed) if seed is not None else int(LLM_SEED)
        self.max_retries = max_retries

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                out = []
                for x in content:
                    if isinstance(x, dict) and "text" in x:
                        out.append(x["text"])
                    else:
                        out.append(str(x))
                return "".join(out)
            return str(content or "")
        except Exception:
            return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _extract_usage_tokens(data: Dict[str, Any]) -> int:
        usage = data.get("usage", {}) or {}
        return (
            usage.get("total_tokens")
            or ((usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0))
            or 0
        )

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        payload = {
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
        }
        payload.update(kwargs)

        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.time()
            try:
                resp = requests.post(
                    self.target_uri,
                    headers=headers,
                    json=payload,
                    timeout=(30, self.timeout),  # connect, read
                )
                resp.raise_for_status()
                data = resp.json()
                return LLMResponse(
                    text=self._extract_text(data),
                    usage_tokens=self._extract_usage_tokens(data),
                    latency_s=(time.time() - t0),
                )

            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                last_err = e

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                body = e.response.text if e.response is not None else ""
                # retry only transient HTTP classes
                if status in {408, 429, 500, 502, 503, 504}:
                    last_err = RuntimeError(f"HTTP {status}: {body}")
                else:
                    raise RuntimeError(f"Azure Grok request failed: HTTP {status} - {body}") from e

            if attempt < self.max_retries:
                sleep_s = min(20, (2 ** (attempt - 1))) + random.uniform(0, 0.5)
                time.sleep(sleep_s)

        raise RuntimeError(f"Azure Grok request failed after {self.max_retries} attempts: {last_err}")

class WatsonxClient(BaseLLM):
    """
    IBM watsonx.ai backend using the official Python SDK.

    It expects messages in the same format as OpenAI chat:
    [{"role": "system"|"user"|"assistant", "content": "..."}]
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        project_id: Optional[str] = None,
        max_tokens: int = 10000,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.model_id = model_id or WATSONX_MODEL_ID
        if not self.model_id:
            raise ValueError("WATSONX_MODEL_ID (or model_id argument) must be set")

        api_key = api_key or WATSONX_API_KEY
        url = url or WATSONX_URL
        project_id = project_id or WATSONX_PROJECT_ID

        if not api_key or not url or not project_id:
            raise ValueError(
                "WATSONX_API_KEY, WATSONX_URL and WATSONX_PROJECT_ID must be set "
                "either via environment variables or constructor arguments."
            )

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed

        creds = Credentials(api_key=api_key, url=url)

        # TextChatParameters drives decoding (max_tokens, temperature, etc.) :contentReference[oaicite:2]{index=2}
        self._params = TextChatParameters(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            #seed=int(seed) if seed is not None else None,
            #response_format={"type": "json_object"},
        )

        # ModelInference.chat(...) will return an OpenAI-like response dict with choices[0].message.content :contentReference[oaicite:3]{index=3}
        self._model = ModelInference(
            model_id=self.model_id,
            params=self._params,
            credentials=creds,
            project_id=project_id,
        )

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
        t0 = time.time()

        # Extract which kwargs belong to TextChatParameters
        if hasattr(self._params, "model_dump"):
            base_dict = self._params.model_dump()
        else:
            base_dict = {k: v for k, v in self._params.__dict__.items() if not k.startswith("_")}

        param_keys = set(base_dict.keys())

        param_updates = {k: v for k, v in kwargs.items() if k in param_keys}
        other_kwargs  = {k: v for k, v in kwargs.items() if k not in param_keys}

        base_dict.update(param_updates)
        params = TextChatParameters(**base_dict)

        # ⬇⬇⬇ now forward tools/tool_choice/etc. as **other_kwargs
        resp = self._model.chat(messages=messages, params=params, **other_kwargs)

        try:
            text = resp["choices"][0]["message"]["content"]
        except Exception:
            text = str(resp)

        usage_tokens = 0
        try:
            if "usage" in resp:
                u = resp["usage"]
                usage_tokens = (
                    u.get("total_tokens")
                    or u.get("generated_token_count", 0) + u.get("input_token_count", 0)
                )
            elif "results" in resp:
                r0 = resp["results"][0]
                usage_tokens = (
                    r0.get("generated_token_count", 0)
                    + r0.get("input_token_count", 0)
                )
        except Exception:
            usage_tokens = 0

        latency_s = time.time() - t0
        return LLMResponse(text=text, usage_tokens=usage_tokens, latency_s=latency_s)

    
class FoundryOpenAIClient(BaseLLM):
    """Calls Azure AI Foundry (Serverless API) via OpenAI-compatible Chat Completions."""
    def __init__(self):
        base_url = AZURE_INFERENCE_ENDPOINT.rstrip("/")
        api_key  = AZURE_INFERENCE_KEY
        self.model = AZURE_INFERENCE_MODEL
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        t0 = time.time()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,   # paper: determinism
            top_p=1.0,
            seed=int(LLM_SEED),
            max_tokens=10000,
            # optional: response_format={"type":"json_object"} when we expect pure JSON
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        total_tokens = 0
        if usage:
            total_tokens = (getattr(usage, "prompt_tokens", 0) or 0) + (getattr(usage, "completion_tokens", 0) or 0)
        return LLMResponse(text=text, usage_tokens=total_tokens, latency_s=(time.time() - t0))

class OpenAIClient(BaseLLM):
    def __init__(self, model: Optional[str] = None, azure: bool = False):
        self.azure = azure
        self.model = model or "gpt-4o-mini"

        if self.azure:
            self._client = OpenAI(
                base_url=AZURE_OPENAI_BASE_URL,
                api_key=AZURE_OPENAI_API_KEY,
            )
            self.model = AZURE_OPENAI_DEPLOYMENT
        else:
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        start = time.time()

        # Allinea i parametri di decoding a quelli già usati negli altri backend
        # (es. Ollama / OpenRouter / Foundry) per avere comportamento più stabile
        # e confrontabile.
        request_kwargs = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": 100000,
        }
        request_kwargs.update(kwargs)

        if self.azure:
            # puoi passare direttamente stringa o lista di messaggi
            if len(messages) == 1 and messages[0]["role"] == "user":
                inp = messages[0]["content"]
            else:
                inp = [{"role": m["role"], "content": m["content"]} for m in messages]

            resp = self._client.responses.create(
                model=self.model,   # deployment name
                input=inp,
                **request_kwargs
            )
        else:
            inp = [{"role": m["role"], "content": m["content"]} for m in messages]
            resp = self._client.responses.create(
                model=self.model,
                input=inp,
                **request_kwargs
            )

        latency = time.time() - start
        content = getattr(resp, "output_text", None) or str(resp)

        usage_tokens = 0
        try:
            u = resp.usage
            usage_tokens = (
                getattr(u, "total_tokens", None)
                or (getattr(u, "input_tokens", 0) or 0)
                + (getattr(u, "output_tokens", 0) or 0)
            )
        except Exception:
            pass

        return LLMResponse(text=content, usage_tokens=usage_tokens, latency_s=latency)
        
class MockLLM(BaseLLM):
    """A mock client useful for dry runs; echoes back minimal JSON."""
    def __init__(self, canned: Optional[str] = None):
        self.canned = canned or "[]"
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        # Heuristic: if the last user message asks for confidence, return a JSON number
        last = messages[-1]["content"].lower()
        if "confidence" in last:
            return LLMResponse(text=json.dumps({"confidence": 0.6}), usage_tokens=1, latency_s=0.01)
        if "list more" in last and "empty" in last:
            return LLMResponse(text="[]", usage_tokens=1, latency_s=0.01)
        return LLMResponse(text=self.canned, usage_tokens=1, latency_s=0.01)
