"""LLM 추론 래퍼.

3가지 백엔드 지원:
  1. local: HuggingFace Transformers (Qwen2.5-14B-4bit, ~10GB VRAM)
  2. anthropic: Anthropic Claude API
  3. openai: OpenAI GPT API
"""

import json
import os
import time
from dataclasses import dataclass


@dataclass
class LLMConfig:
    model_name: str = "Qwen/Qwen2.5-14B-Instruct"
    backend: str = "local"  # "local", "anthropic", "openai"
    max_new_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 0.9
    use_4bit: bool = True
    device: str = "auto"


def _parse_json_response(response: str) -> dict:
    """LLM 응답에서 JSON 추출. Handles chat-template leakage and trailing text."""
    import re

    # Try ```json ... ``` block first
    m = re.search(r"```json\s*(.*?)```", response, re.DOTALL)
    if not m:
        m = re.search(r"```\s*(.*?)```", response, re.DOTALL)
    if m:
        json_str = m.group(1).strip()
    else:
        # Strip leading chat-template tokens (e.g., "assistant\n") by locating
        # the first '{' and extracting the balanced braces span.
        s = response
        start = s.find("{")
        if start < 0:
            return {"raw_response": response, "parse_error": True}
        depth = 0
        end = -1
        in_str = False
        esc = False
        for i, ch in enumerate(s[start:], start=start):
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end < 0:
            return {"raw_response": response, "parse_error": True}
        json_str = s[start:end]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"raw_response": response, "parse_error": True}


class AnthropicInference:
    """Anthropic Claude API 추론."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None

    def load(self) -> None:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Anthropic(api_key=api_key)
        print(f"Anthropic API ready (model: {self.config.model_name})")

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        if self.client is None:
            self.load()

        kwargs = {
            "model": self.config.model_name,
            "max_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)
        return response.content[0].text.strip()

    def generate_json(self, prompt: str, system_prompt: str | None = None) -> dict:
        response = self.generate(prompt, system_prompt)
        return _parse_json_response(response)


class OpenAIInference:
    """OpenAI GPT API 추론."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None

    def load(self) -> None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)
        print(f"OpenAI API ready (model: {self.config.model_name})")

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        if self.client is None:
            self.load()

        # Sanitize: remove null bytes and ensure valid UTF-8
        prompt = prompt.replace("\x00", "").encode("utf-8", errors="replace").decode("utf-8")
        if system_prompt:
            system_prompt = system_prompt.replace("\x00", "").encode("utf-8", errors="replace").decode("utf-8")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        import time
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    max_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < 2:
                    print(f"  API error (attempt {attempt+1}): {e}, retrying...")
                    time.sleep(2 ** attempt)
                else:
                    raise

    def generate_json(self, prompt: str, system_prompt: str | None = None) -> dict:
        try:
            response = self.generate(prompt, system_prompt)
            return _parse_json_response(response)
        except Exception as e:
            return {"raw_response": str(e), "parse_error": True, "api_error": True}


class LocalInference:
    """로컬 HuggingFace Transformers 추론."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"Loading {self.config.model_name}...")
        start = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True,
        )

        if self.config.use_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quant_config,
                device_map=self.config.device,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map=self.config.device,
                trust_remote_code=True,
            )

        elapsed = time.time() - start
        print(f"Model loaded in {elapsed:.1f}s")

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU memory: {mem:.1f} GB")

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        import torch

        if self.model is None:
            self.load()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate_json(self, prompt: str, system_prompt: str | None = None) -> dict:
        response = self.generate(prompt, system_prompt)
        return _parse_json_response(response)


class LLMInference:
    """통합 LLM 추론 인터페이스."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            if self.config.backend == "anthropic":
                self._engine = AnthropicInference(self.config)
            elif self.config.backend == "openai":
                self._engine = OpenAIInference(self.config)
            else:
                self._engine = LocalInference(self.config)
        return self._engine

    def load(self) -> None:
        self._get_engine().load()

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return self._get_engine().generate(prompt, system_prompt)

    def generate_json(self, prompt: str, system_prompt: str | None = None) -> dict:
        return self._get_engine().generate_json(prompt, system_prompt)

    def generate_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        batch_size: int = 4,
    ) -> list[str]:
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            for prompt in batch:
                result = self.generate(prompt, system_prompt)
                results.append(result)
            if i + batch_size < len(prompts):
                print(f"  Processed {min(i + batch_size, len(prompts))}/{len(prompts)}")
        return results
