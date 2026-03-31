import os
from typing import Dict, Any, Union, List

from src.processing.image import encode_image, get_mime_type

try:
    from google import genai
    from google.genai import types as gemini_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class BaseModel:
    """Base class for model wrappers."""

    def generate(self, prompt: Union[str, List[Dict]], system_instruction: str, config: Dict[str, Any]) -> str:
        """Generate response from model."""
        raise NotImplementedError


class GeminiModel(BaseModel):
    """Google Gemini model wrapper."""

    def __init__(self, api_key: str):
        if not GENAI_AVAILABLE:
            raise ImportError("Please install google-genai: pip install google-genai")
        if not api_key:
            raise ValueError("Missing Google API Key. Set GOOGLE_API_KEY environment variable.")
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: Union[str, List[Dict]], system_instruction: str, config: Dict[str, Any]) -> str:
        if "model_name" not in config:
            raise KeyError("Missing 'model_name' in config")

        contents = []
        if isinstance(prompt, list):
            for item in prompt:
                if item["type"] == "text":
                    contents.append(item["text"])
                elif item["type"] == "image":
                    path = item["path"]
                    if not path.exists():
                        raise FileNotFoundError(f"Image not found: {path}")
                    with open(path, "rb") as f:
                        img_bytes = f.read()
                    contents.append(
                        gemini_types.Part.from_bytes(
                            data=img_bytes,
                            mime_type=get_mime_type(path)
                        )
                    )
        else:
            contents = prompt

        response = self.client.models.generate_content(
            model=config["model_name"],
            contents=contents,
            config=gemini_types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=config.get("temperature", 0.0),
                top_p=config.get("top_p", 1.0),
            )
        )

        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        return "ERROR_GEMINI_EMPTY"


class ClaudeModel(BaseModel):
    """Anthropic Claude model wrapper."""

    def __init__(self, api_key: str):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Please install anthropic: pip install anthropic")
        if not api_key:
            raise ValueError("Missing Anthropic API Key. Set ANTHROPIC_API_KEY environment variable.")
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: Union[str, List[Dict]], system_instruction: str, config: Dict[str, Any]) -> str:
        if "model_name" not in config:
            raise KeyError("Missing 'model_name' in config")

        final_content = []
        if isinstance(prompt, list):
            for item in prompt:
                if item["type"] == "text":
                    final_content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image":
                    path = item["path"]
                    b64_data = encode_image(path)
                    final_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": get_mime_type(path),
                            "data": b64_data
                        }
                    })
            messages = [{"role": "user", "content": final_content}]
        else:
            messages = [{"role": "user", "content": prompt}]

        response = self.client.messages.create(
            model=config["model_name"],
            max_tokens=4096,
            temperature=config.get("temperature", 0.0),
            system=system_instruction,
            messages=messages
        )

        if response.content:
            return response.content[0].text
        return "ERROR_CLAUDE_EMPTY"


class OpenAIModel(BaseModel):
    """OpenAI GPT model wrapper."""

    def __init__(self, api_key: str):
        if not OPENAI_AVAILABLE:
            raise ImportError("Please install openai: pip install openai")
        if not api_key:
            raise ValueError("Missing OpenAI API Key. Set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: Union[str, List[Dict]], system_instruction: str, config: Dict[str, Any]) -> str:
        if "model_name" not in config:
            raise KeyError("Missing 'model_name' in config")

        messages = self._build_messages(prompt, system_instruction, config["model_name"])

        kwargs = {
            "model": config["model_name"],
            "messages": messages,
            "temperature": 1.0,
        }

        response = self.client.chat.completions.create(**kwargs)

        if response.choices:
            return response.choices[0].message.content
        return "ERROR_GPT_EMPTY"

    def _build_messages(self, prompt, system_instruction, model_name):
        """Build OpenAI format messages."""
        user_content = []
        if isinstance(prompt, list):
            for item in prompt:
                if item["type"] == "text":
                    user_content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image":
                    path = item["path"]
                    b64_data = encode_image(path)
                    mime = get_mime_type(path)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64_data}",
                            "detail": "high"
                        }
                    })
        else:
            user_content = prompt

        if model_name.startswith("o1") or model_name.startswith("o3"):
            return [{"role": "user", "content": f"{system_instruction}\n\n{str(prompt)}"}]

        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]


class GrokModel(BaseModel):
    """xAI Grok model wrapper (OpenAI compatible)."""

    def __init__(self, api_key: str):
        if not OPENAI_AVAILABLE:
            raise ImportError("Please install openai: pip install openai")
        if not api_key:
            raise ValueError("Missing xAI API Key. Set XAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

    def generate(self, prompt: Union[str, List[Dict]], system_instruction: str, config: Dict[str, Any]) -> str:
        if "model_name" not in config:
            raise KeyError("Missing 'model_name' in config")

        messages = self._build_messages(prompt, system_instruction)

        response = self.client.chat.completions.create(
            model=config["model_name"],
            messages=messages,
            temperature=config.get("temperature", 0.0),
            top_p=config.get("top_p", 1.0),
        )

        if response.choices:
            return response.choices[0].message.content
        return "ERROR_GROK_EMPTY"

    def _build_messages(self, prompt, system_instruction):
        """Build xAI format messages."""
        user_content = []
        if isinstance(prompt, list):
            for item in prompt:
                if item["type"] == "text":
                    user_content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image":
                    path = item["path"]
                    b64_data = encode_image(path)
                    mime = get_mime_type(path)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64_data}",
                            "detail": "high"
                        }
                    })
        else:
            user_content = prompt

        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]


def get_model_wrapper(model_name: str) -> BaseModel:
    """Factory function to get model wrapper."""
    if not model_name:
        raise ValueError("Model name is empty!")

    model_name_lower = model_name.lower()

    if "gemini" in model_name_lower:
        api_key = os.environ.get("GOOGLE_API_KEY")
        return GeminiModel(api_key)

    elif "claude" in model_name_lower:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        return ClaudeModel(api_key)

    elif "gpt" in model_name_lower or "o1-" in model_name_lower:
        api_key = os.environ.get("OPENAI_API_KEY")
        return OpenAIModel(api_key)

    elif "grok" in model_name_lower:
        api_key = os.environ.get("XAI_API_KEY")
        return GrokModel(api_key)

    else:
        raise ValueError(f"Unsupported model family: {model_name}")
