"""
OpenAI-compatible wrapper for AWS Bedrock (Anthropic Claude models).
Lets all existing code that uses client.chat.completions.create(...) work unchanged.
"""
import os


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, text: str):
        self.message = _Message(text)


class _Response:
    def __init__(self, text: str):
        self.choices = [_Choice(text)]


class _Completions:
    def __init__(self, bedrock_client, model: str):
        self._client = bedrock_client
        self._model = model

    def create(self, model=None, messages=None, temperature=0.0, max_tokens=512, **kwargs):
        model = model or self._model

        # Split system message from conversation messages
        system = ""
        anthropic_messages = []
        for m in messages or []:
            if m["role"] == "system":
                system = m["content"]
            else:
                anthropic_messages.append({"role": m["role"], "content": m["content"]})

        create_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
        }
        if system:
            create_kwargs["system"] = system
        # Bedrock Claude accepts temperature 0–1
        if temperature is not None:
            create_kwargs["temperature"] = float(min(max(temperature, 0.0), 1.0))

        response = self._client.messages.create(**create_kwargs)
        text = response.content[0].text if response.content else ""
        return _Response(text)


class _Chat:
    def __init__(self, bedrock_client, model: str):
        self.completions = _Completions(bedrock_client, model)


class BedrockClient:
    """
    Drop-in replacement for openai.OpenAI that routes calls to AWS Bedrock.
    Usage:  client.chat.completions.create(model=..., messages=..., ...)
    """

    def __init__(self, model: str):
        from anthropic import AnthropicBedrock
        self._bedrock = AnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID", ""),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
        )
        self.chat = _Chat(self._bedrock, model)
