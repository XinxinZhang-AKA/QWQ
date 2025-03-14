# qwen_models.py
import dashscope
import numpy as np
import logging


class QwenTextGenerationModel:
    def __init__(self, api_key: str, model_name: str = "qwen-turbo"):
        dashscope.api_key = api_key
        self.model_name = model_name
        self.history = []

    def generate(self, system: str, user: str, **kwargs):
        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user}
        ]

        try:
            response = dashscope.Generation.call(
                model=self.model_name,
                messages=messages,
                result_format='message',  # 明确要求消息格式
                **kwargs
            )

            # 检查响应有效性
            if response and response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                print(f"API Error: {response.code} - {response.message}")
                return ""

        except Exception as e:
            print(f"Generation failed: {str(e)}")
            return ""

    def continue_generate(self, system: str, user1: str, assistant1: str, user2: str, **kwargs):
        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user1},
            {'role': 'assistant', 'content': assistant1},
            {'role': 'user', 'content': user2}
        ]
        response = dashscope.Generation.call(
            self.model_name,
            messages=messages,
            result_format='message',
            **kwargs
        )
        return response.output.choices[0].message.content


class QwenEmbeddingModel:
    def __init__(self, api_key: str, model_name: str = "text-embedding-v1"):
        dashscope.api_key = api_key
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

    def encode(self, text):
        try:
            if isinstance(text, str):
                text = [text]

            embeddings = []
            for t in text:
                resp = dashscope.TextEmbedding.call(
                    model=self.model_name,
                    input=t
                )
                embeddings.append(np.array(resp.output.embeddings[0].embedding))

            return embeddings[0] if len(embeddings) == 1 else embeddings
        except Exception as e:
            self.logger.error(f"Embedding error: {e}")
            return None