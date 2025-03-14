# 临时测试脚本 test_api.py
import dashscope
dashscope.api_key = "sk-275206b2a32b4b55bc079afa1d69b635"

response = dashscope.Generation.call(
    model="qwen-turbo",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response)