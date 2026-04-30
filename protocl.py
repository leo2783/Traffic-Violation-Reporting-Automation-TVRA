from openai import OpenAI

# 1. 初始化客戶端 (指向 NVIDIA API)
client = OpenAI(
  base_url="https://integrate.api.nvidia.com/v1",
  api_key="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"
)

# 2. 呼叫模型
response = client.chat.completions.create(
  model="meta/llama-3.1-8b-instruct",
  messages=[{"role":"user", "content":"你好，請測試連線。"}],
  timeout=10.0
)

print(response.choices[0].message.content)