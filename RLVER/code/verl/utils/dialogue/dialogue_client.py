from openai import OpenAI
from typing import List, Dict, Any

class DialogueClient:
    def __init__(self, api_key: str = "0", base_url: str = "http://29.81.228.5:8081/v1/"):
        """
        初始化对话客户端
        
        Args:
            api_key: API密钥，默认为"0"
            base_url: API基础URL
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
    def generate_reply(self, messages: List[Dict[str, str]], 
                      model: str = "DeepSeek-R1",
                      temperature: float = 0.6,
                      stream: bool = False,
                      max_retries: int = 5,
                      retry_delay: float = 3.0) -> str:
        """
        生成对话回复，失败时会自动重试
        
        Args:
            messages: 对话历史消息列表
            model: 使用的模型名称
            temperature: 温度参数，控制随机性
            stream: 是否使用流式输出
            max_retries: 最大重试次数，默认5次
            retry_delay: 重试间隔时间（秒），默认3秒
            
        Returns:
            str: 生成的回复内容
        """
        attempt = 0
        while True:
            try:
                result = self.client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    stream=stream
                )
                reply = result.choices[0].message.content
                if reply is not None:
                    return reply
                    
            except Exception as e:
                attempt += 1
                if attempt >= max_retries:
                    print(f"达到最大重试次数 {max_retries}，生成回复失败: {str(e)}")
                    return ""
                print(f"第 {attempt} 次重试失败: {str(e)}")
                time.sleep(retry_delay)