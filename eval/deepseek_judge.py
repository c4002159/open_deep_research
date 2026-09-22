"""把DeepSeek包装成DeepEval认识的Judge模型。DeepSeek的API兼容OpenAI协议，
直接换 base_url 就能用 LangChain 的 ChatOpenAI 连过去，不用装额外的包。"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from deepeval.models.base_model import DeepEvalBaseLLM


class DeepSeekJudge(DeepEvalBaseLLM):
    def __init__(self):
        self._llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
            temperature=0.0,  # 评测要的是稳定可复现，不要随机性
        )

    def load_model(self):
        return self._llm

    def generate(self, prompt: str, *args, **kwargs) -> str:
        return str(self._llm.invoke([HumanMessage(content=prompt)]).content)

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        response = await self._llm.ainvoke([HumanMessage(content=prompt)])
        return str(response.content)

    def get_model_name(self) -> str:
        return "deepseek-chat (custom judge)"