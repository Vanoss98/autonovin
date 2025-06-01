from django.conf import settings
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI


class OpenaiService:
    def __init__(self, model="gpt-4.1-nano", temperature=0, api_key=None):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key

    def invoke_llm(self, prompt):
        # Assuming you have the necessary setup to call the LLM API
        llm = ChatOpenAI(model=self.model, temperature=self.temperature, api_key=self.api_key)
        sys_msg = SystemMessage(content=prompt)
        response = llm.invoke([sys_msg])
        return response.content

