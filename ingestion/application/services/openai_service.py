from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class OpenaiService:
    def __init__(self, model="gpt-4.1", temperature=0, api_key=None):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key

    def invoke_llm(self, prompt):
        # Assuming you have the necessary setup to call the LLM API
        llm = ChatOpenAI(model=self.model, temperature=self.temperature, api_key=self.api_key)
        sys_msg = SystemMessage(content=prompt)
        response = llm.invoke([sys_msg])
        return response.content

    def invoke_with_json_output(self, template, output_structure, doc_text):
        parser = JsonOutputParser(pydantic_object=output_structure)
        prompt = PromptTemplate(
            template=template,
            input_variables=["docs"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        llm = ChatOpenAI(temperature=self.temperature, model=self.model, api_key=self.api_key)
        chain = prompt | llm | parser
        result = chain.invoke({"docs": doc_text})
        return result
