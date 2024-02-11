import requests
from typing import Any, List, Mapping, Optional
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

class LlamaLLM(LLM):
    llm_url = 'http://192.168.15.184:8000/generate/'  # Ensure correct URL
    headers = {'Accept': 'application/json'}

    @property
    def _llm_type(self) -> str:
        return "Llama2 70B"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        r = requests.post(self.llm_url,params={'prompt': prompt}, headers=self.headers)  # Corrected line
        r.raise_for_status()

        return r.json()['generated_text']  # Adjusted to match the expected response key

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"llmUrl": self.llm_url}

# Example usage
llm = LlamaLLM()
# Notice that "chat_history" is present in the prompt template
template = """You are a nice chatbot having a conversation with a human.

Previous conversation:
{chat_history}

New human question: {question}
Response:"""
prompt = PromptTemplate.from_template(template)
# Notice that we need to align the `memory_key`
memory = ConversationBufferMemory(memory_key="chat_history")
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)
result=conversation({"question": "hi"})
print(result)

