from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    openai_api_base="https://xh.v1api.cc/v1",
    openai_api_key="sk-NCe9VNEDxfEIZwG7DlJdLaGhvOW1Oyhuh7dWwMP7bbPUB5Dm"
)

res = llm.invoke("hello")
print(res.content)