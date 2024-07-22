import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
load_dotenv()

llm = AzureChatOpenAI(
    openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    temperature = 0.8,
    max_tokens = 60
)

messages = [
    (
        "system",
        "You are a helpful assistant that helps the user write functions and explains its usage",
    ),
    ("human", "write a function that computes the product of all integers in a list "),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)