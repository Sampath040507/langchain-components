from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# llm = HuggingFacePipeline.from_model_id(
#     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
#     task="text-generation",
#     pipeline_kwargs=dict(
#         temperature=0.1,
#         max_new_tokens=50
#     )
# )
# model = ChatHuggingFace(llm=llm)

model = ChatOpenAI()

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat_history=[SystemMessage(content="You are a helpful assistant.")]

while True:
    user_input = input("you:")
    if user_input== "exit":
        print("Exiting the chat.")
        break
    chat_history.append(HumanMessage(content=user_input))
    result = model.invoke(chat_history)
    print(f"AI:{result.content}")
    chat_history.append(AIMessage(content=result.content))

print("Chat History:")
print(chat_history)
