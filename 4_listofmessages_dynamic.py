from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define the chat prompt template
chat_template = ChatPromptTemplate([
    # ("system", "You are a helpful customer support agent."),
    # ("human", "{input}"),
    # The human message contains only the current query. To provide context
    # from previous interactions (e.g., asking about a refund status), we need
    # to include the conversation history in the prompt.
    # MessagesPlaceholder allows us to insert the chat history before the
    # current human message, ensuring the model has the previous context.

    ("system", "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="chat_hist"),
    ("human", "{input}"),
])

# Load the previous conversation history from a text file
c_h = []
with open('4_chat_history_example.txt', 'r') as f:
    for line in f:
        c_h.append(line.strip())
# Pass the loaded conversation history to the MessagesPlaceholder
# using the corresponding variable name during invocation.

# Create the prompt by supplying the current input and chat history
prompt = chat_template.invoke({ 'input': "what is the status of my refund", 'chat_hist': c_h })
print(prompt)
