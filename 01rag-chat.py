import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import streamlit as st



load_dotenv()

# Get environment variables
openai_endpoint = os.getenv("OPENAI_ENDPOINT")
openai_api_key = os.getenv("OPENAI_API_KEY")
chat_model = os.getenv("CHAT_MODEL")
embedding_model = os.getenv("EMBEDDING_MODEL")
search_endpoint=os.getenv("SEARCH_ENDPOINT")
search_api_key=os.getenv("SEARCH_API_KEY")
index_name = os.getenv("INDEX_NAME")
api_version = "2024-12-01-preview"


# Initialize Azure OpenAI client
chat_client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=openai_endpoint,
    api_key=openai_api_key
)

st.title("Margie's Travel Assistant")
st.write("Ask your travel-related question below:")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are a travel assistant that provides information on travel service available from Margie's Travel."
        },
    ]

# display chat history
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

def get_openai_response(messages):
   
    # Additional parameters to apply RAG pattern using the AI Search index
    rag_params = {
        "data_sources": [
            {
                # he following params are used to search the index
                "type": "azure_search",
                "parameters": {
                    "endpoint": search_endpoint,
                    "index_name": index_name,
                    "authentication": {
                        "type": "api_key",
                        "key": search_api_key,
                    },
                    # The following params are used to vectorize the query
                    "query_type": "vector",
                    "embedding_dependency": {
                        "type": "deployment_name",
                        "deployment_name": embedding_model,
                    },
                }
            }
        ],
    }

        
    # Submit the chat request with RAG parameters
    response = chat_client.chat.completions.create(
        model=chat_model,
        messages=messages,
        extra_body=rag_params
    )

    completion = response.choices[0].message.content
    return completion

# Handle user input
if input_text := st.chat_input("Enter your question"):
    st.session_state.messages.append({"role": "user", "content": input_text})
    st.chat_message("user").write(input_text)

    with st.spinner("Generating response..."):
        response = get_openai_response(st.session_state.messages)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)