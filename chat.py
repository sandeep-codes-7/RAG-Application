import streamlit as st
import time
# from rag_model import chain
from langchain_ollama import ChatOllama,OllamaEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader 
from langchain_classic.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# from langchain_classic.output_parsers import StructuredOutputParser
# import json
# import warnings

# warnings.filterwarnings('ignore',message="*.libmagic.*")

# from chat import user_prompt

model = ChatOllama(
    model="llama3.2:3b",
    validate_model_on_init=True,
)



# print("perfect")

loader = TextLoader("context.txt",encoding="utf8")
# loader = DirectoryLoader(
#     path="C:\\Users\\YOGA\\Desktop\\fastAPI-projects\\testing",
#     glob="**/*.py"
# ) 


docs = loader.load()
# print(f"loaded {len(docs)} files...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
split = splitter.split_documents(docs)
# print(f"created {len(split)} chunks...")

vectordb = Chroma.from_documents(
    embedding=OllamaEmbeddings(model="nomic-embed-text:v1.5"),
    documents=split,
    persist_directory="chroma-db",
    collection_name="context-data"
)

ret = vectordb.as_retriever(search_kwargs={"k":2})

template = """

you are a helpful ai assistant at hospital desk.
use the following pieces of context to answer the question at the end.
{context}
Question: {question}
Answer in a concise manner and avoid telling the user that is out of context and speak politely. and also addres what services that you can provide.



"""

# template = """
# You are an expert code explainer assistant. Analyze the provided code context to answer questions about the codebase.

# Code Context:
# {context}

# Question: {question}

# If the question is outside the provided code context, tell the user this information is not available in the codebase.
# """


# parser = StructuredOutputParser.from_response_schemas(
#     [
#         {"name": "question","description":"prompt from the user"},
#         {"name": "answer", "description": "Answer to the question"},
#         {"name": "confidence", "description": "Confidence score between 0 and 1"},
#     ]
# )

# format_instructions = parser.get_format_instructions()




prompt = PromptTemplate.from_template(template)
# prompt = PromptTemplate(
#     input_variables=["question"],
#     partial_variables={"format_instructions": format_instructions},
#     template="""
#         Answer the question below.
#         {format_instructions}
#         Question: {question}

#     """
# )


# | StrOutputParser()
chain = (
    {"context":ret,"question":RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# response = chain.invoke("hi")

# print(response)


# def get_res():
#     print(response)
# # res = model.invoke(messages)

# # print(res.content)
# get_res()












# ------------------ Page Config ------------------
st.set_page_config(
    page_title="AI Chat",
    page_icon="🤖",
    layout="centered"
)

# ------------------ Custom CSS ------------------
st.markdown("""
<style>
.chat-container {
    max-width: 750px;
    margin: auto;
}
.stChatMessage {
    padding: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

# ------------------ Title ------------------
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
st.title("🤖 Bot@HospitalDesk")

# ------------------ Session State ------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey 👋 Ask me anything."}
    ]

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("⚙️ Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 64, 2048, 512)

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared. Start fresh 🚀"}
        ]
        st.rerun()

# ------------------ Chat History ------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------ Chat Input ------------------
user_prompt = st.chat_input("Type your message...")





# ------------------ Response Logic ------------------
def generate_response(user_input):
    # Simulate thinking / streaming
    with st.spinner("generating response..."):
        res = chain.invoke(user_prompt)
        time.sleep(0.3)
    return f"replying to '{user_input}':\n\n{res}"

if user_prompt:
    # User message
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Assistant message (streaming style)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        response = ""

        full_response = generate_response(user_prompt)
        for char in full_response:
            response += char
            placeholder.markdown(response)
            time.sleep(0.01)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )

st.markdown("</div>", unsafe_allow_html=True)
