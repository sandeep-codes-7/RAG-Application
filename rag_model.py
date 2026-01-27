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