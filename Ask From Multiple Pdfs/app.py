import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
# from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

#IMPORT PINECONE
import os
from langchain.vectorstores import Pinecone
import pinecone

INDEX_NAME = "electrical-vehicles-details-index"


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):

    # initialize picone vector database
    pinecone.init(api_key=os.getenv("PICONE_API_KEY"), 
                  environment=os.getenv("PINECONE_ENVIRONMENT"))
    
    print("available indexes:",pinecone.list_indexes())

    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    Pinecone.from_texts(texts=text_chunks, embedding=embeddings, index_name=INDEX_NAME)
    print("Creating Embeddings........")

def run_llm(query,chat_history):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    docsearch = Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    llm = HuggingFaceHub(repo_id="google/flan-ul2", model_kwargs={"temperature":0.5, "max_length":512},verbose=True)
    
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=docsearch.as_retriever(), return_source_documents=True
    )


    return conversational_chain({"question": query, "chat_history": chat_history})


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    
    # initialize picone vector database
    pinecone.init(api_key=os.getenv("PICONE_API_KEY"), 
                  environment=os.getenv("PINECONE_ENVIRONMENT"))
    
    print("available indexes:",pinecone.list_indexes())


    st.set_page_config(page_title="Ask About Electrical Vehicles",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state):
        st.session_state["chat_answers_history"] = []
        st.session_state["user_prompt_history"] = []
        st.session_state["chat_history"] = []

    # Create a Vectorstore
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("In Progress...."):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                get_vectorstore(text_chunks)
    
    st.header("Ask About Electrical Vehicles")
    prompt = st.text_input("Enter You Question", placeholder="Enter your message here...") or st.button("Submit")
    if prompt:
        # handle_userinput(prompt)
        with st.spinner("Generating response..."):
            generated_response = run_llm(
                query=prompt, chat_history=st.session_state["chat_history"]
            )
            st.session_state.chat_history.append((prompt, generated_response["answer"]))
            st.session_state.user_prompt_history.append(prompt)
            st.session_state.chat_answers_history.append(generated_response["answer"])

        if st.session_state["chat_answers_history"]:
            for generated_response, user_query in zip(
                st.session_state["chat_answers_history"],
                st.session_state["user_prompt_history"],
            ):
                message(
                    user_query,
                    is_user=True,
                )
                message(generated_response)

if __name__ == '__main__':
    main()
