import tempfile

import streamlit as st
import pinecone
import os

from langchain_community.vectorstores import Pinecone, Chroma
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain.text_splitter import RecursiveCharacterTextSplitter

st.title("Hybrid-RAG APP ")



upload_option = st.selectbox("Do you want to upload a single PDF or a folder of PDFs:",
                             ("Single PDF", "Folder of PDFs"))

if upload_option == "Single PDF":
    upload_file = st.file_uploader("Upload the PDF for context", type="pdf")

    if upload_file is not None:
        import io
        import os
        from langchain_community.document_loaders import PyPDFLoader

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(upload_file.read())
            path = temp_file.name

            loader = PyPDFLoader(path)
            documents = loader.load()

            st.write(f'The number of pages in the document are: {len(documents)}')
            os.remove(path)
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=150, length_function=len,
                                                  is_separator_regex=False)
        chunks = [splitter.split_text(doc.page_content) for doc in documents]
        flat_chunks = [chunk for sublist in chunks for chunk in sublist]

        splitter1 = RecursiveCharacterTextSplitter(chunk_size=800,
                                                   chunk_overlap=100)
        chunk = splitter1.split_documents(documents)

        import pandas as pd

        df = pd.DataFrame(flat_chunks, columns=['vectors'])

        from langchain.embeddings import HuggingFaceBgeEmbeddings

        model_name = "BAAI/bge-small-en-v1.5"
        model_kwargs = {'normalize_embeddings': True}

        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            encode_kwargs=model_kwargs,
            model_kwargs={'device': 'auto'},
        )
        import os
        from pinecone import Pinecone, ServerlessSpec

        # Initialize Pinecone client
        import os
        api_key = os.environ.get("PINECONE_API_KEY")
        index_name = "apurv"
        pc = Pinecone(api_key=api_key)

        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=384,
                metric='dotproduct',
                spec=ServerlessSpec(cloud='aws', region='us-east-1'),
            )

        index = pc.Index(index_name)
        upsert_vectors = [
            (str(i), embeddings.embed_documents([chunk])[0], {"metadata_key": "metadata_value"})
            for i, chunk in enumerate(df['vectors'])
        ]
        index.upsert(upsert_vectors)
        vectorstore = Chroma.from_documents(chunk, embeddings)
        from langchain.retrievers import EnsembleRetriever
        import pinecone
        import os
        from langchain.vectorstores import Pinecone as LangChainPinecone
        from pinecone import Pinecone, ServerlessSpec

        # Initialize Pinecone client
        import os
        api_key = os.environ.get("PINECONE_API_KEY")
        index_name = "apurv"
        pc = Pinecone(api_key=api_key)

        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=384,  
                metric='dotproduct',
                spec=ServerlessSpec(cloud='aws', region='us-east-1'),
            )

        
        index = pc.Index(index_name)

        retriever1 = vectorstore.as_retriever(search_kwargs={"k": 3})
        retriever2 = LangChainPinecone(

            index=index,  
            embedding=embeddings.embed_query,
            text_key="vectors"  
        ).as_retriever(search_kwargs={"k": 3})

        ensambel_reteriever = EnsembleRetriever(
            retrievers=[retriever1, retriever2],
            weight=[0.6, 0.4]

        )
        
        index = pc.Index(index_name)

        retriever1 = vectorstore.as_retriever(search_kwargs={"k": 3})
        retriever2 = LangChainPinecone(

            index=index,  
            embedding=embeddings.embed_query,
            text_key="vectors"  
        ).as_retriever(search_kwargs={"k": 3})

        ensambel_reteriever = EnsembleRetriever(
            retrievers=[retriever1, retriever2],
            weight=[0.6, 0.4]
        )

        import google.generativeai as genai
        import os

        api_key = os.environ.get("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)

        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.7)
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain

        template = """
        You are a knowledgeable assistant skilled in providing thorough and accurate responses. Use the provided context to answer the user's question comprehensively. Ensure that your answer covers all relevant aspects and is presented in a clear and organized manner and also try to give a full complete response .
        Context:
        {context}

        Question:
        {query}

        Answer (detailed, with examples if relevant):
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", template),
                ("human", '{input}')
            ]
        )

        output_parser = StrOutputParser()
        from langchain.schema.runnable import RunnableMap

        chain = RunnableMap({
            "context": lambda x: ensambel_reteriever.get_relevant_documents(x["query"]),
            "query": lambda x: x["query"],  
            "input": lambda x: x["query"]  
        }) | prompt | llm | output_parser

        import logging

        logging.getLogger("langchain_community.vectorstores.pinecone").setLevel(logging.ERROR)

elif upload_option == "Folder of PDFs":
    from langchain.document_loaders import PyPDFDirectoryLoader

    upload_folder = st.file_uploader("Upload the Folder into zip formate", type="zip")
    if upload_folder:
        import os
        import zipfile

        temp_dir = "/tmp/pdf_folder"
        os.makedirs(temp_dir, exist_ok=True)

        with zipfile.ZipFile(upload_folder, "r") as zip_f:
            zip_f.extractall(temp_dir)

        loader = PyPDFDirectoryLoader(temp_dir)
        documents = loader.load()

        st.write(f'The number of pages in the document are :{len(documents)}')

        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=150, length_function=len,
                                                  is_separator_regex=False)
        chunks = [splitter.split_text(doc.page_content) for doc in documents]
        flat_chunks = [chunk for sublist in chunks for chunk in sublist]

        splitter1 = RecursiveCharacterTextSplitter(chunk_size=800,
                                                   chunk_overlap=100)
        chunk = splitter1.split_documents(documents)

        import pandas as pd

        df = pd.DataFrame(flat_chunks, columns=['vectors'])

        from langchain.embeddings import HuggingFaceBgeEmbeddings

        model_name = "BAAI/bge-small-en-v1.5"
        model_kwargs = {'normalize_embeddings': True}

        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            encode_kwargs=model_kwargs,
            model_kwargs={'device': 'cpu'},
        )
        import os
        from pinecone import Pinecone, ServerlessSpec
        api_key = os.environ.get("PINECONE_API_KEY")
        index_name = "apurv"
        pc = Pinecone(api_key=api_key)

        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=384,
                metric='dotproduct',
                spec=ServerlessSpec(cloud='aws', region='us-east-1'),
            )

        index = pc.Index(index_name)
        upsert_vectors = [
            (str(i), embeddings.embed_documents([chunk])[0], {"metadata_key": "metadata_value"})
            for i, chunk in enumerate(df['vectors'])
        ]
        index.upsert(upsert_vectors)

        vectorstore = Chroma.from_documents(chunk, embeddings)

        from langchain.retrievers import EnsembleRetriever
        import pinecone
        import os
        from langchain.vectorstores import Pinecone as LangChainPinecone
        from pinecone import Pinecone, ServerlessSpec
        index_name = "apurv"
        pc = Pinecone(api_key=api_key)

        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=384,  
                metric='dotproduct',
                spec=ServerlessSpec(cloud='aws', region='us-east-1'),
            )

       
        index = pc.Index(index_name)

        retriever1 = vectorstore.as_retriever(search_kwargs={"k": 3})
        retriever2 = LangChainPinecone(

            index=index,  
            embedding=embeddings.embed_query,
            text_key="vectors" 
        ).as_retriever(search_kwargs={"k": 3})

        ensambel_reteriever = EnsembleRetriever(
            retrievers=[retriever1, retriever2],
            weight=[0.6, 0.4]
        )

        import google.generativeai as genai
        import os

        os.environ.get("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.7)
        from langchain.schema.runnable import RunnableMap

        template = """
        You are a knowledgeable assistant skilled in providing thorough and accurate responses. Use the provided context to answer the user's question comprehensively. Ensure that your answer covers all relevant aspects and is presented in a clear and organized manner and also try to give a full complete response .
        Context:
        {context}

        Question:
        {query}

        Answer (detailed, with examples if relevant):
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", template),
                ("human", '{input}')
            ]
        )

        output_parser = StrOutputParser()

        chain = RunnableMap({
            "context": lambda x: ensambel_reteriever.get_relevant_documents(x["query"]),
            "query": lambda x: x["query"],  
            "input": lambda x: x["query"]  
        }) | prompt | llm | output_parser

        import logging

        logging.getLogger("langchain_community.vectorstores.pinecone").setLevel(logging.ERROR)

user_input = st.text_input("Enter Your Query Here:")
if st.button("Answer:"):
    if user_input:
        with st.spinner("Processing..."):
            try:
                result = chain.invoke({"query": user_input})

                st.write("### Answer ###")
                st.write(result)
            except Exception as ex:
                st.error(f'Error: {str(ex)}')

    else:
        st.warning("Enter the Query First...")



