from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from chat_vector import get_chain
import sys

if __name__ == "__main__":
    print("Initializing")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2250, chunk_overlap=400)
    to_load = sys.argv[1]
    print(f"Accessing {to_load}")
    loader = UnstructuredURLLoader(urls=[to_load])
    docs = loader.load()
    print(f"Processing: {to_load}")
    documents = text_splitter.create_documents(
        [doc.page_content for doc in docs],
        [doc.metadata for doc in docs]
    )

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    qa_chain = get_chain(vectorstore)

    print("Ready\n")
    result = qa_chain({"question": "what is this about?", "chat_history": []})
    print(result["answer"])

    chat_history = []
    while True:
        print("\n")
        print(">> :")
        question = input()
        print("\n")
        result = qa_chain({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))

