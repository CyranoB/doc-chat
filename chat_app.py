import pickle
from chat_vector import get_chain

if __name__ == "__main__":
    print("Initializing")
    with open("vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    qa_chain = get_chain(vectorstore)
    result = qa_chain({"question": "what can you help with?", "chat_history": []})
    print(result["answer"])

    chat_history = []
    while True:
        print("\n")
        print(">> :")
        question = input()
        print("\n")
        result = qa_chain({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))

