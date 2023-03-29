from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.conversational_retrieval.prompts import \
    CONDENSE_QUESTION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.chains import ConversationalRetrievalChain

SYSTEM_TEMPLATE="""You are an AI assistant for answering questions documents.
You are given the following extracted parts of long documents and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure actually." Don't try to make up an answer.
----------------
{context}"""
messages = [
    SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
    HumanMessagePromptTemplate.from_template("{question}")
]
QA_PROMPT = ChatPromptTemplate.from_messages(messages)


def get_chain(vectorstore):
    llm = OpenAI(temperature=0)
    streaming_llm = ChatOpenAI(streaming=True, 
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), 
    verbose=True, temperature=0)

    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_chain(streaming_llm, chain_type="stuff", prompt=QA_PROMPT)

    qa = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(), 
        combine_docs_chain=doc_chain, 
        question_generator=question_generator)
    return qa
