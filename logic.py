from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

# Model 
llm = ChatMistralAI(model = "mistral-medium-3-5")


# prompt template

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful conversational assistant 
    with access to a PDF document.
    
    Follow these rules:
    - If the user asks a question related to the PDF content, 
      answer ONLY from the provided context.
    - If the answer is not in the context, say 
      'I could not find that information in the document.'
    - If the user is making small talk, greetings, or asking 
      general questions unrelated to the PDF, respond naturally 
      and conversationally. Do NOT pretend these answers come 
      from the PDF.
    - Never mix your own knowledge with PDF content when 
      answering document-specific questions."""),
    
    ("human", """
    Context: {context}
    Question: {input}""")
])

def create_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": 0.7
        }
    )
    
    # New LCEL style — pipe operator connects everything
    chain = (
        {
            "context": retriever,
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def ask_question(chain, question):
    return chain.invoke(question)


if __name__ == "__main__":
    from db import create_vectorstore
    
    # Step 1 — Create vectorstore from your test PDF
    vectorstore = create_vectorstore("case_study_2327114.pdf")
    print("✅ Vectorstore created")
    
    # Step 2 — Create chain
    chain = create_chain(vectorstore)
    print("✅ Chain created")
    
    # Step 3 — Ask a question
    answer = ask_question(chain, "Is there a specific target mentioned in the case study?")
    print("✅ Answer:", answer)