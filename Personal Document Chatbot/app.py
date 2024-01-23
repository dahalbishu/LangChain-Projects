import chainlit  as cl
import os
from langchain import PromptTemplate


from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

loader = PyPDFDirectoryLoader("pdfs")
data = loader.load_and_split()

welcome_message = """Welcome to the Chainlit PDF QA demo! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""



from langchain_google_genai import GoogleGenerativeAIEmbeddings

import google.generativeai as genai
import os
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ['GOOGLE_API_KEY'] = "GOOGLE_GENERATIVEAI_API_KEY"
genai.configure(api_key = os.environ['GOOGLE_API_KEY'])


embeddings = GoogleGenerativeAIEmbeddings( model = "models/embedding-001")

global text


@cl.on_chat_start
async def on_chat_start():
    print("A new chat session has started!")

    # Sending an image with the local file path
    await cl.Message(content="You can now chat with your pdfs srijuyu.").send()


    files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    import PyPDF2


    with open(file.path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    



    

    # Let the user know that the system is ready


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
 

    texts = text_splitter.split_text(text)
    msg = cl.Message(content=f"file is proccessed successfully! Now you can go on") 
    await msg.send()
    cl.user_session.set("chain", texts)
    


@cl.on_message
async def main(message : str):

    text = cl.user_session.get("chain") 
    vector_index = Chroma.from_texts(text, embeddings).as_retriever()
   
    question = message.content


   



    docs = vector_index.get_relevant_documents(question)

    #print(docs)
    prompt_template = """   
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer 
    is not in provided context just say, "answer is not available in the context", dont't provide the wrong answer\n\n
    Context:\n{context}?\n
    Question: \n {question}\n

    Answer:
    """

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])

    model = ChatGoogleGenerativeAI(model = "gemini-pro", temperature = 0.5)

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    response = chain(
    {"input_documents":docs, "question":question}
        , return_only_outputs = True
    )
    #print(question)
    #print(response)

    await cl.Message(content = response['output_text']).send()