from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain_community.document_loaders import HuggingFaceDatasetLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cosine_similarity import cosine_similarity

# Define the metadata extraction function.
from langchain_community.document_loaders import JSONLoader

import requests 

def query_ollama(question):
    
    # OLLAMA_URL = "http://172.30.4.31:11434/api/generate"
    OLLAMA_URL = "http://localhost:11434/api/generate"
    payload = {
        "model":"mistral:instruct",
        "prompt": question,"stream":False 
    }
    response = requests.post(OLLAMA_URL, json=payload)
    print (response)
    return response.json()

def metadata_func(record: dict, metadata: dict) -> dict:
    
    metadata["Combined Comments"] = record.get("Combined Comments")
    metadata["Summary"] = record.get("Summary")
    # metadata["Priority"] = record.get("Priority")
    # metadata["Description"] = record.get("Description")
    # metadata["Module"] = record.get("Module")

    return metadata

def RAG_summary(summary):
    
    summary = "MGU time out"
    loader = JSONLoader(
        file_path='./RAG_total/test1.json',
        jq_schema='.message[]',
        content_key="Summary",
        text_content=False,
        metadata_func=metadata_func
    )

    data = loader.load()

    modelPath = "sentence-transformers/all-MiniLM-l6-v2"
    model_kwargs = {'device':'cpu'}
    encode_kwargs = {'normalize_embeddings': False}


    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,    # pre-trained model's path    
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )

    db = FAISS.from_documents(data, embeddings)

    retriever = db.as_retriever(search_kwargs={"k": 4})

    # result = ui.process_data(123, "asd", "asd", "asd")
    # result = "MGU Timed out"
    print(summary)

    docs = retriever.get_relevant_documents(summary)
    # docs = retriever.get_relevant_documents("Unexpected HARD_Error_Reset is triggered.")
    # docs = retriever.get_relevant_documents("Startup time of CID Home Menu is too slow")

    print(docs[0].metadata)

    doc1 = docs[0].metadata['Combined Comments']
    doc2 = docs[1].metadata['Combined Comments']
    doc3 = docs[2].metadata['Combined Comments']
    doc4 = docs[3].metadata['Combined Comments']

    vectorizer = TfidfVectorizer()
    vectors1 = vectorizer.fit_transform([doc1, doc2])
    vectors2 = vectorizer.fit_transform([doc1, doc3])
    vectors3 = vectorizer.fit_transform([doc1, doc4])

    similarity_score1 = cosine_similarity(vectors1, vectors1)
    similarity_score2 = cosine_similarity(vectors2, vectors2)
    similarity_score3 = cosine_similarity(vectors3, vectors3)

    print("Cosine Similarity Score:", similarity_score1[0][1])
    print("Cosine Similarity Score:", similarity_score2[0][1])
    print("Cosine Similarity Score:", similarity_score3[0][1])

    filtered_documents = []
    filtered_documents.append(docs[0])


    if (similarity_score1[0][1] >= 0.8):
        print("1")
        filtered_documents.append(docs[1])
        if (similarity_score2[0][1] >= 0.8):
            print("2")
            filtered_documents.append(docs[2])
            if (similarity_score3[0][1] >= 0.8):
                print("3")
                filtered_documents.append(docs[3])

    list_filtered_docs = []
    for i in range (len(filtered_documents)):
        # print(i)
        list_filtered_docs.append(filtered_documents[i].metadata['Combined Comments'])

    from transformers import pipeline
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    READER_MODEL_NAME = "Qwen/Qwen1.5-0.5B"


    model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

    READER_LLM = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        # temperature=0.2,
        # repetition_penalty=1.1,
        # return_full_text=False,
        # max_new_tokens=500,
    )

    prompt_in_chat_format = [
        {
            "role": "system",      
            "content": """The text below contains Comments from analysis of a bug, 
            It contains comments of comunication for the defect bugs resolution, 
            give more importance to the solution and summarise the steps involved and answer the query based on the same context.
            Most importantly deduce the answer only from the context provided which is actually the comments and dont give your own solutions for the resolving the bug
            also answer hot to solve the issue or the bug based only on the comments""",
        },
        {
            "role": "user",
            "content": """Context:
    {context}
    ---
    Now here is the question you need to answer.
    
    Question: {question}""",
        },
    ]
    RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )
    # print(RAG_PROMPT_TEMPLATE)

    retrieved_docs_text = [doc for doc in list_filtered_docs]  # We only need the text of the documents
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])

    final_prompt = RAG_PROMPT_TEMPLATE.format(question="How was this issue resolved", context=context)

    print(final_prompt)

    # answer = READER_LLM(final_prompt)[0]["generated_text"]
    # print(answer)

    answer = query_ollama(final_prompt)['response']
    # print(answer)

    return answer 


