from langchain.schema import Document
from langchain.vectorstores import FAISS

import os


def get_start():
    document = Document(page_content="init", metadata={})
    doc = document

    os.environ["OPENAI_API_KEY"]="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"

    from langchain.embeddings.openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()
    embedding = embeddings.embed_query(document.page_content)

    vector_store = FAISS.from_documents([doc], embeddings, normalize_L2=True)

    check_zeros(embedding)
    print(embedding)

    print("Finish started.")


def check_zeros(arr):
    if 0 in arr:
        print("Array contains zero values.")
    else:
        print("Array does not contain any zero values.")


if __name__ == "__main__":
    get_start()
