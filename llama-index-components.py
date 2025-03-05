from datasets import load_dataset
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from huggingface_hub import login
import chromadb
import asyncio
import nest_asyncio
import os
from dotenv import load_dotenv

def load_dataset_to_local():
    dataset = load_dataset(path="dvilasuero/finepersonas-v0.1-tiny", split="train")
    Path("data").mkdir(parents=True, exist_ok=True)
    for i, persona in enumerate(dataset):
        with open(Path("data") / f"persona_{i}.txt", "w") as f:
            f.write(persona["persona"])

def load_docs_in_dataset():
    reader = SimpleDirectoryReader(input_dir="data")
    documents = reader.load_data()
    return documents

async def run_pipeline_and_store(documents, vector_store):
    # create the pipeline with transformations
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(),
            HuggingFaceInferenceAPIEmbedding(model_name="BAAI/bge-small-en-v1.5"),
        ],
        vector_store=vector_store,
    )
    nodes = await pipeline.arun(documents=documents[:100])

# store into vector DB
def store_documents(vector_store):
    # data
    documents = load_docs_in_dataset()

    # store
    asyncio.run(run_pipeline_and_store(documents=documents, vector_store=vector_store))

def query_index_and_model(vector_store, query):
    embed_model = HuggingFaceInferenceAPIEmbedding(model_name="BAAI/bge-small-en-v1.5")
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )
    nest_asyncio.apply()  # This is needed to run the query engine
    llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="tree_summarize",
    )
    response = query_engine.query(query)
    return response

load_dotenv(override=True)
token = os.getenv('HF_TOKEN')
login(token)

#load_dataset_to_local()
# vector db
db = chromadb.PersistentClient(path="./alfred_chroma_db")
chroma_collection = db.get_or_create_collection(name="alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

#store_documents(vector_store)
response = query_index_and_model(vector_store, "Respond using a persona that describes author and travel experiences?")
print(response)
'''
An author with a deep passion for storytelling, particularly in the realm of children's literature, who crafts biographical tales about influential historical figures. This author specializes in exploring the formative years and the early roots of social activism in these figures, bringing their stories to life in a way that resonates with young readers. While the provided information does not directly mention travel experiences, one can imagine that such an author might draw inspiration from visiting historical sites and museums related to the figures they write about, enriching their narratives with firsthand experiences and insights
'''

