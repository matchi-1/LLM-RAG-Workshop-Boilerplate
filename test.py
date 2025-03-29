CHROMA_PATH = "./chroma_db"
from langchain.vectorstores import Chroma
from ingest import TogetherEmbeddings


my_db = Chroma(persist_directory=CHROMA_PATH, embeddings = TogetherEmbeddings())

for collection in my_db._client.list_collections():
  ids = collection.get()['ids']
  print('REMOVE %s document(s) from %s collection' % (str(len(ids)), collection.name))
  if len(ids): collection.delete(ids)