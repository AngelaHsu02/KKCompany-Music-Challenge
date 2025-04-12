python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input corpus/ \
  --index indexes/collection_jsonl_sparse \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --storePositions --storeDocvectors --storeContents \
  --stemmer none \
  --keepStopwords