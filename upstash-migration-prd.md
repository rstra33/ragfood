Create a detailed design document to replace ChromaDB with Upstash Vector Database. I have added the Upstash Vector credentials in the .env file.

Key Information about Upstash Vector:
- Built-in embedding model: mixedbread-ai/mxbai-embed-large-v1
- 1024 dimensions, 512 sequence length, MTEB score 64.68
- Automatic text vectorization (no need for external embedding API)
- Serverless and cloud-hosted
- Cosine similarity for semantic search

Current Implementation Details:
- Using ChromaDB for local vector storage
- Using Ollama's mxbai-embed-large for embeddings
- Python-based RAG system with food data
- Manual embedding generation and upsert process

Requirements for Migration:
1. Replace ChromaDB client with Upstash Vector client
2. Remove manual embedding generation (Upstash handles this automatically)
3. Update upsert process to use raw text data instead of pre-computed embeddings
4. Modify query process to work with Upstash Vector API
5. Handle authentication and error management
6. Maintain the same RAG functionality and user experience

Study the following sites to guide the design:
https://upstash.com/docs/vector/features/embeddingmodels 


Please provide:
- Architecture comparison (before vs after)
- Detailed implementation plan
- Code structure changes required
- API differences and implications
- Error handling strategies
- Performance considerations
- Cost implications of cloud vs local
- Security considerations for API keys