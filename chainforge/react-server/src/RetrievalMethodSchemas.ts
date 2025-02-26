export interface RetrievalChunk {
  text: string;
  similarity: number;
  docTitle?: string;
  chunkId?: string;
  chunkMethod?: string;
}

export interface RetrievalMethodResult {
  label: string;
  retrieved: RetrievalChunk[];
}

export interface RetrievalResults {
  [methodKey: string]: RetrievalMethodResult;
}

export interface RetrievalMethodSpec {
  key: string;
  baseMethod: string;
  methodName: string;
  group: string;
  needsVector: boolean;
  vectorLib?: string;
  settings?: Record<string, any>;
  displayName?: string;
}

export const defaultMethodEmojis: { [key: string]: string } = {
  bm25: "📚",
  tfidf: "📈",
  boolean: "🧩",
  overlap: "🤝",
  cosine: "💡",
  sentenceEmbeddings: "🧠",
  customVector: "✨",
  clustered: "🗃️",
};

export const vectorOptions = [
  { label: "🤗 HuggingFace Transformers", value: "HuggingFace Transformers" },
  { label: "🦄 OpenAI Embeddings", value: "OpenAI Embeddings" },
  { label: "🧠 Cohere Embeddings", value: "Cohere Embeddings" },
  { label: "💬 Sentence Transformers", value: "Sentence Transformers" },
];

export const retrievalMethodGroups = [
  {
    label: "Keyword-based Retrieval",
    items: [
      {
        baseMethod: "bm25",
        methodName: "BM25",
        group: "Keyword-based Retrieval",
        needsVector: false,
      },
      {
        baseMethod: "tfidf",
        methodName: "TF-IDF",
        group: "Keyword-based Retrieval",
        needsVector: false,
      },
      {
        baseMethod: "boolean",
        methodName: "Boolean Search",
        group: "Keyword-based Retrieval",
        needsVector: false,
      },
      {
        baseMethod: "overlap",
        methodName: "Keyword Overlap",
        group: "Keyword-based Retrieval",
        needsVector: false,
      },
    ],
  },
  {
    label: "Embedding-based Retrieval",
    items: [
      {
        baseMethod: "cosine",
        methodName: "Cosine Similarity",
        group: "Embedding-based Retrieval",
        needsVector: true,
      },
      {
        baseMethod: "sentenceEmbeddings",
        methodName: "Sentence Embeddings",
        group: "Embedding-based Retrieval",
        needsVector: true,
      },
      {
        baseMethod: "customVector",
        methodName: "Custom Vector Search",
        group: "Embedding-based Retrieval",
        needsVector: true,
      },
      {
        baseMethod: "clustered",
        methodName: "Clustered Embedding",
        group: "Embedding-based Retrieval",
        needsVector: true,
      },
    ],
  },
];
