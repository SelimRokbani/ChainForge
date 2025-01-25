export const BM25Schema = {
  fullName: "BM25 Sparse Retrieval",
  description: "A traditional term-based retrieval method using BM25 scoring.",
  schema: {
    type: "object",
    required: ["top_k"],
    properties: {
      top_k: { type: "number", default: 5, title: "Top-K Results" },
    },
  },
  uiSchema: {
    top_k: { "ui:widget": "updown" },
  },
  postprocessors: {},
};

export const TFIDFSchema = {
  fullName: "TF-IDF Sparse Retrieval",
  description:
    "Retrieves documents using term frequency-inverse document frequency.",
  schema: {
    type: "object",
    required: ["top_k"],
    properties: {
      top_k: { type: "number", default: 5, title: "Top-K Results" },
    },
  },
  uiSchema: {
    top_k: { "ui:widget": "updown" },
  },
  postprocessors: {},
};

export const DPRSchema = {
  fullName: "Dense Passage Retrieval (DPR)",
  description:
    "A dense retriever that uses pre-trained transformers for embedding-based retrieval.",
  schema: {
    type: "object",
    required: ["top_k", "embedding_model"],
    properties: {
      top_k: { type: "number", default: 5, title: "Top-K Results" },
      embedding_model: {
        type: "string",
        default: "facebook/dpr-ctx_encoder-single-nq-base",
        title: "Embedding Model",
        enum: [
          "facebook/dpr-ctx_encoder-single-nq-base",
          "facebook/dpr-ctx_encoder-multiset-base",
          "deepset/roberta-base-squad2",
        ],
      },
    },
  },
  uiSchema: {
    top_k: { "ui:widget": "updown" },
    embedding_model: { "ui:widget": "select" },
  },
  postprocessors: {},
};

export const HuggingFaceSchema = {
  fullName: "HuggingFace Transformers",
  description:
    "Uses a transformer-based model from HuggingFace for dense retrieval.",
  schema: {
    type: "object",
    required: ["top_k", "embedding_model"],
    properties: {
      top_k: { type: "number", default: 5, title: "Top-K Results" },
      embedding_model: {
        type: "string",
        default: "sentence-transformers/all-MiniLM-L6-v2",
        title: "Embedding Model",
        enum: [
          "sentence-transformers/all-MiniLM-L6-v2",
          "sentence-transformers/all-MPNet-base-v2",
          "sentence-transformers/msmarco-distilbert-base-dot-prod-v3",
        ],
      },
    },
  },
  uiSchema: {
    top_k: { "ui:widget": "updown" },
    embedding_model: { "ui:widget": "select" },
  },
  postprocessors: {},
};

export const SentenceTransformersSchema = {
  fullName: "Sentence Transformers",
  description: "Uses Sentence Transformers for vector-based retrieval.",
  schema: {
    type: "object",
    required: ["top_k", "embedding_model"],
    properties: {
      top_k: { type: "number", default: 5, title: "Top-K Results" },
      embedding_model: {
        type: "string",
        default: "all-MiniLM-L6-v2",
        title: "Embedding Model",
        enum: [
          "all-MiniLM-L6-v2",
          "all-mpnet-base-v2",
          "paraphrase-multilingual-MiniLM-L12-v2",
        ],
      },
    },
  },
  uiSchema: {
    top_k: { "ui:widget": "updown" },
    embedding_model: { "ui:widget": "select" },
  },
  postprocessors: {},
};

export const CohereSchema = {
  fullName: "Cohere API",
  description: "Uses Cohere's embedding model for dense retrieval.",
  schema: {
    type: "object",
    required: ["top_k", "embedding_model"],
    properties: {
      top_k: { type: "number", default: 5, title: "Top-K Results" },
      embedding_model: {
        type: "string",
        default: "cohere/large",
        title: "Embedding Model",
        enum: ["cohere/large", "cohere/multilingual", "cohere/small"],
      },
    },
  },
  uiSchema: {
    top_k: { "ui:widget": "updown" },
    embedding_model: { "ui:widget": "select" },
  },
  postprocessors: {},
};

/** The dictionary of retrieval method schemas, keyed by 'baseMethod' */
export const RetrievalMethodSchemas: { [baseMethod: string]: any } = {
  bm25: BM25Schema,
  tfidf: TFIDFSchema,
  dpr: DPRSchema,
  huggingface: HuggingFaceSchema,
  "sentence-transformers": SentenceTransformersSchema,
  cohere: CohereSchema,
};
