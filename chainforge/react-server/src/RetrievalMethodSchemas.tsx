import { ModelSettingsDict } from "./backend/typing";

/**
 * BM25 Retrieval
 */
export const BM25Schema: ModelSettingsDict = {
  fullName: "BM25 Retrieval",
  description: "Retrieves documents using the BM25 ranking algorithm",
  schema: {
    type: "object",
    required: ["top_k", "bm25_k1", "bm25_b"],
    properties: {
      top_k: {
        type: "number",
        default: 5,
        title: "Top K Results",
      },
      bm25_k1: {
        type: "number",
        default: 1.5,
        title: "k1 Parameter",
      },
      bm25_b: {
        type: "number",
        default: 0.75,
        title: "b Parameter",
      },
    },
  },
  uiSchema: {
    top_k: {
      "ui:widget": "range",
      "ui:options": {
        min: 1,
        max: 20,
        step: 1,
      },
    },
    bm25_k1: {
      "ui:widget": "range",
      "ui:options": {
        min: 0.5,
        max: 3.0,
        step: 0.1,
      },
    },
    bm25_b: {
      "ui:widget": "range",
      "ui:options": {
        min: 0,
        max: 1,
        step: 0.05,
      },
    },
  },
  postprocessors: {},
};

/**
 * TF-IDF Retrieval
 */
export const TFIDFSchema: ModelSettingsDict = {
  fullName: "TF-IDF Retrieval",
  description: "Retrieves documents using TF-IDF scoring",
  schema: {
    type: "object",
    required: ["top_k", "max_features"],
    properties: {
      top_k: {
        type: "number",
        default: 5,
        title: "Top K Results",
      },
      max_features: {
        type: "number",
        default: 500,
        title: "Max Features",
      },
    },
  },
  uiSchema: {
    top_k: {
      "ui:widget": "range",
      "ui:options": {
        min: 1,
        max: 20,
        step: 1,
      },
    },
    max_features: {
      "ui:widget": "range",
      "ui:options": {
        min: 100,
        max: 2000,
        step: 100,
      },
    },
  },
  postprocessors: {},
};

/**
 * Boolean Search
 */
export const BooleanSearchSchema: ModelSettingsDict = {
  fullName: "Boolean Search",
  description: "Simple boolean keyword matching",
  schema: {
    type: "object",
    required: ["top_k", "required_match_count"],
    properties: {
      top_k: {
        type: "number",
        default: 5,
        title: "Top K Results",
      },
      required_match_count: {
        type: "number",
        default: 1,
        title: "Required Matches",
      },
    },
  },
  uiSchema: {
    top_k: {
      "ui:widget": "range",
      "ui:options": {
        min: 1,
        max: 20,
        step: 1,
      },
    },
    required_match_count: {
      "ui:widget": "range",
      "ui:options": {
        min: 1,
        max: 10,
        step: 1,
      },
    },
  },
  postprocessors: {},
};

/**
 * Keyword Overlap
 */
export const KeywordOverlapSchema: ModelSettingsDict = {
  fullName: "Keyword Overlap",
  description: "Retrieves documents based on keyword overlap ratio",
  schema: {
    type: "object",
    required: ["top_k", "normalization_factor"],
    properties: {
      top_k: {
        type: "number",
        default: 5,
        title: "Top K Results",
      },
      normalization_factor: {
        type: "number",
        default: 0.75,
        title: "Normalization Factor",
      },
    },
  },
  uiSchema: {
    top_k: {
      "ui:widget": "range",
      "ui:options": {
        min: 1,
        max: 20,
        step: 1,
      },
    },
    normalization_factor: {
      "ui:widget": "range",
      "ui:options": {
        min: 0,
        max: 1,
        step: 0.05,
      },
    },
  },
  postprocessors: {},
};

/**
 * Cosine Similarity Schema
 */
export const CosineSimilaritySchema: ModelSettingsDict = {
  fullName: "Cosine Similarity",
  description: "Retrieves documents using cosine similarity between embeddings",
  schema: {
    type: "object",
    required: ["top_k", "similarity_threshold"],
    properties: {
      top_k: {
        type: "number",
        default: 5,
        title: "Top K Results",
      },
      similarity_threshold: {
        type: "number",
        default: 0.7,
        title: "Similarity Threshold",
      },
    },
  },
  uiSchema: {
    top_k: {
      "ui:widget": "range",
      "ui:options": {
        min: 1,
        max: 20,
        step: 1,
      },
    },
    similarity_threshold: {
      "ui:widget": "range",
      "ui:options": {
        min: 0,
        max: 1,
        step: 0.05,
      },
    },
  },
  postprocessors: {},
};

/**
 * Sentence Embeddings Schema
 */
export const SentenceEmbeddingsSchema: ModelSettingsDict = {
  fullName: "Sentence Embeddings",
  description: "Uses sentence-level embeddings for retrieval",
  schema: {
    type: "object",
    required: ["top_k", "similarity_threshold", "pooling_strategy"],
    properties: {
      top_k: {
        type: "number",
        default: 5,
        title: "Top K Results",
      },
      similarity_threshold: {
        type: "number",
        default: 0.7,
        title: "Similarity Threshold",
      },
      pooling_strategy: {
        type: "string",
        default: "mean",
        title: "Pooling Strategy",
        enum: ["mean", "max", "cls"],
      },
    },
  },
  uiSchema: {
    top_k: {
      "ui:widget": "range",
      "ui:options": {
        min: 1,
        max: 20,
        step: 1,
      },
    },
    similarity_threshold: {
      "ui:widget": "range",
      "ui:options": {
        min: 0,
        max: 1,
        step: 0.05,
      },
    },
    pooling_strategy: {
      "ui:widget": "select",
    },
  },
  postprocessors: {},
};

/**
 * Custom Vector Search Schema
 */
export const CustomVectorSchema: ModelSettingsDict = {
  fullName: "Custom Vector Search",
  description: "Use custom vector representations for retrieval",
  schema: {
    type: "object",
    required: ["top_k", "similarity_threshold", "vector_dimension"],
    properties: {
      top_k: {
        type: "number",
        default: 5,
        title: "Top K Results",
      },
      similarity_threshold: {
        type: "number",
        default: 0.7,
        title: "Similarity Threshold",
      },
      vector_dimension: {
        type: "number",
        default: 768,
        title: "Vector Dimension",
      },
    },
  },
  uiSchema: {
    top_k: {
      "ui:widget": "range",
      "ui:options": {
        min: 1,
        max: 20,
        step: 1,
      },
    },
    similarity_threshold: {
      "ui:widget": "range",
      "ui:options": {
        min: 0,
        max: 1,
        step: 0.05,
      },
    },
    vector_dimension: {
      "ui:widget": "range",
      "ui:options": {
        min: 64,
        max: 1536,
        step: 64,
      },
    },
  },
  postprocessors: {},
};

/**
 * Clustered Embedding Schema
 */
export const ClusteredEmbeddingSchema: ModelSettingsDict = {
  fullName: "Clustered Embedding",
  description: "Uses clustering for efficient embedding-based retrieval",
  schema: {
    type: "object",
    required: ["top_k", "similarity_threshold", "n_clusters"],
    properties: {
      top_k: {
        type: "number",
        default: 5,
        title: "Top K Results",
      },
      similarity_threshold: {
        type: "number",
        default: 0.7,
        title: "Similarity Threshold",
      },
      n_clusters: {
        type: "number",
        default: 5,
        title: "Number of Clusters",
      },
    },
  },
  uiSchema: {
    top_k: {
      "ui:widget": "range",
      "ui:options": {
        min: 1,
        max: 20,
        step: 1,
      },
    },
    similarity_threshold: {
      "ui:widget": "range",
      "ui:options": {
        min: 0,
        max: 1,
        step: 0.05,
      },
    },
    n_clusters: {
      "ui:widget": "range",
      "ui:options": {
        min: 2,
        max: 20,
        step: 1,
      },
    },
  },
  postprocessors: {},
};

/**
 * FAISS Vectorstore
 */
export const FAISSSchema: ModelSettingsDict = {
  fullName: "FAISS Vectorstore",
  description: "Persistent vector storage using FAISS",
  schema: {
    type: "object",
    required: ["top_k", "similarity_threshold", "faissMode", "faissPath"],
    properties: {
      top_k: {
        type: "number",
        default: 5,
        title: "Top K Results",
      },
      similarity_threshold: {
        type: "number",
        default: 0.7,
        title: "Similarity Threshold",
      },
      faissMode: {
        type: "string",
        default: "create",
        title: "FAISS Mode",
        enum: ["create", "load"],
      },
      faissPath: {
        type: "string",
        default: "",
        title: "FAISS Index Path",
      },
    },
  },
  uiSchema: {
    top_k: {
      "ui:widget": "range",
      "ui:options": {
        min: 1,
        max: 20,
        step: 1,
      },
    },
    similarity_threshold: {
      "ui:widget": "range",
      "ui:options": {
        min: 0,
        max: 1,
        step: 0.05,
      },
    },
    faissMode: {
      "ui:widget": "select",
    },
    faissPath: {
      "ui:widget": "text",
    },
  },
  postprocessors: {},
};

// Combined schema object for all retrieval methods
export const RetrievalMethodSchemas: {
  [baseMethod: string]: ModelSettingsDict;
} = {
  bm25: BM25Schema,
  tfidf: TFIDFSchema,
  boolean: BooleanSearchSchema,
  overlap: KeywordOverlapSchema,
  cosine: CosineSimilaritySchema,
  sentenceEmbeddings: SentenceEmbeddingsSchema,
  customVector: CustomVectorSchema,
  clustered: ClusteredEmbeddingSchema,
  faiss: FAISSSchema,
};

// Method groupings for the menu
export const retrievalMethodGroups = [
  {
    label: "Keyword-based Retrieval",
    items: [
      {
        baseMethod: "bm25",
        methodName: "BM25",
        library: "BM25",
        emoji: "📊",
        group: "Keyword-based Retrieval",
        needsVector: false,
      },
      {
        baseMethod: "tfidf",
        methodName: "TF-IDF",
        library: "TF-IDF",
        emoji: "📈",
        group: "Keyword-based Retrieval",
        needsVector: false,
      },
      {
        baseMethod: "boolean",
        methodName: "Boolean Search",
        library: "Boolean Search",
        emoji: "🔍",
        group: "Keyword-based Retrieval",
        needsVector: false,
      },
      {
        baseMethod: "overlap",
        methodName: "Keyword Overlap",
        library: "KeywordOverlap",
        emoji: "🎯",
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
        library: "Cosine",
        emoji: "📐",
        group: "Embedding-based Retrieval",
        needsEmbeddingModel: true,
      },
      {
        baseMethod: "sentenceEmbeddings",
        methodName: "Sentence Embeddings",
        library: "Sentence",
        emoji: "🔤",
        group: "Embedding-based Retrieval",
        needsEmbeddingModel: true,
      },
      {
        baseMethod: "customVector",
        methodName: "Custom Vector Search",
        library: "Custom",
        emoji: "🎯",
        group: "Embedding-based Retrieval",
        needsEmbeddingModel: true,
      },
      {
        baseMethod: "clustered",
        methodName: "Clustered Embedding",
        library: "Clustered",
        emoji: "🎲",
        group: "Embedding-based Retrieval",
        needsEmbeddingModel: true,
      },
    ],
  },
  {
    label: "Vectorstores",
    items: [
      {
        baseMethod: "faiss",
        methodName: "FAISS Vectorstore",
        library: "FAISS",
        emoji: "💾",
        group: "Vectorstores",
        needsEmbeddingModel: true,
      },
    ],
  },
];

// Available embedding models
export const embeddingModels = [
  {
    label: "HuggingFace Transformers",
    value: "HuggingFace Transformers",
    emoji: "🤗",
  },
  {
    label: "OpenAI Embeddings",
    value: "OpenAI Embeddings",
    emoji: "🤖",
  },
  {
    label: "Cohere Embeddings",
    value: "Cohere Embeddings",
    emoji: "💬",
  },
  {
    label: "Sentence Transformers",
    value: "Sentence Transformers",
    emoji: "🧠",
  },
];
