import { ModelSettingsDict } from "./backend/typing";

/**
 * Overlapping + LangChain
 */
export const OverlappingLangChainSchema: ModelSettingsDict = {
  fullName: "Overlapping + LangChain",
  description: "Chunk text via LangChain's RecursiveCharacterTextSplitter.",
  schema: {
    type: "object",
    required: ["chunk_size", "chunk_overlap"],
    properties: {
      chunk_size: { 
        type: "number", 
        default: 50, 
        title: "Chunk Size",
        minimum: 0,
        maximum: 500
      },
      chunk_overlap: { 
        type: "number", 
        default: 10, 
        title: "Overlap",
        minimum: 0,
        maximum: 500
      },
    },
  },
  uiSchema: {
    chunk_size: {
      "ui:widget": "range",
      "ui:description": "Size of each chunk (0-500)",
      "ui:options": {
        step: 1
      }
    },
    chunk_overlap: {
      "ui:widget": "range",
      "ui:description": "Overlap between chunks (0-500)",
      "ui:options": {
        step: 1
      }
    },
  },
  postprocessors: {},
};

/**
 * Overlapping + OpenAI tiktoken
 */
export const OverlappingOpenAITiktokenSchema: ModelSettingsDict = {
  fullName: "Overlapping + OpenAI tiktoken",
  description: "Chunk text using the OpenAI tiktoken library with overlap.",
  schema: {
    type: "object",
    required: ["chunk_size", "chunk_overlap"],
    properties: {
      chunk_size: {
        type: "number",
        default: 50,
        title: "Chunk Size (Tokens)",
        minimum: 0,
        maximum: 500
      },
      chunk_overlap: {
        type: "number",
        default: 10,
        title: "Overlap (Tokens)",
        minimum: 0,
        maximum: 500
      },
    },
  },
  uiSchema: {
    chunk_size: {
      "ui:widget": "range",
      "ui:description": "Size of each chunk in tokens (0-500)",
      "ui:options": {
        step: 1
      }
    },
    chunk_overlap: {
      "ui:widget": "range",
      "ui:description": "Overlap between chunks in tokens (0-500)",
      "ui:options": {
        step: 1
      }
    },
  },
  postprocessors: {},
};

/**
 * Overlapping + HuggingFace Tokenizers
 */
export const OverlappingHuggingfaceTokenizerSchema: ModelSettingsDict = {
  fullName: "Overlapping + HuggingFace Tokenizers",
  description: "Chunk text using HuggingFace tokenizer-based segmentation.",
  schema: {
    type: "object",
    required: ["tokenizer_model", "chunk_size"],
    properties: {
      tokenizer_model: {
        type: "string",
        default: "bert-base-uncased",
        title: "Tokenizer Model",
        enum: ["bert-base-uncased", "roberta-base", "gpt2"],
      },
      chunk_size: {
        type: "number",
        default: 50,
        title: "Approx. tokens per chunk",
        minimum: 0,
        maximum: 500
      },
    },
  },
  uiSchema: {
    tokenizer_model: {
      "ui:widget": "select",
    },
    chunk_size: {
      "ui:widget": "range",
      "ui:description": "Size of each chunk in tokens (0-500)",
      "ui:options": {
        step: 1
      }
    },
  },
  postprocessors: {},
};

/**
 * Syntax-based spaCy
 */
export const SyntaxSpacySchema: ModelSettingsDict = {
  fullName: "Syntax-based spaCy",
  description: "Splits text into sentences using spaCy.",
  schema: { type: "object", required: [], properties: {} },
  uiSchema: {},
  postprocessors: {},
};

/**
 * Syntax-based TextTiling
 */
export const SyntaxTextTilingSchema: ModelSettingsDict = {
  fullName: "Syntax-based TextTiling",
  description: "Splits text into multi-sentence segments using TextTiling.",
  schema: {
    type: "object",
    required: ["w", "k"],
    properties: {
      w: { 
        type: "number", 
        default: 20, 
        title: "Window size (w)",
        minimum: 5,
        maximum: 50
      },
      k: { 
        type: "number", 
        default: 10, 
        title: "Block comparison size (k)",
        minimum: 5,
        maximum: 50
      },
    },
  },
  uiSchema: {
    w: {
      "ui:widget": "range",
      "ui:description": "Window size (5-50)",
      "ui:options": {
        step: 1
      }
    },
    k: {
      "ui:widget": "range",
      "ui:description": "Block comparison size (5-50)",
      "ui:options": {
        step: 1
      }
    },
  },
  postprocessors: {},
};

/**
 * Hybrid: TextTiling + spaCy
 */
export const HybridTextTilingSpacySchema: ModelSettingsDict = {
  fullName: "Hybrid: TextTiling + spaCy",
  description:
    "Combines TextTiling for broad segmentation, then spaCy for finer splits.",
  schema: {
    type: "object",
    required: ["w", "k"],
    properties: {
      w: { 
        type: "number", 
        default: 20, 
        title: "Window size (w)",
        minimum: 5,
        maximum: 50
      },
      k: { 
        type: "number", 
        default: 10, 
        title: "Block comparison size (k)",
        minimum: 5,
        maximum: 50
      },
    },
  },
  uiSchema: {
    w: {
      "ui:widget": "range",
      "ui:description": "Window size (5-50)",
      "ui:options": {
        step: 1
      }
    },
    k: {
      "ui:widget": "range",
      "ui:description": "Block comparison size (5-50)",
      "ui:options": {
        step: 1
      }
    },
  },
  postprocessors: {},
};

/**
 * Hybrid: BERTopic + spaCy
 */
export const HybridBERTopicSchema: ModelSettingsDict = {
  fullName: "BERTopic + spaCy",
  description: "Splits text using a hybrid approach with BERTopic + spaCy.",
  schema: {
    type: "object",
    required: ["min_topic_size"],
    properties: {
      min_topic_size: {
        type: "number",
        default: 2,
        title: "Min Topic Size",
        minimum: 2,
        maximum: 20,
        validationMessage: "Value must be between 2 and 20."
      },
    },
  },
  uiSchema: {
    min_topic_size: {
      "ui:widget": "range",
      "ui:description": "Minimum topic size (2-20)",
      "ui:options": {
        step: 1
      }
    },
  },
  postprocessors: {},
};

/**
 * Hybrid: Recursive + Gensim
 */
export const HybridRecursiveGensimSchema: ModelSettingsDict = {
  fullName: "Hybrid: Recursive TextSplitter + Gensim",
  description: "Combines a recursive approach with Gensim's text modeling.",
  schema: {
    type: "object",
    required: ["max_words"],
    properties: {
      max_words: {
        type: "number",
        default: 300,
        title: "Words per chunk",
        minimum: 0,
        maximum: 500
      },
    },
  },
  uiSchema: {
    max_words: {
      "ui:widget": "range",
      "ui:description": "Maximum words per chunk (0-500)",
      "ui:options": {
        step: 5
      }
    },
  },
  postprocessors: {},
};

/**
 * Hybrid: Recursive + Cohere
 */
export const HybridRecursiveCohereSchema: ModelSettingsDict = {
  fullName: "Hybrid: Recursive TextSplitter + Cohere",
  description: "Uses Cohere's embeddings to guide recursive chunking.",
  schema: {
    type: "object",
    required: ["max_tokens", "threshold"],
    properties: {
      max_tokens: {
        type: "number",
        default: 500,
        title: "Max tokens per chunk",
        minimum: 0,
        maximum: 500
      },
      threshold: {
        type: "number",
        default: 0.75,
        title: "Embedding similarity threshold",
        minimum: 0.0,
        maximum: 1.0
      },
    },
  },
  uiSchema: {
    max_tokens: {
      "ui:widget": "range",
      "ui:description": "Maximum tokens per chunk (0-500)",
      "ui:options": {
        step: 5
      }
    },
    threshold: {
      "ui:widget": "range",
      "ui:description": "Similarity threshold (0.0-1.0)",
      "ui:options": {
        step: 0.05
      }
    },
  },
  postprocessors: {},
};

/**
 * Hybrid: Recursive + BERTopic
 */
export const HybridRecursiveBERTopicSchema: ModelSettingsDict = {
  fullName: "Hybrid: Recursive TextSplitter + BERTopic",
  description:
    "Uses a recursive approach combined with BERTopic for semantic grouping.",
  schema: {
    type: "object",
    required: ["min_topic_size", "chunk_size"],
    properties: {
      min_topic_size: {
        type: "number",
        default: 2,
        title: "Min Topic Size",
        minimum: 2,
        maximum: 20
      },
      chunk_size: {
        type: "number",
        default: 300,
        title: "Base chunk size",
        minimum: 0,
        maximum: 500
      },
    },
  },
  uiSchema: {
    min_topic_size: {
      "ui:widget": "range",
      "ui:description": "Minimum topic size (2-20)",
      "ui:options": {
        step: 1
      }
    },
    chunk_size: {
      "ui:widget": "range", 
      "ui:description": "Base chunk size (0-500)",
      "ui:options": {
        step: 5
      }
    },
  },
  postprocessors: {},
};

export const ChunkMethodSchemas: { [baseMethod: string]: ModelSettingsDict } = {
  overlapping_langchain: OverlappingLangChainSchema,
  overlapping_openai_tiktoken: OverlappingOpenAITiktokenSchema,
  overlapping_huggingface_tokenizers: OverlappingHuggingfaceTokenizerSchema,
  syntax_spacy: SyntaxSpacySchema,
  syntax_texttiling: SyntaxTextTilingSchema,
  hybrid_texttiling_spacy: HybridTextTilingSpacySchema,
  hybrid_bertopic_spacy: HybridBERTopicSchema,
  hybrid_recursive_gensim: HybridRecursiveGensimSchema,
  hybrid_recursive_cohere: HybridRecursiveCohereSchema,
  hybrid_recursive_bertopic: HybridRecursiveBERTopicSchema,
};
