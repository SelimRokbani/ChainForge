// ChunkMethodSchemas.ts
import { ModelSettingsDict } from "./backend/typing";

export const OverlappingLangChainSchema: ModelSettingsDict = {
  fullName: "Overlapping + LangChain",
  description: "Chunk text via LangChain's RecursiveCharacterTextSplitter.",
  schema: {
    type: "object",
    required: ["chunk_size", "chunk_overlap"],
    properties: {
      chunk_size: { type: "number", default: 300, title: "Chunk Size" },
      chunk_overlap: { type: "number", default: 50, title: "Overlap" },
    },
  },
  uiSchema: {
    chunk_size: { "ui:widget": "updown" },
    chunk_overlap: { "ui:widget": "updown" },
  },
  postprocessors: {},
};

export const SyntaxSpacySchema: ModelSettingsDict = {
  fullName: "Syntax-based spaCy",
  description: "Splits text into sentences using spaCy.",
  schema: { type: "object", required: [], properties: {} },
  uiSchema: {},
  postprocessors: {},
};

export const HybridBERTopicSchema: ModelSettingsDict = {
  fullName: "BERTopic + spaCy",
  description: "Splits text using a hybrid approach with BERTopic + spaCy.",
  schema: {
    type: "object",
    required: ["min_topic_size"],
    properties: {
      min_topic_size: { type: "number", default: 2, title: "Min Topic Size" },
    },
  },
  uiSchema: {
    min_topic_size: { "ui:widget": "updown" },
  },
  postprocessors: {},
};

/**
 * The dictionary of chunk method schemas, keyed by 'baseMethod'
 */
export const ChunkMethodSchemas: { [baseMethod: string]: ModelSettingsDict } = {
  overlapping_langchain: OverlappingLangChainSchema,
  syntax_spacy: SyntaxSpacySchema,
  hybrid_bertopic: HybridBERTopicSchema,
};
