// types.ts
export type RetrievalCategory = "Vector" | "Sparse";

export type RetrievalType = "vectorization" | "retrieval";

export interface RetrievalMethodSpec {
  key: string; // Unique identifier
  category: RetrievalCategory;
  name: string; // e.g., "HuggingFace Transformers", "BM25"
  type: RetrievalType;
  settings?: Record<string, any>; // Additional settings specific to the method
}

export interface ChunkMetadata {
  id: string;
  method: string;
  title: string;
}

export interface ChunkVar {
  text: string;
  metadata: ChunkMetadata;
}

export interface RetrievalResult {
  chunkId: string;
  text: string;
  similarity: number;
  method: string;
  title: string;
}
