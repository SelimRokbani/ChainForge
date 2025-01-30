import React, { useState, useEffect, useCallback, useRef } from "react";
import { Handle, Position } from "reactflow";
import {
  Button,
  TextInput,
  Group,
  Text,
  LoadingOverlay,
  ActionIcon,
  Menu,
  Divider,
} from "@mantine/core";
import {
  IconSearch,
  IconPlus,
  IconSettings,
  IconTrash,
} from "@tabler/icons-react";
import { v4 as uuid } from "uuid";

import BaseNode from "./BaseNode";
import NodeLabel from "./NodeLabelComponent";
import useStore from "./store";
import InspectFooter from "./InspectFooter";


import LLMResponseInspectorModal, {
  LLMResponseInspectorModalRef,
} from "./LLMResponseInspectorModal";
import { LLMResponse } from "./backend/typing";

// ---------- Type Declarations (inline) ----------
export interface RetrievalChunk {
  text: string;
  similarity: number;
  docTitle?: string;
  chunkId?: string;
  chunkMethod?: string;
}

export interface RetrievalMethodResult {
  label: string; // "BM25" or "Cosine Similarity (HuggingFace Transformers)"
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
}

// Example data for retrieval methods
const vectorOptions = [
  "HuggingFace Transformers",
  "OpenAI Embeddings",
  "Cohere Embeddings",
  "Sentence Transformers",
];

const retrievalMethodGroups = [
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

// ---------- UI for selecting retrieval methods -----------
interface RetrievalMethodListContainerProps {
  initMethodItems?: RetrievalMethodSpec[];
  onItemsChange?: (
    newItems: RetrievalMethodSpec[],
    oldItems: RetrievalMethodSpec[],
  ) => void;
}

const RetrievalMethodListContainer: React.FC<
  RetrievalMethodListContainerProps
> = ({ initMethodItems = [], onItemsChange }) => {
  const [methodItems, setMethodItems] =
    useState<RetrievalMethodSpec[]>(initMethodItems);
  const [menuOpened, setMenuOpened] = useState(false);

  const oldItemsRef = useRef<RetrievalMethodSpec[]>(initMethodItems);

  const notifyItemsChanged = useCallback(
    (newItems: RetrievalMethodSpec[]) => {
      onItemsChange?.(newItems, oldItemsRef.current);
      oldItemsRef.current = newItems;
    },
    [onItemsChange],
  );

  const addMethod = useCallback(
    (
      method: Omit<RetrievalMethodSpec, "key" | "vectorLib">,
      chosenLibrary?: string,
    ) => {
      const newItem: RetrievalMethodSpec = {
        ...method,
        key: uuid(),
        vectorLib: method.needsVector ? chosenLibrary : undefined,
      };
      const updated = [...methodItems, newItem];
      setMethodItems(updated);
      notifyItemsChanged(updated);
    },
    [methodItems, notifyItemsChanged],
  );

  const handleRemoveMethod = useCallback(
    (key: string) => {
      const newItems = methodItems.filter((m) => m.key !== key);
      setMethodItems(newItems);
      notifyItemsChanged(newItems);
    },
    [methodItems, notifyItemsChanged],
  );

  return (
    <div style={{ border: "1px dashed #ccc", borderRadius: 6, padding: 8 }}>
      <Group position="apart" mb="xs">
        <Text weight={500} size="sm">
          Selected Retrieval Methods
        </Text>
        <Menu
          opened={menuOpened}
          onChange={setMenuOpened}
          position="bottom-end"
          withinPortal
        >
          <Menu.Target>
            <Button
              size="xs"
              variant="light"
              leftIcon={<IconPlus size={14} />}
              onClick={() => setMenuOpened((o) => !o)}
            >
              Add +
            </Button>
          </Menu.Target>
          <Menu.Dropdown>
            {retrievalMethodGroups.map((group, groupIdx) => (
              <React.Fragment key={group.label}>
                <Menu.Label>{group.label}</Menu.Label>
                {group.items.map((item) => {
                  if (item.needsVector) {
                    // Multi-level menu if user must choose a library
                    return (
                      <Menu
                        key={item.baseMethod}
                        trigger="hover"
                        position="right-start"
                      >
                        <Menu.Target>
                          <Menu.Item icon={<IconSettings size={14} />}>
                            {item.methodName}
                          </Menu.Item>
                        </Menu.Target>
                        <Menu.Dropdown>
                          {vectorOptions.map((lib) => (
                            <Menu.Item
                              key={lib}
                              onClick={() => {
                                addMethod(item, lib);
                                setMenuOpened(false);
                              }}
                            >
                              {lib}
                            </Menu.Item>
                          ))}
                        </Menu.Dropdown>
                      </Menu>
                    );
                  }
                  return (
                    <Menu.Item
                      key={item.baseMethod}
                      icon={<IconSettings size={14} />}
                      onClick={() => {
                        addMethod(item);
                        setMenuOpened(false);
                      }}
                    >
                      {item.methodName}
                    </Menu.Item>
                  );
                })}
                {groupIdx < retrievalMethodGroups.length - 1 && (
                  <Divider my="xs" />
                )}
              </React.Fragment>
            ))}
          </Menu.Dropdown>
        </Menu>
      </Group>

      {methodItems.length === 0 ? (
        <Text size="xs" color="dimmed">
          No retrieval methods selected.
        </Text>
      ) : (
        methodItems.map((item) => (
          <div key={item.key} style={{ marginTop: 8 }}>
            <Group position="apart" align="center">
              <div style={{ maxWidth: "70%" }}>
                <Text size="sm" weight={600}>
                  {item.methodName}
                  {item.vectorLib ? ` (${item.vectorLib})` : ""}
                </Text>
              </div>
              <Group spacing="xs">
                <ActionIcon
                  color="red"
                  variant="subtle"
                  onClick={() => handleRemoveMethod(item.key)}
                >
                  <IconTrash size={16} />
                </ActionIcon>
              </Group>
            </Group>
          </div>
        ))
      )}
    </div>
  );
};

// --------------- Main Retrieval Node ---------------
interface RetrievalNodeProps {
  data: {
    title?: string;
    methods?: RetrievalMethodSpec[];
    query?: string;
    refresh?: boolean;
    results?: RetrievalResults;
  };
  id: string;
}

const RetrievalNode: React.FC<RetrievalNodeProps> = ({ data, id }) => {
  const nodeDefaultTitle = "Retrieval Node";
  const nodeIcon = "🔍";

  const setDataPropsForNode = useStore((s) => s.setDataPropsForNode);
  const pingOutputNodes = useStore((s) => s.pingOutputNodes);

  const [query, setQuery] = useState<string>(data.query || "");
  const [methodItems, setMethodItems] = useState<RetrievalMethodSpec[]>(
    data.methods || [],
  );
  const [results, setResults] = useState<RetrievalResults>(data.results || {});
  const [loading, setLoading] = useState(false);

  // We'll keep LLMResponse array for the inspector
  const [jsonResponses, setJsonResponses] = useState<LLMResponse[]>([]);

  // The inspector modal ref
  const inspectorModalRef = useRef<LLMResponseInspectorModalRef>(null);

  // If the user triggers refresh from outside
  useEffect(() => {
    if (data.refresh) {
      setDataPropsForNode(id, { refresh: false, results: {} });
      setResults({});
      setJsonResponses([]);
    }
  }, [data.refresh, id, setDataPropsForNode]);

  // If the user changes retrieval methods
  const handleMethodsChange = useCallback((newItems: RetrievalMethodSpec[]) => {
    setMethodItems(newItems);
    // Remove old results that no longer match
    setResults((prev) => {
      const updated = { ...prev };
      Object.keys(updated).forEach((k) => {
        if (!newItems.some((it) => it.key === k)) {
          delete updated[k];
        }
      });
      return updated;
    });
    // Clear old inspector data
    setJsonResponses([]);
  }, []);

  // Flatten & deduplicate final retrieved chunks
  const prepareOutput = useCallback((resultsData: RetrievalResults) => {
    const allChunks: RetrievalChunk[] = [];
    Object.values(resultsData).forEach((methodObj) => {
      allChunks.push(...methodObj.retrieved);
    });

    // Sort by similarity
    allChunks.sort((a, b) => b.similarity - a.similarity);

    // Déduplication par chunkId pour éviter les doublons
    const seen = new Set<string>();
    return allChunks
      .filter((ch) => {
        if (ch.chunkId && seen.has(ch.chunkId)) return false;
        if (ch.chunkId) seen.add(ch.chunkId);
        return true;
      })
      .map((chunk) => ({
        text: chunk.text,
        similarity: chunk.similarity,
        chunkId: chunk.chunkId || "No ID",
      }));
  }, []);

  const buildLLMResponses = (resultsData: RetrievalResults): LLMResponse[] => {
    const arr: LLMResponse[] = [];

    Object.entries(resultsData).forEach(([methodKey, methodObj]) => {
      const methodLabel = methodObj.label; // e.g. "BM25" or "Cosine Similarity"
      // slice to top 5 if desired
      const topFive = methodObj.retrieved.slice(0, 5);

      topFive.forEach((chunk, idx) => {
        const cUid = chunk.chunkId || `retrieved_${methodKey}_${idx}`;

        arr.push({
          uid: cUid,
          prompt: `Retrieved by: ${methodLabel}`,
          vars: {
            similarity: chunk.similarity.toFixed(3),
            docTitle: chunk.docTitle || "Untitled",
            chunkId: chunk.chunkId || "",
            chunkMethod: chunk.chunkMethod || "",
          },
          responses: [`[Chunk ID: ${cUid}]\n${chunk.text}`],
          llm: methodLabel,
          metavars: {
            retrievalMethod: methodLabel,
            docTitle: chunk.docTitle,
            chunkId: chunk.chunkId,
            chunkMethod: chunk.chunkMethod,
            similarity: chunk.similarity,
          },
        });
      });
    });

    return arr;
  };

  // The main retrieval function
  const runRetrieval = useCallback(async () => {
    if (!query.trim()) {
      alert("Please enter a search query.");
      return;
    }

    // Pull data from chunking node
    let upstreamData: { fields?: Array<any> } = {};
    try {
      upstreamData = useStore.getState().pullInputData(["fields"], id) as {
        fields?: Array<any>;
      };
    } catch (error) {
      alert("No upstream fields found. Connect a ChunkingNode first.");
      return;
    }

    const chunkArr = upstreamData.fields || [];
    if (chunkArr.length === 0) {
      alert("No chunk data found from upstream node.");
      return;
    }

    setLoading(true);
    const newResults: RetrievalResults = {};

    for (const method of methodItems) {
      const payload: any = {
        query,
        top_k: 10, // request up to 10 from server
        chunks: chunkArr.map((chunk) => ({
          text: chunk.text,
          docTitle:
            chunk.fill_history?.docTitle || chunk.metavars?.docTitle || "",
          chunkId: chunk.fill_history?.chunkId || chunk.metavars?.chunkId || "",
          chunkMethod:
            chunk.fill_history?.chunkMethod ||
            chunk.metavars?.chunkMethod ||
            "",
        })),
      };

      // Distinguish vector-based from keyword-based
      if (method.needsVector) {
        payload.library = method.vectorLib || "HuggingFace Transformers";
        payload.type = "vectorization";
        payload.method = method.baseMethod;
      } else {
        switch (method.baseMethod) {
          case "bm25":
            payload.library = "BM25";
            break;
          case "tfidf":
            payload.library = "TF-IDF";
            break;
          case "boolean":
            payload.library = "BooleanSearch";
            break;
          case "overlap":
            payload.library = "KeywordOverlap";
            break;
          default:
            payload.library = "KeywordBased";
        }
        payload.type = "retrieval";
      }

      try {
        const resp = await fetch("http://localhost:5000/retrieve", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!resp.ok) {
          const errData = await resp.json();
          throw new Error(errData.error || "Retrieval request failed");
        }
        const json = await resp.json();

        // method.methodName => e.g. "BM25"
        // If method.needsVector => method.vectorLib => e.g. "HuggingFace Transformers"
        const label =
          method.methodName +
          (method.needsVector && method.vectorLib
            ? ` (${method.vectorLib})`
            : "");

        // Force-slice to top 5
        json.retrieved = json.retrieved.slice(0, 5);

        newResults[method.key] = {
          label,
          retrieved: json.retrieved,
        };
      } catch (err: any) {
        console.error(`Error retrieving for ${method.methodName}:`, err);
        alert(`Retrieval failed: ${err.message}`);
      }
    }

    // Flatten & deduplicate for node output
    const deduped = prepareOutput(newResults);
    setResults(newResults);

    // Build LLMResponse array for inspector
    const newLLMResponses = buildLLMResponses(newResults);
    setJsonResponses(newLLMResponses);

    // Save node data
    setDataPropsForNode(id, {
      query,
      methods: methodItems,
      results: newResults,
      output: deduped.map((chunk) => ({
        text: chunk.text,
        similarity: chunk.similarity,
        chunkId: chunk.chunkId,
      })),
    });
    pingOutputNodes(id);
    setLoading(false);
  }, [
    query,
    methodItems,
    id,
    setDataPropsForNode,
    pingOutputNodes,
    buildLLMResponses,
    prepareOutput,
  ]);

  // Keep node data in sync
  useEffect(() => {
    setDataPropsForNode(id, { query, methods: methodItems, results });
  }, [id, query, methodItems, results, setDataPropsForNode]);

  return (
    <BaseNode
      nodeId={id}
      classNames="retrieval-node"
      style={{ backgroundColor: "rgba(255,255,255,0.9)" }}
    >
      {/* Input from chunking node */}
      <Handle type="target" position={Position.Left} id="fields" />

      <NodeLabel
        title={data.title || nodeDefaultTitle}
        nodeId={id}
        icon={nodeIcon}
        status={undefined}
        handleRunClick={runRetrieval}
        runButtonTooltip="Run Retrieval"
      />

      <div style={{ padding: 8, position: "relative" }}>
        <LoadingOverlay visible={loading} />
        <TextInput
          label="Search Query"
          placeholder="Enter your query..."
          icon={<IconSearch size={16} />}
          value={query}
          onChange={(e) => setQuery(e.currentTarget.value)}
          mb="sm"
        />

        <RetrievalMethodListContainer
          initMethodItems={methodItems}
          onItemsChange={handleMethodsChange}
        />
      </div>

      <InspectFooter
        onClick={() => inspectorModalRef.current?.trigger()}
        showDrawerButton={false}
        onDrawerClick={() => {
          // Do nothing
        }}
        isDrawerOpen={false}
      />

      <Handle type="source" position={Position.Right} id="output" />

      <React.Suspense fallback={null}>
        <LLMResponseInspectorModal
          ref={inspectorModalRef}
          jsonResponses={jsonResponses}
        />
      </React.Suspense>
    </BaseNode>
  );
};

export default RetrievalNode;
