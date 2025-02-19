import React, { useState, useEffect, useCallback, useRef, FC } from "react";
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
  Modal,
  Slider,
  Select,
  Textarea,
  Tooltip,
  SegmentedControl,
} from "@mantine/core";
import {
  IconSearch,
  IconPlus,
  IconSettings,
  IconTrash,
  IconChevronRight,
  IconInfoCircle,
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
  vectorStore?: {
    type: string; // e.g., "memory", "faiss"
    mode?: string; // "create" or "load"
    status: "disconnected" | "loading" | "connected" | "error";
  };
  settings: Record<string, any>;
}

const storeOptions = [
  { label: "💾 In-Memory", value: "memory" },
  { label: "🗄️ FAISS", value: "faiss" },
];

const vectorOptions = [
  { label: "🤗 HuggingFace Transformers", value: "HuggingFace Transformers" },
  { label: "🦄 OpenAI Embeddings", value: "OpenAI Embeddings" },
  { label: "🧠 Cohere Embeddings", value: "Cohere Embeddings" },
  { label: "💬 Sentence Transformers", value: "Sentence Transformers" },
];

// Groups of retrieval methods
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

// -------------------- Method Settings Modal --------------------
// This modal shows a list of settings. For numerical values we use sliders.
interface MethodSettingsModalProps {
  opened: boolean;
  initialSettings: Record<string, any>;
  onSave: (settings: Record<string, any>) => void;
  onClose: () => void;
  isVector: boolean;
  isOpenAI: boolean;
  isBM25: boolean;
  baseMethod: string;
  vectorStore?: { type: string; mode?: string };
}

const MethodSettingsModal: React.FC<MethodSettingsModalProps> = ({
  opened,
  initialSettings,
  onSave,
  onClose,
  isVector,
  isOpenAI,
  isBM25,
  baseMethod,
  vectorStore,
}) => {
  const [topK, setTopK] = useState<number>(initialSettings.top_k ?? 5);
  const [similarityThreshold, setSimilarityThreshold] = useState<number>(
    initialSettings.similarity_threshold ?? 0.7,
  );
  // For OpenAI methods:
  const [openaiModel, setOpenaiModel] = useState<string>(
    initialSettings.openai_model ?? "text-embedding-ada-002",
  );
  // BM25-specific settings:
  const [bm25K1, setBm25K1] = useState<number>(initialSettings.bm25_k1 ?? 1.5);
  const [bm25B, setBm25B] = useState<number>(initialSettings.bm25_b ?? 0.75);
  // For TF-IDF extra setting:
  const [maxFeatures, setMaxFeatures] = useState<number>(
    initialSettings.max_features ?? 500,
  );
  // For Boolean Search extra setting:
  const [requiredMatchCount, setRequiredMatchCount] = useState<number>(
    initialSettings.required_match_count ?? 1,
  );
  // For Keyword Overlap extra setting:
  const [normalizationFactor, setNormalizationFactor] = useState<number>(
    initialSettings.normalization_factor ?? 0.75,
  );

  // Add new state for FAISS settings
  const [faissMode, setFaissMode] = useState<string>(
    initialSettings?.faissMode || "create",
  );
  const [faissPath, setFaissPath] = useState<string>(
    initialSettings?.faissPath || "",
  );

  useEffect(() => {
    setTopK(initialSettings.top_k ?? 5);
    setSimilarityThreshold(initialSettings.similarity_threshold ?? 0.7);
    setOpenaiModel(initialSettings.openai_model ?? "text-embedding-ada-002");
    if (isBM25) {
      setBm25K1(initialSettings.bm25_k1 ?? 1.5);
      setBm25B(initialSettings.bm25_b ?? 0.75);
    }
    if (baseMethod === "tfidf") {
      setMaxFeatures(initialSettings.max_features ?? 500);
    }
    if (baseMethod === "boolean") {
      setRequiredMatchCount(initialSettings.required_match_count ?? 1);
    }
    if (baseMethod === "overlap") {
      setNormalizationFactor(initialSettings.normalization_factor ?? 0.75);
    }
    if (vectorStore?.type === "faiss") {
      setFaissMode(initialSettings?.faissMode || "create");
      setFaissPath(initialSettings?.faissPath || "");
    }
  }, [initialSettings, isBM25, baseMethod, vectorStore]);

  const handleSave = () => {
    const settings: Record<string, any> = {
      top_k: topK,
      similarity_threshold: similarityThreshold,
    };
    if (isOpenAI) {
      settings.openai_model = openaiModel;
    }
    if (isBM25) {
      settings.bm25_k1 = bm25K1;
      settings.bm25_b = bm25B;
    }
    if (baseMethod === "tfidf") {
      settings.max_features = maxFeatures;
    }
    if (baseMethod === "boolean") {
      settings.required_match_count = requiredMatchCount;
    }
    if (baseMethod === "overlap") {
      settings.normalization_factor = normalizationFactor;
    }
    if (vectorStore?.type === "faiss") {
      settings.faissMode = faissMode;
      settings.faissPath = faissPath;
    }
    onSave(settings);
    onClose();
  };

  return (
    <Modal opened={opened} onClose={onClose} title="Method Settings" size="sm">
      <Text weight={500} mb="xs">
        Adjust Settings:
      </Text>
      <Text size="sm" mb="xs">
        Top K (number of results):
        <Tooltip
          label="Sets the maximum number of results returned."
          color="gray"
          withArrow
        >
          <IconInfoCircle
            size={16}
            style={{ marginLeft: 4, verticalAlign: "middle" }}
          />
        </Tooltip>
      </Text>
      <Slider
        value={topK}
        onChange={setTopK}
        min={1}
        max={20}
        step={1}
        label={(val) => String(val)}
        mb="md"
      />
      <Text size="sm" mb="xs">
        Similarity Threshold:
        <Tooltip
          label="Determines the minimum similarity score for a chunk to be included."
          color="gray"
          withArrow
        >
          <IconInfoCircle
            size={16}
            style={{ marginLeft: 4, verticalAlign: "middle" }}
          />
        </Tooltip>
      </Text>
      <Slider
        value={similarityThreshold}
        onChange={setSimilarityThreshold}
        min={0}
        max={1}
        step={0.01}
        label={(val) => val.toFixed(2)}
        mb="md"
      />
      {isOpenAI && (
        <>
          <Text size="sm" mb="xs">
            OpenAI Embedding Model:
            <Tooltip
              label="Select the OpenAI model used for generating embeddings."
              color="gray"
              withArrow
            >
              <IconInfoCircle
                size={16}
                style={{ marginLeft: 4, verticalAlign: "middle" }}
              />
            </Tooltip>
          </Text>
          <Select
            data={[
              "text-embedding-ada-002",
              "text-embedding-3-small",
              "text-embedding-3-large",
            ]}
            value={openaiModel}
            onChange={(value) =>
              setOpenaiModel(value || "text-embedding-ada-002")
            }
            mb="md"
          />
        </>
      )}
      {isBM25 && (
        <>
          <Text size="sm" mb="xs">
            BM25 k1:
            <Tooltip
              label="Controls term frequency scaling in BM25. Typical value is 1.5."
              color="gray"
              withArrow
            >
              <IconInfoCircle
                size={16}
                style={{ marginLeft: 4, verticalAlign: "middle" }}
              />
            </Tooltip>
          </Text>
          <Slider
            value={bm25K1}
            onChange={setBm25K1}
            min={0.5}
            max={2.5}
            step={0.1}
            label={(val) => val.toFixed(1)}
            mb="md"
          />
          <Text size="sm" mb="xs">
            BM25 b:
            <Tooltip
              label="Controls document length normalization in BM25. Typical value is 0.75."
              color="gray"
              withArrow
            >
              <IconInfoCircle
                size={16}
                style={{ marginLeft: 4, verticalAlign: "middle" }}
              />
            </Tooltip>
          </Text>
          <Slider
            value={bm25B}
            onChange={setBm25B}
            min={0}
            max={1}
            step={0.05}
            label={(val) => val.toFixed(2)}
            mb="md"
          />
        </>
      )}
      {baseMethod === "tfidf" && (
        <>
          <Text size="sm" mb="xs">
            TF‑IDF Max Features:
            <Tooltip
              label="Set maximum features for the TF‑IDF vectorizer."
              color="gray"
              withArrow
            >
              <IconInfoCircle
                size={16}
                style={{ marginLeft: 4, verticalAlign: "middle" }}
              />
            </Tooltip>
          </Text>
          <Slider
            value={maxFeatures}
            onChange={setMaxFeatures}
            min={100}
            max={1000}
            step={50}
            label={(val) => String(val)}
            mb="md"
          />
        </>
      )}
      {baseMethod === "boolean" && (
        <>
          <Text size="sm" mb="xs">
            Boolean Required Match Count:
            <Tooltip
              label="Minimum number of matching tokens required."
              color="gray"
              withArrow
            >
              <IconInfoCircle
                size={16}
                style={{ marginLeft: 4, verticalAlign: "middle" }}
              />
            </Tooltip>
          </Text>
          <Slider
            value={requiredMatchCount}
            onChange={setRequiredMatchCount}
            min={1}
            max={5}
            step={1}
            label={(val) => String(val)}
            mb="md"
          />
        </>
      )}
      {baseMethod === "overlap" && (
        <>
          <Text size="sm" mb="xs">
            Keyword Overlap Normalization Factor:
            <Tooltip
              label="Factor applied to the overlap ratio."
              color="gray"
              withArrow
            >
              <IconInfoCircle
                size={16}
                style={{ marginLeft: 4, verticalAlign: "middle" }}
              />
            </Tooltip>
          </Text>
          <Slider
            value={normalizationFactor}
            onChange={setNormalizationFactor}
            min={0.5}
            max={1.0}
            step={0.05}
            label={(val) => val.toFixed(2)}
            mb="md"
          />
        </>
      )}
      {vectorStore?.type === "faiss" && (
        <>
          <Text size="sm" mb="xs">
            FAISS Index Mode:
            <Tooltip
              label="Choose whether to create a new index or load an existing one"
              color="gray"
              withArrow
            >
              <IconInfoCircle
                size={16}
                style={{ marginLeft: 4, verticalAlign: "middle" }}
              />
            </Tooltip>
          </Text>
          <SegmentedControl
            value={faissMode}
            onChange={setFaissMode}
            data={[
              { label: "Create New", value: "create" },
              { label: "Load Existing", value: "load" },
            ]}
            mb="md"
          />
          <Text size="sm" mb="xs">
            {faissMode === "create" ? "Index Directory:" : "Index File:"}
            <Tooltip
              label={
                faissMode === "create"
                  ? "Directory where the new index will be created"
                  : "Path to existing FAISS index file"
              }
              color="gray"
              withArrow
            >
              <IconInfoCircle
                size={16}
                style={{ marginLeft: 4, verticalAlign: "middle" }}
              />
            </Tooltip>
          </Text>
          <TextInput
            value={faissPath}
            onChange={(e) => setFaissPath(e.currentTarget.value)}
            placeholder={
              faissMode === "create"
                ? "Enter directory path..."
                : "Enter index file path..."
            }
            mb="md"
          />
        </>
      )}
      <Group position="right">
        <Button onClick={handleSave}>Save Settings</Button>
      </Group>
    </Modal>
  );
};

// -------------------- Retrieval Method List Container --------------------
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

  // For the settings modal
  const [settingsModalOpened, setSettingsModalOpened] = useState(false);
  const [currentMethodKey, setCurrentMethodKey] = useState<string | null>(null);
  const [currentMethodSettings, setCurrentMethodSettings] = useState<
    Record<string, any>
  >({});
  const [currentIsVector, setCurrentIsVector] = useState<boolean>(false);
  const [currentIsOpenAI, setCurrentIsOpenAI] = useState<boolean>(false);
  const [currentBaseMethod, setCurrentBaseMethod] = useState<string>("");

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
      method: Omit<
        RetrievalMethodSpec,
        "key" | "vectorLib" | "vectorStore" | "settings"
      >,
      chosenLibrary?: string,
      chosenStore?: string,
    ) => {
      const newItem: RetrievalMethodSpec = {
        ...method,
        key: uuid(),
        vectorLib: method.needsVector ? chosenLibrary : undefined,
        vectorStore: method.needsVector
          ? {
              type: chosenStore || "memory",
              status: "disconnected",
            }
          : undefined,
        settings: {},
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

  const openSettingsModal = (method: RetrievalMethodSpec) => {
    setCurrentMethodKey(method.key);
    setCurrentMethodSettings(method.settings || {});
    setCurrentIsVector(method.needsVector);
    setCurrentIsOpenAI(method.vectorLib === "OpenAI Embeddings");
    setCurrentBaseMethod(method.baseMethod); // <-- new
    setSettingsModalOpened(true);
  };

  const handleSaveSettings = (newSettings: Record<string, any>) => {
    const updated = methodItems.map((m) => {
      if (m.key === currentMethodKey) {
        return { ...m, settings: newSettings };
      }
      return m;
    });
    setMethodItems(updated);
    notifyItemsChanged(updated);
  };

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
                    return (
                      <Menu
                        key={item.baseMethod}
                        trigger="hover"
                        position="right-start"
                      >
                        <Menu.Target>
                          <Menu.Item
                            icon={<IconSettings size={14} />}
                            rightSection={<IconChevronRight size={14} />}
                          >
                            {item.methodName}
                          </Menu.Item>
                        </Menu.Target>

                        <Menu.Dropdown>
                          {/* Second level: Vector libraries */}
                          {vectorOptions.map((lib) => (
                            <Menu
                              key={lib.value}
                              trigger="hover"
                              position="right-start"
                            >
                              <Menu.Target>
                                <Menu.Item
                                  rightSection={<IconChevronRight size={14} />}
                                >
                                  {lib.label}
                                </Menu.Item>
                              </Menu.Target>

                              <Menu.Dropdown>
                                {/* Third level: Vectorstore options */}
                                {storeOptions.map((store) => (
                                  <Menu.Item
                                    key={store.value}
                                    onClick={() => {
                                      addMethod(item, lib.value, store.value);
                                      setMenuOpened(false);
                                    }}
                                  >
                                    {store.label}
                                  </Menu.Item>
                                ))}
                              </Menu.Dropdown>
                            </Menu>
                          ))}
                        </Menu.Dropdown>
                      </Menu>
                    );
                  }
                  // Keyword-based methods remain unchanged
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
        methodItems.map((method) => (
          <Group key={method.key} position="apart" mb="xs">
            <Group spacing="xs">
              {/* Only show status dot if vectorStore exists */}
              {method.vectorStore && (
                <div
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: "50%",
                    backgroundColor: getStatusDotColor(
                      method.vectorStore.status,
                    ),
                  }}
                />
              )}
              <Text size="sm">
                {method.methodName}
                {method.vectorLib && (
                  <>
                    {" | "}
                    <Text span inherit>
                      {method.vectorLib}
                    </Text>
                  </>
                )}
                {method.vectorStore && method.vectorStore.type !== "memory" && (
                  <>
                    {" | "}
                    <Text span inherit>
                      {method.vectorStore.type.toUpperCase()}
                    </Text>
                  </>
                )}
              </Text>
            </Group>
            <Group spacing={4}>
              <ActionIcon
                size="sm"
                variant="subtle"
                onClick={() => openSettingsModal(method)}
              >
                <IconSettings size={14} />
              </ActionIcon>
              <ActionIcon
                size="sm"
                variant="subtle"
                color="red"
                onClick={() => handleRemoveMethod(method.key)}
              >
                <IconTrash size={14} />
              </ActionIcon>
            </Group>
          </Group>
        ))
      )}

      <MethodSettingsModal
        opened={settingsModalOpened}
        initialSettings={currentMethodSettings}
        onSave={handleSaveSettings}
        onClose={() => setSettingsModalOpened(false)}
        isVector={currentIsVector}
        isOpenAI={currentIsOpenAI}
        isBM25={
          currentMethodSettings?.methodName === "BM25" ||
          methodItems.find((m) => m.key === currentMethodKey)?.baseMethod ===
            "bm25"
        }
        baseMethod={currentBaseMethod}
        vectorStore={
          methodItems.find((m) => m.key === currentMethodKey)?.vectorStore
        }
      />
    </div>
  );
};

// Add this helper function to get status dot color
const getStatusDotColor = (status?: string) => {
  switch (status) {
    case "connected":
      return "green";
    case "loading":
      return "yellow";
    case "error":
      return "red";
    case "disconnected":
    default:
      return "black";
  }
};

// -------------------- Main Retrieval Node --------------------
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
  const [jsonResponses, setJsonResponses] = useState<LLMResponse[]>([]);

  const inspectorModalRef = useRef<LLMResponseInspectorModalRef>(null);

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
      const methodLabel = methodObj.label;
      methodObj.retrieved.forEach((chunk, idx) => {
        // Always generate a new unique ID, regardless of the chunk's chunkId
        const cUid = uuid();

        arr.push({
          uid: cUid,
          prompt: `Retrieved by: ${methodLabel}`,
          vars: {
            similarity: chunk.similarity.toFixed(3),
            docTitle: chunk.docTitle || "Untitled",
            chunkId: chunk.chunkId || "",
            chunkMethod: chunk.chunkMethod || "",
          },
          responses: [`[Chunk ID: ${chunk.chunkId}]\n${chunk.text}`],
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

  const runRetrieval = useCallback(async () => {
    if (!query.trim()) {
      alert("Please enter a search query.");
      return;
    }
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
      const topKSetting = method.settings?.top_k ?? 5;
      const payload: any = {
        query,
        top_k: topKSetting,
        similarity_threshold: method.settings?.similarity_threshold ?? 0.7,
        // embedding_model is not passed here because we already have it in the main vector list
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

      if (method.needsVector) {
        payload.library = method.vectorLib || "HuggingFace Transformers";
        payload.type = "vectorization";
        payload.method = method.baseMethod;
        if (method.vectorLib === "OpenAI Embeddings") {
          payload.openai_model =
            method.settings?.openai_model ?? "text-embedding-3-small";
        }
        if (method.vectorStore && method.vectorStore.type !== "memory") {
          payload.vectorStore = {
            type: method.vectorStore.type,
            mode: method.settings?.faissMode || "create",
            path: method.settings?.faissPath || "",
          };
        }
      } else {
        switch (method.baseMethod) {
          case "bm25":
            payload.library = "BM25";
            payload.bm25_k1 = method.settings?.bm25_k1 ?? 1.5;
            payload.bm25_b = method.settings?.bm25_b ?? 0.75;
            break;
          case "tfidf":
            payload.library = "TF-IDF";
            payload.max_features = method.settings?.max_features ?? 500;
            break;
          case "boolean":
            payload.library = "Boolean Search";
            payload.required_match_count =
              method.settings?.required_match_count ?? 1;
            break;
          case "overlap":
            payload.library = "KeywordOverlap";
            payload.normalization_factor =
              method.settings?.normalization_factor ?? 0.75;
            break;
          default:
            payload.library = "KeywordBased";
        }
        payload.type = "retrieval";
      }
      try {
        const resp = await fetch("http://localhost:5001/retrieve", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!resp.ok) {
          const errData = await resp.json();
          throw new Error(errData.error || "Retrieval request failed");
        }
        const json = await resp.json();
        const label =
          method.methodName +
          (method.needsVector && method.vectorLib
            ? ` (${method.vectorLib})`
            : "");
        const topKSetting = method.settings?.top_k ?? 5;
        json.retrieved = json.retrieved.slice(0, topKSetting);
        newResults[method.key] = {
          label,
          retrieved: json.retrieved,
        };
      } catch (err: any) {
        console.error(`Error retrieving for ${method.methodName}:`, err);
        alert(`Retrieval failed: ${err.message}`);
      }
    }
    // Combine all retrieved chunks
    const allChunks: RetrievalChunk[] = [];
    Object.values(newResults).forEach((m) => {
      allChunks.push(...m.retrieved);
    });
    // New output preparation: sort, deduplicate and limit to the top 10 best chunks
    const sortedChunks = allChunks.sort((a, b) => b.similarity - a.similarity);
    const seen = new Set<string>();
    const dedupedChunks: RetrievalChunk[] = [];
    for (const chunk of sortedChunks) {
      const key = chunk.chunkId || chunk.text;
      if (!seen.has(key)) {
        dedupedChunks.push(chunk);
        seen.add(key);
      }
    }
    const outputChunks = dedupedChunks.slice(0, 10).map((chunk) => ({
      text: chunk.text,
      similarity: chunk.similarity,
      chunkId: chunk.chunkId || "No ID",
    }));

    setResults(newResults);
    const newLLMResponses = buildLLMResponses(newResults);
    setJsonResponses(newLLMResponses);
    setDataPropsForNode(id, {
      query,
      methods: methodItems,
      results: newResults,
      output: outputChunks,
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
  ]);

  useEffect(() => {
    setDataPropsForNode(id, { query, methods: methodItems, results });
  }, [id, query, methodItems, results, setDataPropsForNode]);

  return (
    <BaseNode
      nodeId={id}
      classNames="retrieval-node"
      style={{ backgroundColor: "rgba(255,255,255,0.9)" }}
    >
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
        <Textarea
          label="Search Query"
          placeholder="Enter your query..."
          className="prompt-field-fixed nodrag nowheel"
          autosize
          minRows={4}
          maxRows={12}
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
          // nothing here
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
