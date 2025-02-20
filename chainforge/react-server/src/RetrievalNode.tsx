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
  Popover,
  Card,
  Loader,
} from "@mantine/core";
import {
  IconPlus,
  IconSettings,
  IconTrash,
  IconChevronRight,
  IconInfoCircle,
  IconSquare,
  IconArrowNarrowRight,
} from "@tabler/icons-react";
import { v4 as uuid } from "uuid";
import emojidata from "@emoji-mart/data";
import Picker from "@emoji-mart/react";

import BaseNode from "./BaseNode";
import NodeLabel from "./NodeLabelComponent";
import useStore from "./store";
import InspectFooter from "./InspectFooter";
import LLMResponseInspectorModal, {
  LLMResponseInspectorModalRef,
} from "./LLMResponseInspectorModal";
import { LLMResponse } from "./backend/typing";
import { Status } from "./StatusIndicatorComponent";

// ---------- Default Emoji Mapping for Retrieval Methods ----------
const defaultMethodEmojis: { [key: string]: string } = {
  bm25: "📚",
  tfidf: "📈",
  boolean: "🧩",
  overlap: "🤝",
  cosine: "💡",
  sentenceEmbeddings: "🧠",
  customVector: "✨",
  clustered: "🗃️",
};

// ---------- Type Declarations ----------
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
  displayName?: string; // Modifiable display name (may include emoji)
}

const vectorOptions = [
  { label: "🤗 HuggingFace Transformers", value: "HuggingFace Transformers" },
  { label: "🦄 OpenAI Embeddings", value: "OpenAI Embeddings" },
  { label: "🧠 Cohere Embeddings", value: "Cohere Embeddings" },
  { label: "💬 Sentence Transformers", value: "Sentence Transformers" },
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

// -------------------- Method Settings Modal --------------------
// This modal shows a list of settings. For numerical values we use sliders.
interface MethodSettingsModalProps {
  opened: boolean;
  initialSettings: Record<string, any>;
  onSave: (newSettings: Record<string, any>) => void;
  onClose: () => void;
  isVector: boolean;
  isOpenAI?: boolean;
  isBM25?: boolean;
  baseMethod?: string;
}

// Updated MethodSettingsModal with BM25 sliders:
const MethodSettingsModal: FC<MethodSettingsModalProps> = ({
  opened,
  initialSettings,
  onSave,
  onClose,
  isVector,
  isOpenAI = false,
  isBM25 = false,
  baseMethod = "",
}) => {
  const [customName, setCustomName] = useState<string>(
    initialSettings.displayName || "",
  );
  const [emoji, setEmoji] = useState<string>(initialSettings.emoji || "🔍");
  const [emojiPickerOpen, setEmojiPickerOpen] = useState<boolean>(false);
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

  useEffect(() => {
    setCustomName(initialSettings.displayName || "");
    setEmoji(initialSettings.emoji || "🔍");
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
  }, [initialSettings, isBM25, baseMethod]);

  const handleEmojiSelect = useCallback((emojiData: any) => {
    setEmoji(emojiData.native);
    setEmojiPickerOpen(false);
  }, []);

  const handleSave = () => {
    const newSettings: Record<string, any> = {
      top_k: topK,
      similarity_threshold: similarityThreshold,
      displayName: customName,
      emoji,
    };
    if (isOpenAI) newSettings.openai_model = openaiModel;
    if (isBM25) {
      newSettings.bm25_k1 = bm25K1;
      newSettings.bm25_b = bm25B;
    }
    if (baseMethod === "tfidf") newSettings.max_features = maxFeatures;
    if (baseMethod === "boolean")
      newSettings.required_match_count = requiredMatchCount;
    if (baseMethod === "overlap")
      newSettings.normalization_factor = normalizationFactor;
    onSave(newSettings);
    onClose();
  };

  return (
    <Modal
      opened={opened}
      onClose={onClose}
      title={
        <div style={{ display: "flex", alignItems: "center" }}>
          <Popover
            width={250}
            position="bottom"
            withArrow
            shadow="md"
            opened={emojiPickerOpen}
            onChange={setEmojiPickerOpen}
          >
            <Popover.Target>
              <Button
                variant="subtle"
                compact
                style={{ fontSize: "16pt" }}
                onClick={() => setEmojiPickerOpen((o) => !o)}
              >
                {emoji}
              </Button>
            </Popover.Target>
            <Popover.Dropdown>
              <Picker
                data={emojidata}
                onEmojiSelect={handleEmojiSelect}
                theme="light"
              />
            </Popover.Dropdown>
          </Popover>
          <TextInput
            value={customName}
            onChange={(e) => setCustomName(e.target.value)}
            placeholder="Method Name"
            style={{ marginLeft: 8, fontSize: "16pt", flex: 1 }}
          />
        </div>
      }
      centered
    >
      <Text weight={500} mb="xs">
        Adjust Settings:
      </Text>
      <Text size="xs" mb="xs">
        Top K (number of results):
        <Tooltip
          label="Sets the maximum number of results returned."
          color="gray"
          withArrow
        >
          <IconInfoCircle
            size={14}
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
        size="xs"
      />
      <Text size="xs" mb="xs">
        Similarity Threshold:
        <Tooltip
          label="Determines the minimum similarity score for a chunk to be included."
          color="gray"
          withArrow
        >
          <IconInfoCircle
            size={14}
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
        size="xs"
      />
      {isOpenAI && (
        <>
          <Text size="xs" mb="xs">
            OpenAI Embedding Model:
            <Tooltip
              label="Select the OpenAI model used for generating embeddings."
              color="gray"
              withArrow
            >
              <IconInfoCircle
                size={14}
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
            size="xs"
          />
        </>
      )}
      {isBM25 && (
        <>
          <Text size="xs" mb="xs">
            BM25 k1:
            <Tooltip
              label="Controls term frequency scaling in BM25. Typical value is 1.5."
              color="gray"
              withArrow
            >
              <IconInfoCircle
                size={14}
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
            size="xs"
          />
          <Text size="xs" mb="xs">
            BM25 b:
            <Tooltip
              label="Controls document length normalization in BM25. Typical value is 0.75."
              color="gray"
              withArrow
            >
              <IconInfoCircle
                size={14}
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
            size="xs"
          />
        </>
      )}
      {baseMethod === "tfidf" && (
        <>
          <Text size="xs" mb="xs">
            TF‑IDF Max Features:
            <Tooltip
              label="Set maximum features for the TF‑IDF vectorizer."
              color="gray"
              withArrow
            >
              <IconInfoCircle
                size={14}
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
            size="xs"
          />
        </>
      )}
      {baseMethod === "boolean" && (
        <>
          <Text size="xs" mb="xs">
            Boolean Required Match Count:
            <Tooltip
              label="Minimum number of matching tokens required."
              color="gray"
              withArrow
            >
              <IconInfoCircle
                size={14}
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
            size="xs"
          />
        </>
      )}
      {baseMethod === "overlap" && (
        <>
          <Text size="xs" mb="xs">
            Keyword Overlap Normalization Factor:
            <Tooltip
              label="Factor applied to the overlap ratio."
              color="gray"
              withArrow
            >
              <IconInfoCircle
                size={14}
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
            size="xs"
          />
        </>
      )}
      <Group position="right">
        <Button onClick={handleSave} size="xs">
          Save Settings
        </Button>
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
  loadingMethods?: { [key: string]: boolean }; // Added prop
}

const RetrievalMethodListContainer: React.FC<
  RetrievalMethodListContainerProps
> = ({ initMethodItems = [], onItemsChange, loadingMethods }) => {
  const [methodItems, setMethodItems] =
    useState<RetrievalMethodSpec[]>(initMethodItems);
  const [menuOpened, setMenuOpened] = useState(false);

  // Settings modal state
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
        "key" | "vectorLib" | "settings" | "displayName"
      >,
      chosenLibrary?: string,
    ) => {
      const baseName =
        method.methodName + (chosenLibrary ? ` (${chosenLibrary})` : "");
      const existingCount = methodItems.filter((m) =>
        m.displayName ? m.displayName.startsWith(baseName) : false,
      ).length;
      const displayName = existingCount
        ? `${baseName} (${existingCount + 1})`
        : baseName;
      const newItem: RetrievalMethodSpec = {
        ...method,
        key: uuid(),
        vectorLib: method.needsVector ? chosenLibrary : undefined,
        settings: {},
        displayName,
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

  const openSettingsModal = useCallback((method: RetrievalMethodSpec) => {
    setCurrentMethodKey(method.key);
    setCurrentMethodSettings(method.settings || {});
    setCurrentIsVector(method.needsVector);
    setCurrentIsOpenAI(method.vectorLib === "OpenAI Embeddings");
    setCurrentBaseMethod(method.baseMethod);
    setSettingsModalOpened(true);
  }, []);

  const handleSaveSettings = useCallback(
    (newSettings: Record<string, any>) => {
      if (!currentMethodKey) return;
      const updated = methodItems.map((m) => {
        if (m.key === currentMethodKey) {
          const updatedDisplayName =
            newSettings.displayName !== ""
              ? newSettings.displayName
              : m.displayName;
          const updatedEmoji =
            newSettings.emoji !== ""
              ? newSettings.emoji
              : m.settings?.emoji || defaultMethodEmojis[m.baseMethod];
          return {
            ...m,
            settings: { ...newSettings, emoji: updatedEmoji },
            displayName: updatedDisplayName,
          };
        }
        return m;
      });
      setMethodItems(updated);
      notifyItemsChanged(updated);
    },
    [currentMethodKey, methodItems, notifyItemsChanged],
  );

  return (
    <div style={{ border: "1px dashed #ccc", borderRadius: 6, padding: 4 }}>
      <Group position="apart" mb="xs">
        <Text size="xs" weight={500}>
          Selected Retrieval Methods
        </Text>
        <Menu
          position="bottom-end"
          withinPortal
          closeOnClickOutside
          onOpen={() => setMenuOpened(true)}
          onClose={() => setMenuOpened(false)}
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
                            icon={
                              <span style={{ fontSize: "12px" }}>
                                {defaultMethodEmojis[item.baseMethod] || "🔍"}
                              </span>
                            }
                            rightSection={<IconChevronRight size={14} />}
                          >
                            <Text size="xs">{item.methodName}</Text>
                          </Menu.Item>
                        </Menu.Target>
                        <Menu.Dropdown>
                          {vectorOptions.map((lib) => (
                            <Menu.Item
                              key={lib.value}
                              onClick={() => {
                                addMethod(item, lib.value);
                                setMenuOpened(false);
                              }}
                            >
                              <Text size="xs">{lib.label}</Text>
                            </Menu.Item>
                          ))}
                        </Menu.Dropdown>
                      </Menu>
                    );
                  }
                  return (
                    <Menu.Item
                      key={item.baseMethod}
                      icon={
                        <span style={{ fontSize: "12px" }}>
                          {defaultMethodEmojis[item.baseMethod] || "🔍"}
                        </span>
                      }
                      onClick={() => {
                        addMethod(item);
                        setMenuOpened(false);
                      }}
                    >
                      <Text size="xs">{item.methodName}</Text>
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
          <Card
            key={item.key}
            shadow="sm"
            withBorder
            style={{ fontSize: "10px", padding: "2px", marginTop: "2px" }}
          >
            <Group position="apart" align="center">
              <div>
                <Text size="xs" weight={600}>
                  {item.settings && item.settings.emoji
                    ? item.settings.emoji
                    : defaultMethodEmojis[item.baseMethod] || "🔍"}{" "}
                  {item.displayName
                    ? item.displayName
                    : `${item.methodName}${item.vectorLib ? ` (${item.vectorLib})` : ""}`}{" "}
                  {loadingMethods && loadingMethods[item.key] && (
                    <Loader size="xs" color="blue" style={{ marginLeft: 4 }} />
                  )}
                </Text>
              </div>
              <Group spacing="xs">
                <ActionIcon
                  variant="subtle"
                  onClick={() => openSettingsModal(item)}
                  title="Settings"
                >
                  <IconSettings size={16} />
                </ActionIcon>
                <ActionIcon
                  color="red"
                  variant="subtle"
                  onClick={() => handleRemoveMethod(item.key)}
                  title="Remove"
                >
                  <IconTrash size={16} />
                </ActionIcon>
              </Group>
            </Group>
          </Card>
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
      />
    </div>
  );
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

  const [status, setStatus] = useState<Status>(Status.NONE);
  const [queries, setQueries] = useState<string[]>(data.query ? [data.query] : [""]);
  const [methodItems, setMethodItems] = useState<RetrievalMethodSpec[]>(
    data.methods || [],
  );
  const [results, setResults] = useState<RetrievalResults>(data.results || {});
  const [loading, setLoading] = useState(false);
  const [jsonResponses, setJsonResponses] = useState<LLMResponse[]>([]);
  const [loadingMethods, setLoadingMethods] = useState<{
    [key: string]: boolean;
  }>({});
  const [cancelId, setCancelId] = useState(Date.now());
  const refreshCancelId = useCallback(() => setCancelId(Date.now()), []);

  const inspectorModalRef = useRef<LLMResponseInspectorModalRef>(null);

  useEffect(() => {
    if (data.refresh) {
      setDataPropsForNode(id, { refresh: false, results: {} });
      setResults({});
      setJsonResponses([]);
      setStatus(Status.NONE);
    }
  }, [data.refresh, id, setDataPropsForNode]);

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
    setJsonResponses([]);
  }, []);

  const runRetrieval = useCallback(async () => {
    if (!queries.some(q => q.trim())) {
      alert("Please enter at least one query.");
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
    setStatus(Status.LOADING);
    setLoading(true);
    const newResults: { [key: string]: { label: string; retrieved: any[] } } = {};
    const newLoading: { [key: string]: boolean } = {};
    methodItems.forEach((m) => {
      newLoading[m.key] = true;
      // initialize results for each method
      newResults[m.key] = { label: m.displayName || m.methodName, retrieved: [] };
    });
    setLoadingMethods(newLoading);

    for (const method of methodItems) {
      try {
        const topKSetting = method.settings?.top_k ?? 5;
        // For each non-empty query, call the API and append results.
        for (const singleQuery of queries) {
          if (!singleQuery.trim()) continue;
          const payload: any = {
            query: singleQuery,
            top_k: topKSetting,
            similarity_threshold: method.settings?.similarity_threshold ?? 0.7,
            chunks: chunkArr.map((chunk) => ({
              text: chunk.text,
              docTitle:
                chunk.fill_history?.docTitle ||
                chunk.metavars?.docTitle ||
                "",
              chunkId:
                chunk.fill_history?.chunkId ||
                chunk.metavars?.chunkId ||
                "",
            })),
            custom_method_key: method.key,
          };

          if (method.needsVector) {
            payload.library = method.vectorLib || "HuggingFace Transformers";
            payload.type = "vectorization";
            payload.method = method.baseMethod;
            if (method.vectorLib === "OpenAI Embeddings") {
              payload.openai_model =
                method.settings?.openai_model ?? "text-embedding-ada-002";
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
            const baseLabel = method.displayName || method.methodName;
            let label = baseLabel;
            if (method.needsVector && method.vectorLib) {
              if (
                !baseLabel
                  .toLowerCase()
                  .endsWith(`(${method.vectorLib.toLowerCase()})`)
              ) {
                label = `${baseLabel} (${method.vectorLib})`;
              }
            }
            // Prepend query text to each retrieved chunk
            json.retrieved.forEach((chunk: any) => {
              newResults[method.key].retrieved.push({
                ...chunk,
                text: `${chunk.text}`,
              });
            });
            newResults[method.key].label = label;
          } catch (err: any) {
            console.error(`Error retrieving for ${method.methodName}:`, err);
            alert(`Retrieval failed for query "${singleQuery}": ${err.message}`);
          }
        }
      } catch (err: any) {
        console.error(`Error retrieving for ${method.methodName}:`, err);
        alert(`Retrieval failed: ${err.message}`);
      } finally {
        setLoadingMethods((prev) => ({ ...prev, [method.key]: false }));
      }
    }
    // Build output chunks from combined results
    const outputChunks: Array<{
      text: string;
      similarity: number;
      chunkId: string;
    }> = [];
    Object.entries(newResults).forEach(([_, methodResult]) => {
      methodResult.retrieved.forEach((chunk) => {
        outputChunks.push({
          text: `Chunk ID: ${chunk.chunkId || "No ID"}\nRetrieval Method: ${methodResult.label}\n\n${chunk.text}`,
          similarity: chunk.similarity,
          chunkId: chunk.chunkId || "No ID",
        });
      });
    });
    setResults(newResults);
    const buildLLMResponses = (
      resultsData: { [key: string]: { label: string; retrieved: any[] } },
    ): LLMResponse[] => {
      const arr: LLMResponse[] = [];
      Object.entries(resultsData).forEach(([methodKey, methodObj]) => {
        const methodItem = methodItems.find((m) => m.key === methodKey);
        const baseLabel =
          methodItem?.displayName ||
          methodItem?.methodName ||
          "Retrieval Method";
        let finalLabel = baseLabel;
        if (methodItem?.needsVector && methodItem?.vectorLib) {
          if (
            !baseLabel
              .toLowerCase()
              .endsWith(`(${methodItem.vectorLib.toLowerCase()})`)
          ) {
            finalLabel = `${baseLabel} (${methodItem.vectorLib})`;
          }
        }
        methodObj.retrieved.forEach((chunk) => {
          const cUid = uuid();
          arr.push({
            uid: cUid,
            prompt: `Retrieved by: ${finalLabel}`,
            vars: {
              similarity: chunk.similarity.toFixed(3),
              docTitle: chunk.docTitle || "Untitled",
              chunkId: chunk.chunkId || "",
              chunkMethod: finalLabel,
            },
            responses: [`[Chunk ID: ${chunk.chunkId}]\n${chunk.text}`],
            llm: finalLabel,
            metavars: {
              retrievalMethod: finalLabel,
              docTitle: chunk.docTitle,
              chunkId: chunk.chunkId,
              chunkMethod: finalLabel,
              similarity: chunk.similarity,
            },
          });
        });
      });
      return arr;
    };
    const newLLMResponses = buildLLMResponses(newResults);
    setJsonResponses(newLLMResponses);
    setDataPropsForNode(id, {
      queries, // update to store multiple queries
      methods: methodItems,
      results: newResults,
      output: outputChunks,
    });
    pingOutputNodes(id);
    setLoading(false);
    setStatus(Status.READY);
  }, [queries, methodItems, id, setDataPropsForNode, pingOutputNodes]);

  const handleStopClick = useCallback(() => {
    setStatus(Status.NONE);
    setLoading(false);
    refreshCancelId();
  }, [refreshCancelId]);

  useEffect(() => {
    setDataPropsForNode(id, { query: queries[0], methods: methodItems, results });
  }, [id, queries, methodItems, results, setDataPropsForNode]);

  return (
    <BaseNode
      nodeId={id}
      classNames="retrieval-node"
      style={{ width: "450px", backgroundColor: "rgba(255,255,255,0.9)" }}
    >
      <Handle type="target" position={Position.Left} id="fields" />
      <NodeLabel
        title={data.title || nodeDefaultTitle}
        nodeId={id}
        icon={nodeIcon}
        status={status} // Pass the status state
        handleRunClick={
          status === Status.LOADING ? handleStopClick : runRetrieval
        }
        runButtonTooltip={
          status === Status.LOADING ? "Stop Retrieval" : "Run Retrieval"
        }
      />
      <div style={{ padding: 8, position: "relative" }}>
        {queries.map((q, idx) => (
          <div
            key={idx}
            style={{
              display: "flex",
              alignItems: "center",
              marginBottom: 4,
            }}
          >
            <Textarea
              className="prompt-field-fixed nodrag nowheel"
              label={idx === 0 ? "Search Queries" : undefined}
              placeholder={`Enter query ${idx + 1}...`}
              value={q}
              minRows={4}
              maxRows={12}
              autosize
              style={{ flex: 1 }}
              onChange={(e) => {
                const newQueries = [...queries];
                newQueries[idx] = e.currentTarget.value;
                setQueries(newQueries);
              }}
            />
            {queries.length > 1 && (
              <ActionIcon
                onClick={() =>
                  setQueries(queries.filter((_, index) => index !== idx))
                }
                style={{ marginLeft: 4 }}
                title="Delete this query"
              >
                <span style={{ fontSize: "16px", fontWeight: "bold" }}>X</span>
              </ActionIcon>
            )}
            {idx === queries.length - 1 && (
              <ActionIcon
                onClick={() => setQueries([...queries, ""])}
                style={{ marginLeft: 4 }}
                title="Add another query"
              >
                <IconPlus size={16} />
              </ActionIcon>
            )}
          </div>
        ))}
        <RetrievalMethodListContainer
          initMethodItems={methodItems}
          onItemsChange={handleMethodsChange}
          loadingMethods={loadingMethods}
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
