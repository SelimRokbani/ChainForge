import React, { useState, useEffect, useCallback, useRef } from "react";
import { Handle, Position } from "reactflow";
import {
  Group,
  Text,
  Button,
  Menu,
  Divider,
  ActionIcon,
  Card,
  Modal,
  Slider,
  Select,
  TextInput,
  Popover,
  Tooltip,
  Loader,
  Center,
} from "@mantine/core";
import {
  IconPlus,
  IconSettings,
  IconTrash,
  IconChevronRight,
  IconInfoCircle,
} from "@tabler/icons-react";
import { v4 as uuid } from "uuid";
import emojidata from "@emoji-mart/data";
import Picker from "@emoji-mart/react";
import { useDisclosure } from "@mantine/hooks";

import BaseNode from "./BaseNode";
import NodeLabel from "./NodeLabelComponent";
import useStore from "./store";
import InspectFooter from "./InspectFooter";
import LLMResponseInspectorModal, {
  LLMResponseInspectorModalRef,
} from "./LLMResponseInspectorModal";
import { LLMResponse, TemplateVarInfo } from "./backend/typing";
import { Status } from "./StatusIndicatorComponent";
import TemplateHooks, {
  extractBracketedSubstrings,
} from "./TemplateHooksComponent";

// ### Type Definitions
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

// ### Constants
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

// ### PromptListPopover Component (Simplified Placeholder)
interface PromptListPopoverProps {
  promptInfos: string[];
  onHover: () => void;
  onClick: () => void;
}

const PromptListPopover: React.FC<PromptListPopoverProps> = ({
  promptInfos,
  onHover,
  onClick,
}) => {
  const [opened, { close, open }] = useDisclosure(false);

  const _onHover = useCallback(() => {
    onHover();
    open();
  }, [onHover, open]);

  return (
    <Popover
      position="right-start"
      withArrow
      withinPortal
      shadow="rgb(38, 57, 77) 0px 10px 30px -14px"
      opened={opened}
      styles={{
        dropdown: {
          maxHeight: "500px",
          maxWidth: "400px",
          overflowY: "auto",
          backgroundColor: "#fff",
        },
      }}
    >
      <Popover.Target>
        <Tooltip label="Click to view query previews" withArrow>
          <Button
            className="custom-button"
            onMouseEnter={_onHover}
            onMouseLeave={close}
            onClick={onClick}
            style={{ border: "none", background: "none", padding: 0 }}
          >
            <IconInfoCircle
              size="12pt"
              color="gray"
              style={{ marginBottom: "-4px" }}
            />
          </Button>
        </Tooltip>
      </Popover.Target>
      <Popover.Dropdown sx={{ pointerEvents: "none" }}>
        <Center>
          <Text size="xs" fw={500} color="#666">
            Query Previews ({promptInfos?.length} total)
          </Text>
        </Center>
        {promptInfos.map((info, idx) => (
          <Text
            key={idx}
            size="xs"
            style={{ whiteSpace: "pre-wrap", margin: "4px 0" }}
          >
            {info}
          </Text>
        ))}
      </Popover.Dropdown>
    </Popover>
  );
};

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

const MethodSettingsModal: React.FC<MethodSettingsModalProps> = ({
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
  loadingMethods?: { [key: string]: boolean };
}

const RetrievalMethodListContainer: React.FC<
  RetrievalMethodListContainerProps
> = ({ initMethodItems = [], onItemsChange, loadingMethods }) => {
  const [methodItems, setMethodItems] =
    useState<RetrievalMethodSpec[]>(initMethodItems);
  const [menuOpened, setMenuOpened] = useState(false);
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
  const pullInputData = useStore((s) => s.pullInputData);

  const [status, setStatus] = useState<Status>(Status.NONE);
  const [queryText, setQueryText] = useState<string>(data.query || "");
  const [templateVars, setTemplateVars] = useState<string[]>([]);
  const [methodItems, setMethodItems] = useState<RetrievalMethodSpec[]>(
    data.methods || [],
  );
  const [results, setResults] = useState<RetrievalResults>(data.results || {});
  const [loading, setLoading] = useState(false);
  const [jsonResponses, setJsonResponses] = useState<LLMResponse[]>([]);
  const [loadingMethods, setLoadingMethods] = useState<{
    [key: string]: boolean;
  }>({});
  const [queryPreviews, setQueryPreviews] = useState<string[]>([]);

  const inspectorModalRef = useRef<LLMResponseInspectorModalRef>(null);
  const textAreaRef = useRef<HTMLTextAreaElement | null>(null);
  const [hooksY, setHooksY] = useState(138);
  const [infoModalOpened, { open: openInfoModal, close: closeInfoModal }] =
    useDisclosure(false);

  useEffect(() => {
    if (data.refresh) {
      setDataPropsForNode(id, { refresh: false, results: {} });
      setResults({});
      setJsonResponses([]);
      setStatus(Status.NONE);
    }
  }, [data.refresh, id, setDataPropsForNode]);

  const refreshTemplateHooks = useCallback(
    (text: string) => {
      const foundTemplateVars = new Set(extractBracketedSubstrings(text));
      if (
        foundTemplateVars.size !== templateVars.length ||
        ![...foundTemplateVars].every((v) => templateVars.includes(v))
      ) {
        setTemplateVars(Array.from(foundTemplateVars));
      }
    },
    [templateVars],
  );

  const handleQueryChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = event.target.value;
    setQueryText(value);
    refreshTemplateHooks(value);
    generateQueryPreviews(value); // Update previews on change
  };

  const setTextAreaRef = useCallback((elem: HTMLTextAreaElement | null) => {
    if (!elem || !window.ResizeObserver) return;
    if (!textAreaRef.current) {
      let pastHooksY = 138;
      const observer = new window.ResizeObserver(() => {
        if (!textAreaRef.current) return;
        const newHooksY = textAreaRef.current.clientHeight + 68;
        if (pastHooksY !== newHooksY) {
          setHooksY(newHooksY);
          pastHooksY = newHooksY;
        }
      });
      observer.observe(elem);
      textAreaRef.current = elem;
    }
  }, []);

  const generateQueryPreviews = useCallback(
    (text: string) => {
      let pulledData: { [key: string]: any[] } = {};
      try {
        pulledData = pullInputData(templateVars.concat(["fields"]), id);
      } catch (error) {
        console.error("Error pulling input data for preview:", error);
      }

      const baseQueries = text.split("\n").filter((q) => q.trim());
      let maxRows = 1;
      if (templateVars.length > 0) {
        maxRows = Math.max(
          ...templateVars.map((varName) =>
            pulledData[varName] ? pulledData[varName].length : 0,
          ),
          1,
        );
      }

      const previews: string[] = [];
      for (let rowIdx = 0; rowIdx < maxRows && previews.length < 10; rowIdx++) {
        baseQueries.forEach((baseQuery) => {
          let resolvedQuery = baseQuery;
          templateVars.forEach((varName) => {
            if (pulledData[varName] && pulledData[varName].length > rowIdx) {
              resolvedQuery = resolvedQuery.replace(
                `{${varName}}`,
                pulledData[varName][rowIdx]?.text ||
                  pulledData[varName][rowIdx] ||
                  varName,
              );
            } else {
              resolvedQuery = resolvedQuery.replace(`{${varName}}`, varName);
            }
          });
          if (resolvedQuery.trim() && previews.length < 10) {
            previews.push(resolvedQuery);
          }
        });
      }
      setQueryPreviews(previews);
    },
    [templateVars, pullInputData, id],
  );

  const buildLLMResponses = (resultsData: {
    [key: string]: { label: string; retrieved: any[] };
  }): LLMResponse[] => {
    const arr: LLMResponse[] = [];
    Object.entries(resultsData).forEach(([methodKey, methodObj]) => {
      const methodItem = methodItems.find((m) => m.key === methodKey);
      const baseLabel =
        methodItem?.displayName || methodItem?.methodName || "Retrieval Method";
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
        // Use the chunk's stored chunking method display if present.
        const chunkMethodDisplay =
          chunk.chunkMethod ||
          (chunk.fill_history && chunk.fill_history.chunkMethod) ||
          finalLabel;
        arr.push({
          uid: cUid,
          prompt: `Retrieved by chunk method: ${chunkMethodDisplay}`,
          vars: {
            similarity: chunk.similarity.toFixed(3),
            docTitle: chunk.docTitle || "Untitled",
            chunkId: chunk.chunkId || "",
            // Use the chunking method display name here.
            chunkMethod: chunkMethodDisplay,
          },
          responses: [` ${chunk.text}`],
          llm: chunkMethodDisplay,
          metavars: {
            retrievalMethod: chunkMethodDisplay,
            docTitle: chunk.docTitle,
            chunkId: chunk.chunkId,
            chunkMethod: chunkMethodDisplay,
            similarity: chunk.similarity,
          },
        });
      });
    });
    return arr;
  };

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
    if (!queryText.trim() && templateVars.length === 0) {
      alert("Please enter at least one query or connect a data source.");
      return;
    }

    let pulledData: { [key: string]: any[] } = {};
    try {
      pulledData = pullInputData(templateVars.concat(["fields"]), id);
    } catch (error) {
      console.error("Error pulling input data:", error);
    }

    const baseQueries = queryText.split("\n").filter((q) => q.trim());
    const chunkArr = pulledData.fields || [];

    if (chunkArr.length === 0 && !templateVars.length) {
      alert("No chunk data found from upstream node or template variables.");
      return;
    }

    let maxRows = 1;
    if (templateVars.length > 0) {
      maxRows = Math.max(
        ...templateVars.map((varName) =>
          pulledData[varName] ? pulledData[varName].length : 0,
        ),
        1,
      );
    }

    const queries: string[] = [];
    for (let rowIdx = 0; rowIdx < maxRows; rowIdx++) {
      baseQueries.forEach((baseQuery) => {
        let resolvedQuery = baseQuery;
        templateVars.forEach((varName) => {
          if (pulledData[varName] && pulledData[varName].length > rowIdx) {
            resolvedQuery = resolvedQuery.replace(
              `{${varName}}`,
              pulledData[varName][rowIdx]?.text ||
                pulledData[varName][rowIdx] ||
                varName,
            );
          } else {
            resolvedQuery = resolvedQuery.replace(`{${varName}}`, varName);
          }
        });
        if (resolvedQuery.trim()) {
          queries.push(resolvedQuery);
        }
      });
    }

    if (queries.length === 0) {
      alert("No valid queries generated from input data.");
      return;
    }

    setStatus(Status.LOADING);
    setLoading(true);
    const newResults: { [key: string]: { label: string; retrieved: any[] } } =
      {};
    const newLoading: { [key: string]: boolean } = {};
    methodItems.forEach((m) => {
      newLoading[m.key] = true;
      newResults[m.key] = {
        label: m.displayName || m.methodName,
        retrieved: [],
      };
    });
    setLoadingMethods(newLoading);

    // Store results grouped by method and query
    const groupedResults: {
      [methodKey: string]: {
        [query: string]: any[];
      };
    } = {};

    // Initialize the grouped structure
    methodItems.forEach((method) => {
      groupedResults[method.key] = {};
      queries.forEach((q) => {
        groupedResults[method.key][q] = [];
      });
    });

    for (const method of methodItems) {
      try {
        const topKSetting = method.settings?.top_k ?? 5;
        for (const resolvedQuery of queries) {
          const payload: any = {
            query: resolvedQuery,
            top_k: topKSetting,
            similarity_threshold: method.settings?.similarity_threshold ?? 0.7,
            chunks: chunkArr.map((chunk) => ({
              text: chunk.text,
              docTitle:
                chunk.fill_history?.docTitle || chunk.metavars?.docTitle || "",
              chunkId:
                chunk.fill_history?.chunkId || chunk.metavars?.chunkId || "",
              chunkMethod: chunk.metavars?.chunkMethod || "",
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

          const seenTexts = new Set<string>();
          const uniqueChunks = json.retrieved.filter((chunk: any) => {
            if (seenTexts.has(chunk.text)) return false;
            seenTexts.add(chunk.text);
            return true;
          });

          // Add chunks to the grouped structure
          uniqueChunks.forEach((chunk: any) => {
            groupedResults[method.key][resolvedQuery].push({
              ...chunk,
              text: `${chunk.text}`,
              query: resolvedQuery,
            });
          });
        }

        // Set the label
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

        // Flatten the grouped results for this method and add to newResults
        newResults[method.key].label = label;
        Object.values(groupedResults[method.key]).forEach((chunks) => {
          newResults[method.key].retrieved.push(...chunks);
        });
      } catch (err: any) {
        console.error(`Error retrieving for ${method.methodName}:`, err);
        alert(`Retrieval failed: ${err.message}`);
      } finally {
        setLoadingMethods((prev) => ({ ...prev, [method.key]: false }));
      }
    }

    // Generate output chunks maintaining the grouping
    const outputChunks: TemplateVarInfo[] = [];

    Object.entries(groupedResults).forEach(([methodKey, methodQueries]) => {
      const method = methodItems.find((m) => m.key === methodKey);
      if (!method) return;

      const baseLabel = method.displayName || method.methodName;
      let methodLabel = baseLabel;
      if (method.needsVector && method.vectorLib) {
        if (
          !baseLabel
            .toLowerCase()
            .endsWith(`(${method.vectorLib.toLowerCase()})`)
        ) {
          methodLabel = `${baseLabel} (${method.vectorLib})`;
        }
      }

      // For each query, create chunks with a group marker
      Object.entries(methodQueries).forEach(([query, chunks]) => {
        if (chunks.length === 0) return;

        // Add all chunks for this method-query pair
        chunks.forEach((chunk) => {
          outputChunks.push({
            text: chunk.text,
            fill_history: {},
            metavars: {
              retrievalMethod: methodLabel,
              query: query,
              similarity: chunk.similarity.toString(),
              docTitle: chunk.docTitle || "",
              chunkId: chunk.chunkId || "",
              chunkMethod: chunk.chunkMethod || "",
              methodGroup: methodKey,
              queryGroup: query,
            },
            uid: uuid(),
          });
        });
      });
    });

    setResults(newResults);
    const newLLMResponses = buildLLMResponses(newResults);
    setJsonResponses(newLLMResponses);
    setDataPropsForNode(id, {
      query: queryText,
      methods: methodItems,
      results: newResults,
      output: outputChunks,
    });
    pingOutputNodes(id);
    setLoading(false);
    setStatus(Status.READY);
  }, [
    queryText,
    methodItems,
    id,
    templateVars,
    setDataPropsForNode,
    pingOutputNodes,
    pullInputData,
  ]);

  const handleStopClick = useCallback(() => {
    setStatus(Status.NONE);
    setLoading(false);
  }, []);

  const handleRunClick = useCallback(() => {
    runRetrieval();
  }, [runRetrieval]);

  const handleRunHover = useCallback(() => {
    generateQueryPreviews(queryText);
  }, [queryText, generateQueryPreviews]);

  const handlePreviewHover = useCallback(() => {
    generateQueryPreviews(queryText);
  }, [queryText, generateQueryPreviews]);

  useEffect(() => {
    setDataPropsForNode(id, {
      query: queryText,
      methods: methodItems,
      results,
    });
    refreshTemplateHooks(queryText);
    generateQueryPreviews(queryText); // Initial preview generation
  }, [id, queryText, methodItems, results, setDataPropsForNode]);

  const runTooltip =
    status === Status.LOADING ? "Stop Retrieval" : "Run Retrieval";

  return (
    <BaseNode
      nodeId={id}
      classNames="retrieval-node"
      style={{ width: "400px", backgroundColor: "rgba(255,255,255,0.9)" }}
    >
      <Handle
        type="target"
        position={Position.Left}
        id="fields"
        style={{ top: "50%", left: "0px", transform: "translate(-50%, -50%)" }}
      />
      <NodeLabel
        title={data.title || nodeDefaultTitle}
        nodeId={id}
        icon={nodeIcon}
        status={status}
        isRunning={status === Status.LOADING}
        handleRunClick={handleRunClick}
        handleStopClick={handleStopClick}
        handleRunHover={handleRunHover}
        runButtonTooltip={runTooltip}
        customButtons={[
          <PromptListPopover
            key="prompt-previews"
            promptInfos={queryPreviews}
            onHover={handlePreviewHover}
            onClick={openInfoModal}
          />,
        ]}
      />
      <Modal
        title={`Query Previews (${queryPreviews.length} total)`}
        size="xl"
        opened={infoModalOpened}
        onClose={closeInfoModal}
        styles={{
          header: { backgroundColor: "#FFD700" },
          root: { position: "relative", left: "-5%" },
        }}
      >
        <div style={{ padding: "16px" }}>
          {queryPreviews.map((preview, idx) => (
            <Text
              key={idx}
              size="xs"
              style={{ whiteSpace: "pre-wrap", margin: "4px 0" }}
            >
              {preview}
            </Text>
          ))}
        </div>
      </Modal>
      <div style={{ padding: 8, position: "relative" }}>
        <div>
          <Text size="xs" weight={500} mb={4}>
            Search Queries
          </Text>
          <textarea
            ref={setTextAreaRef}
            className="query-field-fixed nodrag nowheel"
            placeholder="Enter queries (one per line, use {} for variables)"
            value={queryText}
            onChange={handleQueryChange}
            style={{
              width: "100%",
              minHeight: "100px",
              resize: "vertical",
              padding: "8px",
              fontSize: "14px",
              border: "1px solid #ced4da",
              borderRadius: "4px",
              backgroundColor: "#fff",
              boxSizing: "border-box",
            }}
          />
        </div>
        <TemplateHooks
          vars={templateVars}
          nodeId={id}
          startY={hooksY - 15}
          position={Position.Left}
        />
        <hr />
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
      <Handle
        type="source"
        position={Position.Right}
        id="output"
        style={{ top: "50%" }}
      />
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
