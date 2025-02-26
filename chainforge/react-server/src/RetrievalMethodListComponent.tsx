import React, { useState, useEffect, useCallback, useRef } from "react";
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
import {
  RetrievalMethodSpec,
  defaultMethodEmojis,
  vectorOptions,
  retrievalMethodGroups,
} from "./RetrievalMethodSchemas";

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
  const [openaiModel, setOpenaiModel] = useState<string>(
    initialSettings.openai_model ?? "text-embedding-ada-002",
  );
  const [bm25K1, setBm25K1] = useState<number>(initialSettings.bm25_k1 ?? 1.5);
  const [bm25B, setBm25B] = useState<number>(initialSettings.bm25_b ?? 0.75);
  const [maxFeatures, setMaxFeatures] = useState<number>(
    initialSettings.max_features ?? 500,
  );
  const [requiredMatchCount, setRequiredMatchCount] = useState<number>(
    initialSettings.required_match_count ?? 1,
  );
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

interface MethodListComponentProps {
  initMethodItems?: RetrievalMethodSpec[];
  onItemsChange?: (
    newItems: RetrievalMethodSpec[],
    oldItems: RetrievalMethodSpec[],
  ) => void;
  loadingMethods?: { [key: string]: boolean };
}

const MethodListComponent: React.FC<MethodListComponentProps> = ({
  initMethodItems = [],
  onItemsChange,
  loadingMethods,
}) => {
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

export default MethodListComponent;
