import React, {
  useState,
  useRef,
  forwardRef,
  useImperativeHandle,
  useCallback,
} from "react";
import {
  Menu,
  Button,
  Card,
  Group,
  Text,
  ActionIcon,
  Modal,
  Divider,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { IconPlus, IconTrash, IconSettings } from "@tabler/icons-react";
import Form from "@rjsf/core";
import validator from "@rjsf/validator-ajv8";
import { v4 as uuid } from "uuid";
import { ChunkMethodSchemas } from "./ChunkMethodSchemas";

const chunkMethodGroups = [
  {
    label: "Overlapping Chunking",
    items: [
      {
        baseMethod: "overlapping_langchain",
        methodName: "Overlapping Chunking",
        library: "LangChain's TextSplitter",
        emoji: "🌐",
      },
      {
        baseMethod: "overlapping_openai_tiktoken",
        methodName: "Overlapping Chunking",
        library: "OpenAI tiktoken",
        emoji: "🤖",
      },
      {
        baseMethod: "overlapping_huggingface_tokenizers",
        methodName: "Overlapping Chunking",
        library: "HuggingFace Tokenizers",
        emoji: "🤗",
      },
    ],
  },
  {
    label: "Syntax-Based Chunking",
    items: [
      {
        baseMethod: "syntax_spacy",
        methodName: "Syntax-Based Chunking",
        library: "spaCy Sentence Splitter",
        emoji: "🐍",
      },
      {
        baseMethod: "syntax_texttiling",
        methodName: "Syntax-Based Chunking",
        library: "TextTilingTokenizer",
        emoji: "📑",
      },
    ],
  },
  {
    label: "Hybrid Chunking",
    items: [
      {
        baseMethod: "hybrid_texttiling_spacy",
        methodName: "Hybrid Chunking",
        library: "TextTiling + spaCy",
        emoji: "⚗️",
      },
      {
        baseMethod: "hybrid_bertopic_spacy",
        methodName: "Hybrid Chunking",
        library: "BERTopic + spaCy",
        emoji: "🧠",
      },
      {
        baseMethod: "hybrid_recursive_gensim",
        methodName: "Hybrid Chunking",
        library: "Recursive TextSplitter + Gensim",
        emoji: "🔎",
      },
      {
        baseMethod: "hybrid_recursive_cohere",
        methodName: "Hybrid Chunking",
        library: "Recursive TextSplitter + Cohere",
        emoji: "💬",
      },
      {
        baseMethod: "hybrid_recursive_bertopic",
        methodName: "Hybrid Chunking",
        library: "Recursive TextSplitter + BERTopic",
        emoji: "🌐",
      },
    ],
  },
];

export interface ChunkMethodSpec {
  key: string;
  baseMethod: string;
  methodName: string;
  library: string;
  emoji?: string;
  settings?: Record<string, any>;
}

export interface ChunkMethodListContainerProps {
  initMethodItems?: ChunkMethodSpec[];
  onItemsChange?: (
    newItems: ChunkMethodSpec[],
    oldItems: ChunkMethodSpec[],
  ) => void;
}
export type ChunkMethodListContainerRef = Record<string, never>;

const ChunkMethodListItem: React.FC<{
  methodItem: ChunkMethodSpec;
  onRemove: (key: string) => void;
  onSettingsUpdate: (key: string, newSettings: any) => void;
}> = ({ methodItem, onRemove, onSettingsUpdate }) => {
  const schemaEntry = ChunkMethodSchemas[methodItem.baseMethod];
  const { schema, uiSchema, description, fullName } = schemaEntry || {
    schema: {},
    uiSchema: {},
    description: "",
    fullName: "",
  };

  const [settingsModalOpen, { open, close }] = useDisclosure(false);

  const handleFormChange = useCallback(
    (evt: any) => {
      onSettingsUpdate(methodItem.key, evt.formData);
    },
    [methodItem.key, onSettingsUpdate],
  );

  return (
    <Card shadow="sm" p="sm" withBorder mt="xs">
      <Group position="apart" align="center">
        <div style={{ maxWidth: "70%" }}>
          <Text size="sm" weight={600}>
            {methodItem.emoji ? methodItem.emoji + " " : ""}
            {methodItem.methodName} ({methodItem.library})
          </Text>
          <Text size="xs" color="dimmed">
            {fullName || description || ""}
          </Text>
        </div>
        <Group spacing="xs">
          <ActionIcon variant="subtle" onClick={open} title="Open Settings">
            <IconSettings size={16} />
          </ActionIcon>
          <ActionIcon
            color="red"
            variant="subtle"
            onClick={() => onRemove(methodItem.key)}
          >
            <IconTrash size={16} />
          </ActionIcon>
        </Group>
      </Group>

      <Modal
        opened={settingsModalOpen}
        onClose={close}
        title="Chunk Method Settings"
        size="md"
      >
        {schema && Object.keys(schema).length > 0 ? (
          <Form
            schema={schema}
            uiSchema={uiSchema}
            formData={methodItem.settings}
            onChange={handleFormChange}
            validator={validator as any}
            liveValidate
            noHtml5Validate
          ></Form>
        ) : (
          <Text size="sm" color="dimmed">
            (No custom settings for this method.)
          </Text>
        )}
      </Modal>
    </Card>
  );
};

const ChunkMethodListContainer = forwardRef<
  ChunkMethodListContainerRef,
  ChunkMethodListContainerProps
>((props, ref) => {
  const [methodItems, setMethodItems] = useState<ChunkMethodSpec[]>(
    props.initMethodItems || [],
  );
  const oldItemsRef = useRef<ChunkMethodSpec[]>(methodItems);

  useImperativeHandle(ref, () => ({}));

  // If parent node wants to track changes
  const notifyItemsChanged = useCallback(
    (newItems: ChunkMethodSpec[]) => {
      props.onItemsChange?.(newItems, oldItemsRef.current);
      oldItemsRef.current = newItems;
    },
    [props.onItemsChange],
  );

  // Remove
  const handleRemoveMethod = useCallback(
    (key: string) => {
      const newItems = methodItems.filter((m) => m.key !== key);
      setMethodItems(newItems);
      notifyItemsChanged(newItems);
    },
    [methodItems, notifyItemsChanged],
  );

  // Update settings
  const handleSettingsUpdate = useCallback(
    (key: string, newSettings: any) => {
      const newItems = methodItems.map((m) =>
        m.key === key
          ? { ...m, settings: { ...m.settings, ...newSettings } }
          : m,
      );
      setMethodItems(newItems);
      notifyItemsChanged(newItems);
    },
    [methodItems, notifyItemsChanged],
  );


  const addMethod = useCallback(
    (m: Omit<ChunkMethodSpec, "key" | "settings">) => {
      const newItem: ChunkMethodSpec = {
        key: uuid(),
        baseMethod: m.baseMethod,
        methodName: m.methodName,
        library: m.library,
        emoji: m.emoji,
        settings: {},
      };
      const newItems = [...methodItems, newItem];
      setMethodItems(newItems);
      notifyItemsChanged(newItems);
    },
    [methodItems, notifyItemsChanged],
  );

  const [menuOpened, setMenuOpened] = useState(false);

  return (
    <div style={{ border: "1px dashed #ccc", borderRadius: 6, padding: 8 }}>
      <Group position="apart" mb="xs">
        <Text weight={500} size="sm">
          Selected Chunk Methods
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
            {chunkMethodGroups.map((group, groupIdx) => (
              <React.Fragment key={group.label}>
                <Menu.Label>{group.label}</Menu.Label>
                {group.items.map((item) => (
                  <Menu.Item
                    key={item.baseMethod}
                    icon={item.emoji ? <Text>{item.emoji}</Text> : undefined}
                    onClick={() => {
                      addMethod(item);
                      setMenuOpened(false);
                    }}
                  >
                    {item.library}
                  </Menu.Item>
                ))}

                {groupIdx < chunkMethodGroups.length - 1 && <Divider my="xs" />}
              </React.Fragment>
            ))}
          </Menu.Dropdown>
        </Menu>
      </Group>

      {methodItems.length === 0 ? (
        <Text size="xs" color="dimmed">
          No chunk methods selected.
        </Text>
      ) : (
        methodItems.map((item) => (
          <ChunkMethodListItem
            key={item.key}
            methodItem={item}
            onRemove={handleRemoveMethod}
            onSettingsUpdate={handleSettingsUpdate}
          />
        ))
      )}
    </div>
  );
});

ChunkMethodListContainer.displayName = "ChunkMethodListContainer";
export default ChunkMethodListContainer;
