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
  Select,
  Box,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { IconPlus, IconTrash, IconSettings } from "@tabler/icons-react";
import Form from "@rjsf/core";
import { RJSFSchema, UiSchema } from "@rjsf/utils";
import validator from "@rjsf/validator-ajv8";
import { v4 as uuid } from "uuid";
import {
  RetrievalMethodSchemas,
  retrievalMethodGroups,
  embeddingModels,
} from "./RetrievalMethodSchemas";

// Individual retrieval method item interface
export interface RetrievalMethodSpec {
  key: string;
  baseMethod: string;
  methodName: string;
  library: string;
  emoji?: string;
  needsEmbeddingModel?: boolean;
  embeddingModel?: string;
  settings?: Record<string, any>;
}

// Settings modal for individual retrieval methods
interface SettingsModalProps {
  opened: boolean;
  onClose: () => void;
  methodItem: RetrievalMethodSpec;
  onSettingsUpdate: (settings: any) => void;
}

const SettingsModal: React.FC<SettingsModalProps> = ({
  opened,
  onClose,
  methodItem,
  onSettingsUpdate,
}) => {
  const schema = RetrievalMethodSchemas[methodItem.baseMethod];
  if (!schema) return null;

  return (
    <Modal
      opened={opened}
      onClose={onClose}
      title={`Settings: ${methodItem.methodName}`}
      size="lg"
    >
      <Form<any, RJSFSchema, any>
        schema={schema.schema as RJSFSchema}
        uiSchema={schema.uiSchema as UiSchema}
        validator={validator}
        formData={methodItem.settings}
        onChange={(e) => onSettingsUpdate(e.formData)}
      >
        <Button type="submit" style={{ display: "none" }} />
      </Form>
    </Modal>
  );
};

// Individual retrieval method list item
interface RetrievalMethodListItemProps {
  methodItem: RetrievalMethodSpec;
  onRemove: (key: string) => void;
  onSettingsUpdate: (key: string, settings: any) => void;
  onEmbeddingModelUpdate?: (key: string, model: string) => void;
}

const RetrievalMethodListItem: React.FC<RetrievalMethodListItemProps> = ({
  methodItem,
  onRemove,
  onSettingsUpdate,
  onEmbeddingModelUpdate,
}) => {
  const [opened, { open, close }] = useDisclosure(false);

  return (
    <Card withBorder mb="xs" padding="xs">
      <Group position="apart" noWrap>
        <Box style={{ flex: 1 }}>
          <Group spacing="xs" noWrap>
            <Text size="sm">
              {methodItem.emoji && `${methodItem.emoji} `}
              {methodItem.methodName}
            </Text>
            {methodItem.needsEmbeddingModel && (
              <Select
                size="xs"
                placeholder="Select embedding model"
                value={methodItem.embeddingModel}
                onChange={(value) =>
                  onEmbeddingModelUpdate?.(methodItem.key, value || "")
                }
                data={embeddingModels}
                styles={(theme) => ({
                  root: {
                    flex: 1,
                    minWidth: 200,
                  },
                  input: {
                    minHeight: 28,
                  },
                })}
                withinPortal
                searchable
                clearable
              />
            )}
          </Group>
        </Box>
        <Group spacing={4} noWrap>
          <ActionIcon
            size="sm"
            variant="subtle"
            color="gray"
            onClick={() => open()}
          >
            <IconSettings size={14} />
          </ActionIcon>
          <ActionIcon
            size="sm"
            variant="subtle"
            color="red"
            onClick={() => onRemove(methodItem.key)}
          >
            <IconTrash size={14} />
          </ActionIcon>
        </Group>
      </Group>

      <SettingsModal
        opened={opened}
        onClose={close}
        methodItem={methodItem}
        onSettingsUpdate={(settings) =>
          onSettingsUpdate(methodItem.key, settings)
        }
      />
    </Card>
  );
};

// Main container component
export interface RetrievalMethodListContainerProps {
  initMethodItems?: RetrievalMethodSpec[];
  onItemsChange?: (
    newItems: RetrievalMethodSpec[],
    oldItems: RetrievalMethodSpec[],
  ) => void;
}

export const RetrievalMethodListContainer = forwardRef<
  any,
  RetrievalMethodListContainerProps
>((props, ref) => {
  const [methodItems, setMethodItems] = useState<RetrievalMethodSpec[]>(
    props.initMethodItems || [],
  );
  const oldItemsRef = useRef<RetrievalMethodSpec[]>(methodItems);

  useImperativeHandle(ref, () => ({
    getMethodItems: () => methodItems,
  }));

  const notifyItemsChanged = useCallback(
    (newItems: RetrievalMethodSpec[]) => {
      props.onItemsChange?.(newItems, oldItemsRef.current);
      oldItemsRef.current = newItems;
    },
    [props.onItemsChange],
  );

  const handleRemoveMethod = useCallback(
    (key: string) => {
      const newItems = methodItems.filter((m) => m.key !== key);
      setMethodItems(newItems);
      notifyItemsChanged(newItems);
    },
    [methodItems, notifyItemsChanged],
  );

  const handleSettingsUpdate = useCallback(
    (key: string, newSettings: any) => {
      const newItems = methodItems.map((m) =>
        m.key === key ? { ...m, settings: newSettings } : m,
      );
      setMethodItems(newItems);
      notifyItemsChanged(newItems);
    },
    [methodItems, notifyItemsChanged],
  );

  const handleEmbeddingModelUpdate = useCallback(
    (key: string, model: string) => {
      const newItems = methodItems.map((m) =>
        m.key === key ? { ...m, embeddingModel: model } : m,
      );
      setMethodItems(newItems);
      notifyItemsChanged(newItems);
    },
    [methodItems, notifyItemsChanged],
  );

  const addMethod = useCallback(
    (m: Omit<RetrievalMethodSpec, "key" | "settings">) => {
      const newItem: RetrievalMethodSpec = {
        key: uuid(),
        baseMethod: m.baseMethod,
        methodName: m.methodName,
        library: m.library,
        emoji: m.emoji,
        needsEmbeddingModel: m.needsEmbeddingModel,
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
          <RetrievalMethodListItem
            key={item.key}
            methodItem={item}
            onRemove={handleRemoveMethod}
            onSettingsUpdate={handleSettingsUpdate}
            onEmbeddingModelUpdate={handleEmbeddingModelUpdate}
          />
        ))
      )}
    </div>
  );
});

RetrievalMethodListContainer.displayName = "RetrievalMethodListContainer";
export default RetrievalMethodListContainer;
