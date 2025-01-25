import React, {
  useState,
  useCallback,
  useEffect,
  useImperativeHandle,
  forwardRef,
  Suspense,
  useRef,
} from "react";
import { Handle, Position } from "reactflow";
import {
  Button,
  TextInput,
  Group,
  Text,
  Modal,
  LoadingOverlay,
  Table,
  ScrollArea,
  Tabs,
  ActionIcon,
  Menu,
  Select,
  Divider,
} from "@mantine/core";
import {
  IconSearch,
  IconLayoutList,
  IconTable,
  IconTrash,
  IconPlus,
  IconSettings,
} from "@tabler/icons-react";
import { useDisclosure } from "@mantine/hooks";
import { v4 as uuid } from "uuid";
import Form from "@rjsf/core";
import validator from "@rjsf/validator-ajv8";

// Import your custom node and store
import BaseNode from "./BaseNode";
import NodeLabel from "./NodeLabelComponent";
import useStore from "./store";
import InspectFooter from "./InspectFooter";

// -------------------- Types & Interfaces --------------------

export interface RetrievalChunk {
  text: string;
  similarity: number;
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
}

// -------------------- Chunk Inspector UI --------------------

const methodColors = ["#FFEDD5", "#FEF3C7", "#ECFDF5", "#E0F2FE", "#EDE9FE"];
const colorForIndex = (index: number) =>
  methodColors[index % methodColors.length];

const ChunkDisplay: React.FC<{ chunk: RetrievalChunk }> = ({ chunk }) => {
  const [expanded, setExpanded] = useState(false);
  const sampleText =
    chunk.text.length > 40 ? chunk.text.substring(0, 40) + "..." : chunk.text;

  return (
    <div>
      <Text size="xs">{expanded ? chunk.text : sampleText}</Text>
      {chunk.text.length > 40 && (
        <Button
          variant="subtle"
          compact
          onClick={() => setExpanded((v) => !v)}
          mt={4}
        >
          {expanded ? "Show Less" : "Show Full"}
        </Button>
      )}
    </div>
  );
};

interface ChunkInspectorProps {
  results: RetrievalResults;
  wideFormat?: boolean;
}

const ChunkInspector: React.FC<ChunkInspectorProps> = ({
  results,
  wideFormat,
}) => {
  const [viewFormat, setViewFormat] = useState<"list" | "table">("list");
  const entries = Object.entries(results);

  const tableView = (
    <Table striped withColumnBorders fontSize={wideFormat ? "sm" : "xs"}>
      <thead>
        <tr>
          <th>Method</th>
          <th>Chunk Sample</th>
          <th>Similarity</th>
        </tr>
      </thead>
      <tbody>
        {entries.map(([key, result], methodIndex) =>
          result.retrieved.map((chunk, idx) => (
            <tr
              key={`${key}-${idx}`}
              style={{ backgroundColor: colorForIndex(methodIndex) }}
            >
              {idx === 0 && (
                <td
                  rowSpan={result.retrieved.length}
                  style={{ verticalAlign: "middle", fontWeight: 600 }}
                >
                  {result.label}
                </td>
              )}
              <td>
                <ScrollArea style={{ maxHeight: 100 }}>
                  <ChunkDisplay chunk={chunk} />
                </ScrollArea>
              </td>
              <td>
                <Text size={wideFormat ? "sm" : "xs"}>
                  {chunk.similarity.toFixed(3)}
                </Text>
              </td>
            </tr>
          )),
        )}
      </tbody>
    </Table>
  );

  const listView = (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {entries.map(([key, result], methodIndex) => (
        <div
          key={key}
          style={{
            padding: "8px",
            backgroundColor: colorForIndex(methodIndex),
            borderRadius: 6,
          }}
        >
          <Text weight={600} mb={4}>
            {result.label}
          </Text>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {result.retrieved.map((chunk, idx) => (
              <div
                key={`${key}-${idx}`}
                style={{
                  flex: "1 0 30%",
                  padding: "8px",
                  border: "1px solid #ccc",
                  borderRadius: 4,
                  backgroundColor: "#ffffff",
                }}
              >
                <ChunkDisplay chunk={chunk} />
                <Text size={wideFormat ? "sm" : "xs"} weight={500} mt={4}>
                  Similarity: {chunk.similarity.toFixed(3)}
                </Text>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );

  return (
    <div style={{ padding: 8 }}>
      <Tabs
        value={viewFormat}
        onTabChange={(val) => setViewFormat(val as "list" | "table")}
        styles={{ tabLabel: { fontSize: wideFormat ? "12pt" : "9pt" } }}
      >
        <Tabs.List>
          <Tabs.Tab value="list">
            <IconLayoutList size="10pt" style={{ marginBottom: 2 }} /> List View
          </Tabs.Tab>
          <Tabs.Tab value="table">
            <IconTable size="10pt" style={{ marginBottom: 2 }} /> Table View
          </Tabs.Tab>
        </Tabs.List>
        <Tabs.Panel value="list" pt="xs">
          {listView}
        </Tabs.Panel>
        <Tabs.Panel value="table" pt="xs">
          {tableView}
        </Tabs.Panel>
      </Tabs>
    </div>
  );
};

export interface ChunkInspectorModalRef {
  trigger: () => void;
}

export interface ChunkInspectorModalProps {
  results: RetrievalResults;
}

const ChunkInspectorModal = forwardRef<
  ChunkInspectorModalRef,
  ChunkInspectorModalProps
>(function ChunkInspectorModal({ results }, ref) {
  const [opened, setOpened] = useState(false);
  const trigger = () => setOpened(true);
  useImperativeHandle(ref, () => ({ trigger }));

  return (
    <Modal
      opened={opened}
      onClose={() => setOpened(false)}
      size="90%"
      title={
        <Group position="apart" noWrap>
          <Text>Retrieved Chunks</Text>
          <Button
            variant="outline"
            size="xs"
            onClick={() =>
              navigator.clipboard.writeText(JSON.stringify(results, null, 2))
            }
          >
            Copy JSON
          </Button>
        </Group>
      }
    >
      <Suspense fallback={<LoadingOverlay visible />}>
        <ChunkInspector results={results} wideFormat={true} />
      </Suspense>
    </Modal>
  );
});
ChunkInspectorModal.displayName = "ChunkInspectorModal";

// -------------------- Retrieval Method Groups & Menu --------------------

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

  // Insert new method with optional vector library
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

// -------------------- Main Retrieval Node Component --------------------

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
  const [loading, setLoading] = useState<boolean>(false);

  const inspectorModalRef = useRef<ChunkInspectorModalRef>(null);

  // Clear existing results if refresh is triggered
  useEffect(() => {
    if (data.refresh) {
      setDataPropsForNode(id, { refresh: false, results: {} });
      setResults({});
    }
  }, [data.refresh, id, setDataPropsForNode]);

  // Fired whenever the user changes retrieval methods
  const handleMethodsChange = useCallback(
    (newItems: RetrievalMethodSpec[]) => {
      setMethodItems(newItems);
      // Remove any old result items that no longer match
      setResults((prevResults) => {
        const updated = { ...prevResults };
        Object.keys(updated).forEach((key) => {
          if (!newItems.some((item) => item.key === key)) {
            delete updated[key];
          }
        });
        return updated;
      });
    },
    [setMethodItems, setResults],
  );

  // Flatten & deduplicate final retrieved chunks
  const prepareOutput = (resultsData: RetrievalResults): RetrievalChunk[] => {
    const allChunks: RetrievalChunk[] = [];
    Object.values(resultsData).forEach((methodResult) => {
      allChunks.push(...methodResult.retrieved);
    });
    // Sort descending by similarity
    allChunks.sort((a, b) => b.similarity - a.similarity);

    const seen = new Set<string>();
    return allChunks.filter((chunk) => {
      if (seen.has(chunk.text)) return false;
      seen.add(chunk.text);
      return true;
    });
  };

  // The main action that queries your backend for retrieval
  const runRetrieval = useCallback(async () => {
    if (!query.trim()) {
      alert("Please enter a search query.");
      return;
    }
    let upstreamData: { text?: Array<{ text: string }> } = {};
    try {
      // Attempt to pull input data from connected nodes
      upstreamData = useStore.getState().pullInputData(["text"], id) as {
        text?: Array<{ text: string }>;
      };
    } catch (error) {
      alert("No input text found. Is an UploadNode or ChunkingNode connected?");
      return;
    }
    const texts = upstreamData.text || [];
    if (texts.length === 0) {
      alert("No text found in upstream node.");
      return;
    }

    const newResults: RetrievalResults = {};
    setLoading(true);

    for (const method of methodItems) {
      const payload: any = {
        query,
        chunks: texts.map((t) => t.text || ""),
        top_k: 5,
      };

      // Distinguish vector-based vs. keyword-based
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
            payload.library = "Boolean Search";
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
        const res = await fetch("http://localhost:5000/retrieve", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.error || "Retrieval request failed");
        }
        const json = await res.json();
        const label =
          method.methodName +
          (method.needsVector && method.vectorLib
            ? ` (${method.vectorLib})`
            : "");
        newResults[method.key] = { label, retrieved: json.retrieved };
      } catch (err: any) {
        console.error(`Error with method ${method.methodName}:`, err);
        alert(`Retrieval failed for ${method.methodName}: ${err.message}`);
      }
    }

    const outputChunks = prepareOutput(newResults);
    setResults(newResults);

    // Persist to node data
    setDataPropsForNode(id, {
      query,
      methods: methodItems,
      results: newResults,
      output: outputChunks,
    });

    pingOutputNodes(id);
    setLoading(false);
  }, [query, methodItems, id, setDataPropsForNode, pingOutputNodes]);

  // Persist whenever query/methodItems/results changes
  useEffect(() => {
    setDataPropsForNode(id, { query, methods: methodItems, results });
  }, [id, query, methodItems, results, setDataPropsForNode]);

  return (
    <BaseNode
      nodeId={id}
      classNames="retrieval-node"
      style={{ backgroundColor: "rgba(255, 255, 255, 0.9)" }}
    >
      <Handle
        type="target"
        position={Position.Left}
        id="text"
        style={{ top: "50%" }}
      />

      {/* Node label with your run button */}
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
        {/* Our new multi-level menu container */}
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
        label={
          <>
            Inspect responses <IconSearch size="12pt" />
          </>
        }
      />

      <Handle
        type="source"
        position={Position.Right}
        id="output"
        style={{ top: "50%" }}
      />

      <Suspense fallback={null}>
        <ChunkInspectorModal ref={inspectorModalRef} results={results} />
      </Suspense>
      <Handle
        type="source"
        position={Position.Right}
        id="output"
        style={{ top: "50%" }}
      />

      <Suspense fallback={null}>
        <ChunkInspectorModal ref={inspectorModalRef} results={results} />
      </Suspense>
    </BaseNode>
  );
};

export default RetrievalNode;
