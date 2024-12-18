import React, { useState, useRef, useCallback } from "react";
import { Handle, Position } from "reactflow";
import {
  Box,
  Button,
  Text,
  Divider,
  Popover,
  Group,
  useMantineTheme,
  Tooltip,
  Menu,
} from "@mantine/core";
import { IconPlus, IconTrash, IconList } from "@tabler/icons-react";
import BaseNode from "./BaseNode";
import NodeLabel from "./NodeLabelComponent";
import InspectFooter from "./InspectFooter";
import LLMResponseInspectorModal, {
  LLMResponseInspectorModalRef,
} from "./LLMResponseInspectorModal";
import LLMResponseInspectorDrawer from "./LLMResponseInspectorDrawer";
import { v4 as uuid } from "uuid";
import { Status } from "./StatusIndicatorComponent";

interface ChunkingNodeProps {
  data: {
    title?: string;
    nodeId?: string;
    onRemoveNode?: (nodeId: string) => void;
    setDataPropsForNode?: (nodeId: string, props: any) => void;
  };
}

interface ChunkDetail {
  index: number;
  length: number;
}

interface ResponseObject {
  methodName: string;
  library: string;
  chunks: string[];
}

interface JSONResponse {
  prompt: string;
  vars: Record<string, any>;
  llm: string;
  responses: string[];
  uid: string;
  metavars: {
    method: string;
    library: string;
    chunk_count: number;
    chunk_details: ChunkDetail[];
  };
}

interface TemplateVarInfo {
  text: string;
  prompt: string;
  fill_history: Record<string, any>;
  llm: { name: string };
  uid: string;
  metavars: {
    method: string;
    library: string;
    chunk_index: number;
    total_chunks: number;
  };
}

function ChunkingNode({ data }: ChunkingNodeProps) {
  const theme = useMantineTheme();
  const [selectedMethods, setSelectedMethods] = useState<
    { name: string; library: string }[]
  >([]);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [menuOpened, setMenuOpened] = useState(false);
  const [hoveredMethod, setHoveredMethod] = useState<{
    id: string;
    name: string;
    libraries: string[];
  } | null>(null);

  const [status, setStatus] = useState(Status.NONE);
  const [isRunning, setIsRunning] = useState(false);

  // For Inspecting responses
  const inspectModal = useRef<LLMResponseInspectorModalRef>(null);
  const [showDrawer, setShowDrawer] = useState(false);
  const [jsonResponses, setJsonResponses] = useState<JSONResponse[] | null>(
    null,
  );

  // Methods available for chunking
  const methods = [
    {
      id: "overlapping",
      name: "Overlapping Chunking",
      libraries: [
        "LangChain's TextSplitter",
        "OpenAI tiktoken",
        "HuggingFace Tokenizers",
      ],
    },
    {
      id: "syntax",
      name: "Syntax-Based Chunking",
      libraries: ["spaCy Sentence Splitter", "TextTilingTokenizer"],
    },
    {
      id: "hybrid",
      name: "Hybrid Chunking",
      libraries: ["TextTiling + spaCy", "BERTopic + spaCy"],
    },
  ];
  // Removed unused closeNode function

  const convertToJSONResponsesAndFields = useCallback(
    (results: ResponseObject[]): JSONResponse[] => {
      const newResponses: JSONResponse[] = [];
      const newFields: TemplateVarInfo[] = [];

      for (const resObj of results) {
        const llm = `${resObj.methodName} - ${resObj.library}`;
        const uid = uuid();

        const chunk_count = resObj.chunks.length;
        const chunk_details: ChunkDetail[] = resObj.chunks.map((chunk, i) => ({
          index: i,
          length: chunk.length,
        }));

        const resp: JSONResponse = {
          prompt: `File processed by ${resObj.methodName} using ${resObj.library}. Total chunks: ${chunk_count}`,
          vars: {},
          llm,
          responses: resObj.chunks,
          uid,
          metavars: {
            method: resObj.methodName,
            library: resObj.library,
            chunk_count: chunk_count,
            chunk_details: chunk_details,
          },
        };
        newResponses.push(resp);

        // Create TemplateVarInfo objects for downstream nodes
        resObj.chunks.forEach((chunk, i) => {
          const field: TemplateVarInfo = {
            text: chunk,
            prompt: resp.prompt,
            fill_history: resp.vars,
            llm: { name: llm },
            uid: uid,
            metavars: {
              method: resObj.methodName,
              library: resObj.library,
              chunk_index: i,
              total_chunks: chunk_count,
            },
          };
          newFields.push(field);
        });
      }

      // Set fields on the node’s data so downstream nodes can read them via "output"
      if (data?.setDataPropsForNode && data.nodeId) {
        data.setDataPropsForNode(data.nodeId, { fields: newFields });
      }

      return newResponses;
    },
    [data],
  );

  const runChunking = async () => {
    if (!uploadedFile || selectedMethods.length === 0) {
      console.warn("Cannot run chunking: no file or no methods selected.");
      return;
    }

    setIsRunning(true);
    setStatus(Status.LOADING);

    const newResults: ResponseObject[] = [];
    for (const methodObj of selectedMethods) {
      const { name, library } = methodObj;
      const formData = new FormData();
      formData.append("file", uploadedFile);
      formData.append("methodName", name);
      formData.append("library", library);

      try {
        const res = await fetch("http://localhost:5000/process", {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          console.error(
            `API Error running ${name} (${library}): ${res.statusText}`,
          );
          continue;
        }

        const json = await res.json();
        newResults.push({
          methodName: name,
          library: library,
          chunks: json.chunks || [],
        });
      } catch (error) {
        console.error(`Fetch error for ${name} (${library}): ${error}`);
      }
    }

    setIsRunning(false);
    setStatus(Status.READY);

    const responses = convertToJSONResponsesAndFields(newResults);
    setJsonResponses(responses);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || !files[0]) return;
    const file = files[0];
    setUploadedFile(file);
  };

  const handleMethodSelect = (method: { name: string; library: string }) => {
    setSelectedMethods((prev) => [...prev, method]);
    setMenuOpened(false);
    setHoveredMethod(null);
  };

  const removeMethod = (index: number) => {
    setSelectedMethods((prev) => prev.filter((_, i) => i !== index));
  };

  const showResponseInspector = useCallback(() => {
    if (inspectModal.current && jsonResponses && jsonResponses.length > 0) {
      inspectModal.current.trigger();
    }
  }, [inspectModal, jsonResponses]);

  const [methodsPopoverOpened, setMethodsPopoverOpened] = useState(false);
  const handleMethodsHover = () => {
    setMethodsPopoverOpened(true);
  };

  const displaySelectedMethods = (wideFormat: boolean) =>
    selectedMethods.map((m, idx) => (
      <div key={idx} className="prompt-preview">
        {m.name} - {m.library}
      </div>
    ));

  const MethodsListPopover = useCallback(
    () => (
      <Popover
        position="right-start"
        withArrow
        withinPortal
        shadow="rgb(38, 57, 77) 0px 10px 30px -14px"
        opened={methodsPopoverOpened}
        onClose={() => setMethodsPopoverOpened(false)}
        styles={{
          dropdown: {
            maxHeight: "200px",
            maxWidth: "200px",
            overflowY: "auto",
            backgroundColor: "#fff",
          },
        }}
      >
        <Popover.Target>
          <Tooltip label="View selected methods" withArrow>
            <button
              className="custom-button"
              onMouseEnter={handleMethodsHover}
              onMouseLeave={() => setMethodsPopoverOpened(false)}
              style={{ border: "none" }}
            >
              <IconList
                size="12pt"
                color="gray"
                style={{ marginBottom: "-4px" }}
              />
            </button>
          </Tooltip>
        </Popover.Target>
        <Popover.Dropdown sx={{ pointerEvents: "none" }}>
          <Box m="xs">
            <Text size="xs" fw={500} color="#666">
              Selected Methods ({selectedMethods.length})
            </Text>
          </Box>
          {displaySelectedMethods(false)}
        </Popover.Dropdown>
      </Popover>
    ),
    [selectedMethods, methodsPopoverOpened],
  );

  return (
    <BaseNode nodeId={data?.nodeId || "?"}>
      <div
        style={{
          background: "#fff",
          border: "1px solid #ccc",
          borderRadius: "8px",
          padding: "10px",
          minWidth: "250px",
        }}
      >
        <Handle
          type="source"
          position={Position.Right}
          id="output"
          style={{ background: "#555" }}
        />

        <NodeLabel
          title={data?.title || "Chunking Node"}
          nodeId={data?.nodeId || "?"}
          icon="📝"
          status={status}
          isRunning={isRunning}
          handleRunClick={runChunking}
          handleStopClick={() => {
            setIsRunning(false);
            setStatus(Status.NONE);
          }}
          runButtonTooltip={"Run all selected methods"}
          customButtons={[<MethodsListPopover key="methods-list-popover" />]}
        />

        <LLMResponseInspectorModal
          ref={inspectModal}
          jsonResponses={jsonResponses ?? []}
        />

        <Handle
          type="target"
          position={Position.Left}
          style={{ background: "#555" }}
        />

        <div style={{ marginBottom: "10px" }}>
          <Text size="xs" color="#333">
            Select a file:
          </Text>
          <Button
            onClick={() => fileInputRef.current?.click()}
            variant="filled"
            color="teal"
            size="xs"
            mt="sm"
          >
            Choose File
          </Button>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            style={{ display: "none" }}
          />
          {uploadedFile && (
            <Text size="xs" mt="xs">
              {uploadedFile.name}
            </Text>
          )}
          {selectedMethods.length === 0 && uploadedFile && (
            <Text size="xs" color="red" mt="xs">
              No methods selected. Please select at least one method.
            </Text>
          )}
        </div>

        <Divider mb="sm" />

        <Box mb="sm" mt="sm">
          <Text weight={500} mb="xs">
            Selected Methods:
          </Text>
          {selectedMethods.length === 0 && (
            <Text size="sm" color="dimmed">
              No methods selected yet.
            </Text>
          )}
          {selectedMethods.map((m, i) => (
            <Group key={i} spacing="xs" mb="xs">
              <Text size="sm">
                {m.name} - {m.library}
              </Text>
              <IconTrash
                size={14}
                style={{ marginLeft: "auto", cursor: "pointer" }}
                onClick={() => removeMethod(i)}
              />
            </Group>
          ))}
        </Box>

        <Menu
          opened={menuOpened}
          onClose={() => {
            setMenuOpened(false);
            setHoveredMethod(null);
          }}
          position="bottom-start"
          withinPortal
        >
          <Menu.Target>
            <Button
              leftIcon={<IconPlus size={16} />}
              variant="outline"
              size="xs"
              onClick={() => setMenuOpened((o) => !o)}
            >
              Add Method
            </Button>
          </Menu.Target>

          <Menu.Dropdown>
            {methods.map((method) => (
              <Popover
                key={method.id}
                opened={hoveredMethod?.id === method.id}
                position="right-start"
                withinPortal
                offset={0}
                trapFocus={false}
                onClose={() => setHoveredMethod(null)}
                closeOnClickOutside
                styles={{
                  dropdown: {
                    backgroundColor: theme.colors.gray[0],
                    border: `1px solid ${theme.colors.gray[3]}`,
                    borderRadius: theme.radius.sm,
                  },
                }}
              >
                <Popover.Target>
                  <Box
                    onMouseEnter={() => setHoveredMethod(method)}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      padding: "6px 8px",
                      borderRadius: theme.radius.sm,
                      cursor: "default",
                      backgroundColor:
                        hoveredMethod?.id === method.id
                          ? theme.colors.gray[1]
                          : "transparent",
                    }}
                  >
                    <Text size="sm">{method.name}</Text>
                    <Text
                      size="xs"
                      color="dimmed"
                      style={{ marginLeft: theme.spacing.xs }}
                    >
                      ▶
                    </Text>
                  </Box>
                </Popover.Target>

                <Popover.Dropdown>
                  {method.libraries.map((lib) => (
                    <Box
                      key={lib}
                      onClick={() =>
                        handleMethodSelect({ name: method.name, library: lib })
                      }
                      style={{
                        display: "flex",
                        alignItems: "center",
                        cursor: "pointer",
                        padding: "4px 8px",
                        borderRadius: theme.radius.sm,
                      }}
                      onMouseEnter={(e) =>
                        (e.currentTarget.style.backgroundColor =
                          theme.colors.gray[1])
                      }
                      onMouseLeave={(e) =>
                        (e.currentTarget.style.backgroundColor = "transparent")
                      }
                    >
                      <Text size="sm">{lib}</Text>
                    </Box>
                  ))}
                </Popover.Dropdown>
              </Popover>
            ))}
          </Menu.Dropdown>
        </Menu>

        <Divider mb="sm" mt="sm" />

        {jsonResponses && jsonResponses.length > 0 && status !== "loading" ? (
          <InspectFooter
            onClick={showResponseInspector}
            isDrawerOpen={showDrawer}
            showDrawerButton={true}
            onDrawerClick={() => setShowDrawer(!showDrawer)}
          />
        ) : null}

        <LLMResponseInspectorDrawer
          jsonResponses={jsonResponses ?? []}
          showDrawer={showDrawer}
        />
      </div>
    </BaseNode>
  );
}

export default ChunkingNode;
