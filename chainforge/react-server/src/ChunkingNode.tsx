import React, {
  useState,
  useEffect,
  useCallback,
  useRef,
  useContext,
} from "react";
import { Handle, Position } from "reactflow";
import { v4 as uuid } from "uuid";
import { Status } from "./StatusIndicatorComponent";
import { AlertModalContext } from "./AlertModal";
import { Button, Group, ActionIcon } from "@mantine/core";
import BaseNode from "./BaseNode";
import NodeLabel from "./NodeLabelComponent";
import useStore from "./store";

import LLMResponseInspectorModal, {
  LLMResponseInspectorModalRef,
} from "./LLMResponseInspectorModal";
import InspectFooter from "./InspectFooter";
import { IconSearch, IconChevronDown } from "@tabler/icons-react";

import ChunkMethodListContainer, {
  ChunkMethodSpec,
} from "./ChunkMethodListComponent";

import { TemplateVarInfo, LLMResponse } from "./backend/typing";

interface ChunkingNodeProps {
  data: {
    title?: string;
    methods?: ChunkMethodSpec[];
    refresh?: boolean;
  };
  id: string;
}

const ChunkingNode: React.FC<ChunkingNodeProps> = ({ data, id }) => {
  const nodeDefaultTitle = "Chunking Node";
  const nodeIcon = "🔪";

  const pullInputData = useStore((s) => s.pullInputData);
  const setDataPropsForNode = useStore((s) => s.setDataPropsForNode);
  const pingOutputNodes = useStore((s) => s.pingOutputNodes);

  const showAlert = useContext(AlertModalContext);

  const [methodItems, setMethodItems] = useState<ChunkMethodSpec[]>(
    data.methods || [],
  );
  const [status, setStatus] = useState<Status>(Status.NONE);
  const [jsonResponses, setJSONResponses] = useState<LLMResponse[]>([]);
  const [chunks, setChunks] = useState<TemplateVarInfo[]>([]);

  const inspectorRef = useRef<LLMResponseInspectorModalRef>(null);

  // On refresh
  useEffect(() => {
    if (data.refresh) {
      setDataPropsForNode(id, { refresh: false, fields: [], output: [] });
      setJSONResponses([]);
      setStatus(Status.NONE);
    }
  }, [data.refresh, id, setDataPropsForNode]);

  // Track changes in chunk methods
  const handleMethodItemsChange = useCallback(
    (newItems: ChunkMethodSpec[], _oldItems: ChunkMethodSpec[]) => {
      setMethodItems(newItems);
      setDataPropsForNode(id, { methods: newItems });
      if (status === Status.READY) setStatus(Status.WARNING);
    },
    [id, status, setDataPropsForNode],
  );

  // Truncate string helper
  const truncateString = (str: string, maxLen = 25): string => {
    if (!str) return "";
    if (str.length <= maxLen) return str;
    return `${str.slice(0, 12)}...${str.slice(-10)}`;
  };

  // The main chunking function
  const runChunking = useCallback(async () => {
    if (methodItems.length === 0) {
      showAlert?.("No chunk methods selected!");
      return;
    }

    // 1) Pull text from upstream (the UploadNode)
    let inputData: { text?: TemplateVarInfo[] } = {};
    try {
      inputData = pullInputData(["text"], id) as { text?: TemplateVarInfo[] };
    } catch (error) {
      console.error(error);
      showAlert?.("No input text found. Is UploadNode connected?");
      return;
    }
    const fileArr = inputData.text || [];
    if (fileArr.length === 0) {
      showAlert?.(
        "No text found. Please attach an UploadNode or provide text.",
      );
      return;
    }

    setStatus(Status.LOADING);
    setJSONResponses([]);

    // We'll group by library to call your chunker
    const allChunksByLibrary: Record<string, TemplateVarInfo[]> = {};
    const allResponsesByLibrary: Record<string, LLMResponse[]> = {};

    // Group methods by library
    const methodsByLibrary = methodItems.reduce(
      (acc, method) => {
        if (!acc[method.library]) acc[method.library] = [];
        acc[method.library].push(method);
        return acc;
      },
      {} as Record<string, ChunkMethodSpec[]>,
    );

    // 2) For each library and each doc
    for (const [library, methods] of Object.entries(methodsByLibrary)) {
      allChunksByLibrary[library] = [];
      allResponsesByLibrary[library] = [];

      // Track chunk counts per method
      const chunkCounters: Record<string, number> = {};

      for (const fileInfo of fileArr) {
        // Initialize method count for this file
        const docTitle = fileInfo?.metavars?.filename || "Untitled";

        for (const method of methods) {
          // Increment counter to ensure distinct method instances
          try {
            const formData = new FormData();
            formData.append("methodName", method.methodName);
            formData.append("library", method.library);
            formData.append("text", fileInfo.text || "");

            // Add the user settings
            if (method.settings) {
              // Extract specific settings based on the method type
              const { baseMethod } = method;
              const settingsObj = { ...method.settings };

              // Debug the settings
              console.log(
                `Sending settings for ${method.methodName}:`,
                settingsObj,
              );

              // Add all settings from the settings object
              Object.entries(settingsObj).forEach(([k, v]) => {
                // Only send relevant settings (not UI-specific ones)
                if (k !== "displayName" && k !== "emoji") {
                  formData.append(k, String(v));
                }
              });
            }

            const res = await fetch("http://localhost:5000/process", {
              method: "POST",
              body: formData,
            });

            if (!res.ok) {
              const err = await res.json();
              throw new Error(err.error || "Chunking request failed");
            }

            const json = await res.json();
            const chunks = json.chunks as string[];

            // We'll build chunk IDs for each doc
            const docTitleSafe = docTitle.replace(/\W+/g, "_");
            const methodSafe = method.methodName.replace(/\W+/g, "_");
            const libSafe = library.replace(/\W+/g, "_");
            // Compute displayed name from settings if available.
            const displayName =
              method.settings?.displayName || method.methodName;

            // Initialize the counter for this display name if nonexistent
            if (chunkCounters[displayName] === undefined) {
              chunkCounters[displayName] = 0;
            }

            chunks.forEach((cText) => {
              // Skip empty chunks
              if (!cText || cText.trim() === '') {
                console.log("Skipping empty chunk");
                return;
              }
              
              // Use the displayName-based counter
              const cId = `${displayName}_${chunkCounters[displayName]}`;
              chunkCounters[displayName]++;
              
              // Create the chunk object with displayed name in proper fields
              const chunkVar: TemplateVarInfo = {
                text: cText,
                prompt: "",
                fill_history: {
                  chunkMethod: displayName, // Explicitly set chunkMethod
                  docTitle,
                  chunkLibrary: library,
                  chunkId: cId,
                },
                llm: undefined,
                metavars: {
                  docTitle,
                  chunkLibrary: library,
                  chunkId: cId,
                  chunkMethod: displayName, // Explicitly set chunkMethod in metavars too
                },
              };

              // Log what we're creating to verify the data
              console.log(`Creating chunk with method: ${displayName}`, chunkVar);

              allChunksByLibrary[library].push(chunkVar);

              const respObj: LLMResponse = {
                uid: cId,
                prompt: `Doc: ${docTitle} | Method: ${displayName} | Chunk: ${truncateString(cId, 25)}`,
                vars: {
                  
                },
                responses: [`[Chunk ID: ${cId}]\n${cText}`],
                llm: displayName,
                metavars: {
                  docTitle,
                  chunkLibrary: library,
                  chunkId: cId,
                  chunkMethod: displayName,
                },
              };

              allResponsesByLibrary[library].push(respObj);
            });
          } catch (err: any) {
            console.error(err);
            showAlert?.(
              `Error chunking "${docTitle}" with ${method.methodName}: ${err.message}`,
            );
          }
        }
      }
    }

    // Combine results
    const allChunks = Object.values(allChunksByLibrary).flat();
    setChunks(allChunks);
    const allResponses = Object.values(allResponsesByLibrary).flat();

    // 3) Output data grouped by library
    const groupedOutput = Object.entries(allChunksByLibrary).reduce(
      (acc, [lib, chunks]) => {
        acc[lib] = chunks.map((ch) => ({
          id: ch.metavars?.chunkId,
          docTitle: ch.metavars?.docTitle,
          method: ch.fill_history?.chunkMethod,
          text: ch.text,
          chunkMethod: ch.fill_history?.chunkMethod || ch.metavars?.chunkMethod, // Explicitly include chunkMethod
          chunkId: ch.metavars?.chunkId,
          chunkLibrary: lib,
          // Include full metadata
          metavars: {
            ...(ch.metavars || {}),
            chunkMethod: ch.fill_history?.chunkMethod || ch.metavars?.chunkMethod, // Ensure chunkMethod is in metavars
          },
          fill_history: ch.fill_history,
        }));
        return acc;
      },
      {} as Record<string, any[]>,
    );

    // Pass output with enriched metadata
    const finalChunks = allChunks
      .filter(chunk => chunk.text && chunk.text.trim() !== '') // Filter out empty chunks
      .map(chunk => {
        // Ensure chunkMethod is properly set in all locations
        const chunkMethod = chunk.fill_history?.chunkMethod || chunk.metavars?.chunkMethod;
        
        // Log to verify what we're outputting
        console.log(`Outputting chunk with method: ${chunkMethod}`, {
          id: chunk.metavars?.chunkId,
          chunkMethod
        });
        
        return {
          ...chunk,
          metavars: {
            ...(chunk.metavars || {}),
            chunkMethod, // Ensure chunkMethod is correctly set
          },
          fill_history: {
            ...(chunk.fill_history || {}),
            chunkMethod, // Ensure chunkMethod is correctly set
          }
        };
      });

    setDataPropsForNode(id, {
      fields: finalChunks,
      output: groupedOutput,
    });
    pingOutputNodes(id);

    setJSONResponses(allResponses);
    setStatus(Status.READY);
  }, [
    id,
    methodItems,
    pullInputData,
    setDataPropsForNode,
    showAlert,
    pingOutputNodes,
  ]);

  const downloadChunks = useCallback(() => {
    if (chunks.length === 0) {
      showAlert?.("No chunks available to download.");
      return;
    }
    const header = "Chunk ID,Doc Title,Method Name,Text";
    const rows = chunks.map((chunk) => {
      const chunkId = chunk.metavars?.chunkId || "";
      const docTitle = chunk.metavars?.docTitle || "";
      const methodName = chunk.fill_history?.chunkMethod || "";
      const text = (chunk.text ?? "").replace(/"/g, '""');
      return `"${chunkId}","${docTitle}","${methodName}","${text}"`;
    });
    const csvContent = [header, ...rows].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", "chunks.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, [chunks, showAlert]);

  // Open inspector
  const openInspector = () => {
    if (jsonResponses.length > 0 && inspectorRef.current) {
      inspectorRef.current.trigger();
    }
  };

  return (
    <BaseNode nodeId={id} classNames="chunking-node">
      <Handle
        type="target"
        position={Position.Left}
        id="text"
        style={{ top: "50%" }}
      />

      <NodeLabel
        title={data.title || nodeDefaultTitle}
        nodeId={id}
        icon={nodeIcon}
        status={status}
        handleRunClick={runChunking}
        runButtonTooltip="Perform chunking on input text"
        extraActions={
          <button
            onClick={downloadChunks}
            title="Download Chunks"
            className="AmitSahoo45-button-3 nodrag"
            style={{
              backgroundColor: "white",
              border: "1px solid black",
              color: "black",
              padding: "0px 10px",
              borderRadius: "4px",
              cursor: "pointer",
            }}
          >
            &#9660;
          </button>
        }
      />

      <ChunkMethodListContainer
        initMethodItems={data.methods || []}
        onItemsChange={handleMethodItemsChange}
      />

      <InspectFooter
        onClick={openInspector}
        showDrawerButton={false}
        onDrawerClick={() => {
          // Do nothing
        }}
        isDrawerOpen={false}
        label={
          <>
            Inspect chunks <IconSearch size="12pt" />
          </>
        }
      />

      {/* The LLM Response Inspector */}
      <LLMResponseInspectorModal
        ref={inspectorRef}
        jsonResponses={jsonResponses}
      />

      <Handle
        type="source"
        position={Position.Right}
        id="output"
        style={{ top: "50%" }}
      />
    </BaseNode>
  );
};

export default ChunkingNode;
