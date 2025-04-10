import React, { useState, useEffect, useCallback, useRef } from "react";
import { Handle, Position } from "reactflow";
import { Group, Text, Button, Center, Tooltip, Loader, Popover } from "@mantine/core";
import { IconInfoCircle, IconList } from "@tabler/icons-react";
import { useDisclosure } from "@mantine/hooks";
import { v4 as uuid } from "uuid";

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
import MethodListComponent from "./RetrievalMethodListComponent";
import {
  RetrievalChunk,
  RetrievalMethodResult,
  RetrievalResults,
  RetrievalMethodSpec,
} from "./RetrievalMethodSchemas";

type Dict<T> = { [key: string]: T };

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
          <button
            className="custom-button"
            onMouseEnter={_onHover}
            onMouseLeave={close}
            onClick={onClick}
            style={{ border: "none", background: "none", padding: 0 }}
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

interface RetrievalNodeProps {
  data: {
    title?: string;
    methods?: RetrievalMethodSpec[];
    query?: string;
    refresh?: boolean;
    results?: RetrievalResults;
    vars?: string[]; // Add vars to persist template variables
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
  const [templateVars, setTemplateVars] = useState<string[]>(data.vars || []);
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
      setDataPropsForNode(id, { refresh: false, results: {}, output: [] });
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
        setDataPropsForNode(id, { vars: Array.from(foundTemplateVars) });
      }
    },
    [templateVars, id, setDataPropsForNode],
  );

  const handleQueryChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = event.target.value;
    setQueryText(value);
    refreshTemplateHooks(value);
    generateQueryPreviews(value);
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

  const buildLLMResponses = (resultsData: RetrievalResults): LLMResponse[] => {
    const arr: LLMResponse[] = [];
    
    // Process each method's results
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
      
      // Sort the retrieved chunks by similarity in descending order
      const sortedChunks = [...methodObj.retrieved].sort((a, b) => 
        b.similarity - a.similarity
      );
      
      console.log("Sorted chunks by similarity (desc):", 
        sortedChunks.map(c => ({ id: c.chunkId, sim: c.similarity }))
      );
      
      // Create LLM responses from the sorted chunks
      sortedChunks.forEach((chunk) => {
        const cUid = uuid();
        
        // Preserve original chunk method (ensure it exists)
        const originalChunkMethod = chunk.chunkMethod || "Unknown Method";
        
        arr.push({
          uid: cUid,
          prompt: `Retrieved by chunk method: ${originalChunkMethod} using ${finalLabel} [similarity: ${chunk.similarity.toFixed(3)}]`,
          vars: {
            similarity: chunk.similarity.toFixed(3),
            chunkMethod: originalChunkMethod,
            retrievalMethod: finalLabel,
          },
          responses: [` ${chunk.text}`],
          llm: finalLabel,
          metavars: {
            retrievalMethod: finalLabel,
            docTitle: chunk.docTitle,
            chunkId: chunk.chunkId,
            chunkMethod: originalChunkMethod,
            similarity: chunk.similarity,
          },
        });
      });
    });
    return arr;
  };

  const handleMethodsChange = useCallback(
    (newItems: RetrievalMethodSpec[]) => {
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
      setDataPropsForNode(id, { methods: newItems });
    },
    [id, setDataPropsForNode],
  );

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

    // Extract tabular data
    const tabularData: Dict<Dict<string>> = {};

    // Check for tabular data in the pulled data
    Object.entries(pulledData).forEach(([key, values]) => {
      if (Array.isArray(values) && values.length > 0) {
        values.forEach((value, idx) => {
          if (value && typeof value === "object") {
            // This might be tabular data
            const queryText = value.text || "";

            // Initialize the table for this query if needed
            if (!tabularData[queryText]) {
              tabularData[queryText] = {};
            }

            // Store all metavars
            if (value.metavars) {
              Object.entries(value.metavars).forEach(([metaKey, metaValue]) => {
                tabularData[queryText][metaKey] = metaValue as string;
              });
            }

            // Store all fill_history
            if (value.fill_history) {
              Object.entries(value.fill_history).forEach(
                ([histKey, histValue]) => {
                  tabularData[queryText][histKey] = histValue as string;
                },
              );
            }
          }
        });
      }
    });

    console.log("Tabular data extracted:", tabularData);

    const baseQueries = queryText.split("\n").filter((q) => q.trim());
    
    // Filter out any empty chunks from the input data
    const chunkArr = (pulledData.fields || []).filter(chunk => chunk && chunk.text && chunk.text.trim() !== '');

    if (chunkArr.length === 0 && !templateVars.length) {
      alert("No chunk data found from upstream node or template variables.");
      return;
    }

    console.log(`Found ${chunkArr.length} valid chunks for retrieval`);

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
    const newResults: RetrievalResults = {};
    const newLoading: { [key: string]: boolean } = {};
    methodItems.forEach((m) => {
      newLoading[m.key] = true;
      newResults[m.key] = {
        label: m.displayName || m.methodName,
        retrieved: [],
      };
    });
    setLoadingMethods(newLoading);

    const groupedResults: {
      [methodKey: string]: { [query: string]: RetrievalChunk[] };
    } = {};

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
          // Log the chunk data to see what we're actually sending
          console.log("Chunks being sent to retrieval:", chunkArr);
          
          const payload: any = {
            query: resolvedQuery,
            top_k: topKSetting,
            similarity_threshold: method.settings?.similarity_threshold ?? 0.7,
            chunks: chunkArr
              .filter(chunk => chunk && chunk.text && chunk.text.trim() !== '') // Additional filter for safety
              .map((chunk) => {
                // Extract the chunking method from the proper location
                const chunkMethod = 
                  chunk.metavars?.chunkMethod || 
                  chunk.fill_history?.chunkMethod || 
                  "";
                
                console.log("Processing chunk:", {
                  chunkId: chunk.metavars?.chunkId || "",
                  chunkMethod,
                  textLength: chunk.text?.length || 0
                });
                
                return {
                  text: chunk.text,
                  docTitle: chunk.metavars?.docTitle || chunk.fill_history?.docTitle || "",
                  chunkId: chunk.metavars?.chunkId || chunk.fill_history?.chunkId || "",
                  chunkMethod: chunkMethod,
                };
              }),
            custom_method_key: method.key,
          };
          
          // Log the first chunk to debug
          if (payload.chunks.length > 0) {
            console.log("Sample chunk in payload:", payload.chunks[0]);
          } else {
            console.warn("No valid chunks in payload!");
            continue; // Skip this query if no valid chunks
          }

          if (method.needsVector) {
            // Ensure we have a default vectorization library
            payload.library = method.vectorLib || "HuggingFace Transformers";
            payload.type = "vectorization";
            payload.method = method.baseMethod || "cosine";
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
                // Set a default library for non-vector methods to prevent "Unknown vector library: None" error
                payload.library = "KeywordOverlap";
                payload.type = "retrieval";
                break;
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

          uniqueChunks.forEach((chunk: any) => {
            groupedResults[method.key][resolvedQuery].push({
              ...chunk,
              text: `${chunk.text}`,
              query: resolvedQuery,
            });
          });
        }

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

      Object.entries(methodQueries).forEach(([query, chunks]) => {
        if (chunks.length === 0) return;

        // Find any tabular data for this query
        const queryTabularData = tabularData[query] || {};

        chunks.forEach((chunk) => {
          // Extract and preserve original chunk method
          const originalChunkMethod = chunk.chunkMethod || "";
          
          outputChunks.push({
            text: chunk.text,
            prompt: query,
            fill_history: {
              methodId: methodKey,
              embeddingModel:
                method.vectorLib && method.settings?.openai_model
                  ? method.settings.openai_model
                  : undefined,
              chunkMethod: originalChunkMethod, // Preserve original chunking method
              retrievalMethod: methodLabel, // Add retrieval method separately
              ...queryTabularData, // Add all tabular data to fill_history
            },
            metavars: {
              method: methodLabel,
              baseMethod: method.baseMethod,
              similarity: chunk.similarity.toString(),
              docTitle: chunk.docTitle || "",
              chunkId: chunk.chunkId || "",
              chunkMethod: originalChunkMethod, // Preserve original chunking method
              retrievalMethod: methodLabel, // Add retrieval method as a separate field
              methodGroup: methodKey,
              queryGroup: query,
              ...queryTabularData, // Add all tabular data to metavars
            },
            llm: methodLabel,
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
      vars: templateVars,
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
      vars: templateVars,
    });
    refreshTemplateHooks(queryText);
    generateQueryPreviews(queryText);
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
              minHeight: "40px",
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
        <MethodListComponent
          initMethodItems={methodItems}
          onItemsChange={handleMethodsChange}
          loadingMethods={loadingMethods}
        />
      </div>
      <InspectFooter
        onClick={() => inspectorModalRef.current?.trigger()}
        showDrawerButton={false}
        onDrawerClick={() => {
          // sss
        }}
        isDrawerOpen={false}
      />
      <Handle
        type="source"
        position={Position.Right}
        id="output"
        style={{ top: "50%" }}
      />
      <React.Suspense fallback={<Loader />}>
        <LLMResponseInspectorModal
          ref={inspectorModalRef}
          jsonResponses={jsonResponses}
        />
      </React.Suspense>
    </BaseNode>
  );
};

export default RetrievalNode;
