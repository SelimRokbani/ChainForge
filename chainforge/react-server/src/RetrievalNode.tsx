import React, {
  useState,
  useEffect,
  useCallback,
  useRef,
  useContext,
} from "react";
import { Handle, Position } from "reactflow";
import { LoadingOverlay, Textarea } from "@mantine/core";
import { IconSearch } from "@tabler/icons-react";

import BaseNode from "./BaseNode";
import NodeLabel from "./NodeLabelComponent";
import useStore from "./store";
import InspectFooter from "./InspectFooter";
import { AlertModalContext } from "./AlertModal";
import AreYouSureModal, { AreYouSureModalRef } from "./AreYouSureModal";

import LLMResponseInspectorModal, {
  LLMResponseInspectorModalRef,
} from "./LLMResponseInspectorModal";
import RetrievalMethodListContainer, {
  RetrievalMethodSpec,
} from "./RetrievalMethodListComponent";
import { LLMResponse, TemplateVarInfo } from "./backend/typing";
import TemplateHooks, {
  extractBracketedSubstrings,
} from "./TemplateHooksComponent";
import { setsAreEqual } from "./backend/utils";

interface RetrievalNodeProps {
  id: string;
  data: {
    title?: string;
    query?: string;
    methods?: RetrievalMethodSpec[];
    results?: Record<string, any>;
    refresh?: boolean;
    vars?: string[]; // Add this line to store template variables
  };
}

const RetrievalNode: React.FC<RetrievalNodeProps> = ({ id, data }) => {
  const nodeDefaultTitle = "Retrieval Node";
  const nodeIcon = "🔍";

  // Store hooks
  const pullInputData = useStore((s) => s.pullInputData);
  const setDataPropsForNode = useStore((s) => s.setDataPropsForNode);
  const pingOutputNodes = useStore((s) => s.pingOutputNodes);

  // Context
  const showAlert = useContext(AlertModalContext);

  // State
  const [query, setQuery] = useState<string>(data.query || "");
  const [templateVars, setTemplateVars] = useState<string[]>(data.vars || []);
  const [methodItems, setMethodItems] = useState<RetrievalMethodSpec[]>(
    data.methods || [],
  );
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<Record<string, any>>(
    data.results || {},
  );
  const [jsonResponses, setJsonResponses] = useState<LLMResponse[]>([]);

  // Refs
  const inspectorModalRef = useRef<LLMResponseInspectorModalRef>(null);
  const textAreaRef = useRef<HTMLTextAreaElement | null>(null);
  const retrievalConfirmModalRef = useRef<AreYouSureModalRef>(null);
  const [hooksY, setHooksY] = useState(138);

  // Reset on refresh
  useEffect(() => {
    if (data.refresh) {
      setDataPropsForNode(id, {
        refresh: false,
        results: {},
        output: [],
      });
      setResults({});
      setJsonResponses([]);
    }
  }, [data.refresh, id, setDataPropsForNode]);

  // Template variables handling
  const refreshTemplateHooks = useCallback(
    (text: string) => {
      // Extract template variables (strings within {} that aren't escaped)
      const foundTemplateVars = new Set(extractBracketedSubstrings(text));

      if (!setsAreEqual(foundTemplateVars, new Set(templateVars))) {
        setTemplateVars(Array.from(foundTemplateVars));
        setDataPropsForNode(id, { vars: Array.from(foundTemplateVars) });
      }
    },
    [templateVars, id, setDataPropsForNode],
  );

  // Handle query text changes
  const handleQueryChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = event.currentTarget.value;
    setQuery(value);

    // Debounce refreshing template hooks to avoid too many updates
    const timeoutId = setTimeout(() => {
      refreshTemplateHooks(value);
    }, 500);

    return () => clearTimeout(timeoutId);
  };

  // Handle method changes
  const handleMethodsChange = useCallback(
    (newItems: RetrievalMethodSpec[]) => {
      setMethodItems(newItems);
      setDataPropsForNode(id, { methods: newItems });
    },
    [id, setDataPropsForNode],
  );

  // Build LLM responses for inspector
  const buildLLMResponses = useCallback(
    (results: Record<string, any>): LLMResponse[] => {
      const responses: LLMResponse[] = [];

      Object.entries(results).forEach(([methodKey, result]) => {
        // Handle the nested structure of retrieved results
        if (result.retrieved && typeof result.retrieved === "object") {
          // For each query in the retrieved results
          Object.entries(result.retrieved).forEach(
            ([queryKey, queryResults]) => {
              // Make sure queryResults is an array before proceeding
              if (!Array.isArray(queryResults)) return;

              responses.push({
                uid: `${methodKey}-${queryKey}-${Date.now()}`,
                prompt: queryKey, // Use the query key as the prompt
                vars: {},
                metavars: {
                  method: methodKey,
                  query: queryKey,
                  ...result.metavars,
                },
                responses: queryResults.map(
                  (item: any) =>
                    `[Score: ${item.similarity?.toFixed(4) || "N/A"}] ${item.text || item.content || JSON.stringify(item)}`,
                ),
                llm: result.metavars.method,
              });
            },
          );
        } else if (result.error) {
          // Handle error case
          responses.push({
            uid: `${methodKey}-error-${Date.now()}`,
            prompt: query,
            vars: {},
            metavars: {
              method: methodKey,
              error: result.error,
            },
            responses: [`Error: ${result.error}`],
            llm: methodKey,
          });
        }
      });

      return responses;
    },
    [results],
  );

  // Main retrieval function
  const confirmAndRunRetrieval = () => {
    retrievalConfirmModalRef.current?.trigger();
  };

  const runRetrieval = useCallback(async () => {
    if (methodItems.length === 0) {
      showAlert?.("Please add at least one retrieval method");
      return;
    }

    setLoading(true);

    try {
      // Get input data from connected nodes
      const inputData = pullInputData(["chunks", "queries"], id) as {
        chunks?: any[];
        queries?: any[];
      };

      // Format methods for the API request
      const formattedMethods = methodItems.map((method) => ({
        id: method.key,
        baseMethod: method.baseMethod,
        methodName: method.methodName,
        library: method.library,
        embeddingProvider: method.embeddingProvider,
        settings: method.settings || {},
      }));

      if (!inputData.chunks || !inputData.queries) {
        throw new Error("No input chunks or queries found");
      }

      // Make the API request
      const response = await fetch("http://localhost:5001/retrieve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          methods: formattedMethods,
          chunks: inputData.chunks,
          queries: inputData.queries,
        }),
      });

      if (!response.ok) {
        throw new Error(`Retrieval failed: ${response.statusText}`);
      }
      // Define types for the API response
      interface RetrievalMethodResult {
        baseMethod: string;
        methodName: string;
        retrieved: any[];
        status: "success" | "error";
        error?: string;
        vectorstore_status?: string;
        embeddingModel?: string;
      }

      interface RetrievalResponse {
        results: Record<string, RetrievalMethodResult>;
      }

      const responseData = (await response.json()) as RetrievalResponse;

      // Process the results from the response
      const allResults: Record<string, any> = {};

      if (responseData.results) {
        for (const [methodId, result] of Object.entries(responseData.results)) {
          const method = methodItems.find((m) => m.key === methodId);
          if (!method) continue;

          if (result.status === "success") {
            allResults[methodId] = {
              retrieved: result.retrieved || [],
              metavars: {
                method: method.methodName,
                library: method.library,
                embeddingModel: method.needsEmbeddingModel,
                vectorstoreStatus: result.vectorstore_status,
              },
            };
          } else {
            allResults[methodId] = {
              error: result.error || "Unknown error",
              retrieved: [],
            };
          }
        }
      }
      console.log(allResults);
      // Update results
      setResults(allResults);
      setJsonResponses(buildLLMResponses(allResults));

      // Prepare output chunks (combined from all methods)
      const outputChunks = Object.values(allResults)
        .flatMap((result: any) => (result.retrieved ? result.retrieved : []))
        .sort((a: any, b: any) => b.similarity - a.similarity)
        .filter(
          (chunk: any, index: number, self: any[]) =>
            index === self.findIndex((c) => c.chunkId === chunk.chunkId),
        )
        .slice(0, 10);

      // Update node data
      setDataPropsForNode(id, {
        query,
        methods: methodItems,
        results: allResults,
        output: outputChunks,
        vars: templateVars,
      });

      // Notify downstream nodes
      pingOutputNodes(id);
    } catch (error) {
      console.error("Detailed error:", error);
      showAlert?.(error instanceof Error ? error.message : "Retrieval failed");
    } finally {
      setLoading(false);
    }
  }, [
    query,
    methodItems,
    templateVars,
    id,
    pullInputData,
    setDataPropsForNode,
    pingOutputNodes,
    showAlert,
    buildLLMResponses,
  ]);

  // Update stored data when query or methods change
  useEffect(() => {
    setDataPropsForNode(id, {
      query,
      methods: methodItems,
      results,
      vars: templateVars,
    });
  }, [id, query, methodItems, results, templateVars, setDataPropsForNode]);

  // Initialize template variables on first load
  useEffect(() => {
    refreshTemplateHooks(query);
  }, []);

  // Set up ResizeObserver for dynamic textarea height
  const setRef = useCallback((elem: HTMLTextAreaElement | null) => {
    if (!elem) return;

    textAreaRef.current = elem;

    // Set up ResizeObserver to adjust template hooks position when textarea resizes
    if (window.ResizeObserver) {
      const observer = new window.ResizeObserver(() => {
        if (textAreaRef.current) {
          // Adjust hooksY based on the textarea height
          setHooksY(textAreaRef.current.clientHeight + 80);
        }
      });

      observer.observe(elem);
      return () => observer.disconnect();
    }
  }, []);

  return (
    <BaseNode
      nodeId={id}
      classNames="retrieval-node"
      style={{ width: "400px", backgroundColor: "rgba(255,255,255,0.9)" }}
    >
      <NodeLabel
        title={data.title || nodeDefaultTitle}
        nodeId={id}
        icon={nodeIcon}
        status={undefined}
        handleRunClick={confirmAndRunRetrieval}
        runButtonTooltip="Run Retrieval"
      />

      <div style={{ padding: 24, position: "relative" }}>
        <LoadingOverlay visible={loading} />
      </div>
      <Handle
        type="target"
        position={Position.Left}
        id="queries"
        style={{ top: "45%", left: "0px", transform: "translate(-50%, -50%)" }}
      />
      <Handle
        type="target"
        position={Position.Left}
        id="chunks"
        style={{ top: "55%", left: "0px", transform: "translate(-50%, -50%)" }}
      />
      <RetrievalMethodListContainer
        initMethodItems={methodItems}
        onItemsChange={handleMethodsChange}
      />

      <InspectFooter
        onClick={() => inspectorModalRef.current?.trigger()}
        showDrawerButton={false}
        onDrawerClick={() => undefined}
        isDrawerOpen={false}
        label={
          <>
            Inspect results <IconSearch size="12pt" />
          </>
        }
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
      <AreYouSureModal
        ref={retrievalConfirmModalRef}
        title="Confirm Retrieval"
        message={`⚠️ You're about to run all configured retrieval methods.\n\n
          Some methods may create, load, or modify vector stores, which could:\n 
          Overwrite existing data\n or append new data.
          Make sure your settings and input data are correct before proceeding.`}
        onConfirm={runRetrieval}
      />
    </BaseNode>
  );
};
export default RetrievalNode;
