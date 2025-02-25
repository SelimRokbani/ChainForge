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
import LLMResponseInspectorModal, {
  LLMResponseInspectorModalRef,
} from "./LLMResponseInspectorModal";
import RetrievalMethodListContainer, {
  RetrievalMethodSpec,
} from "./RetrievalMethodListComponent";
import { LLMResponse, TemplateVarInfo } from "./backend/typing";

interface RetrievalNodeProps {
  id: string;
  data: {
    title?: string;
    query?: string;
    methods?: RetrievalMethodSpec[];
    results?: Record<string, any>;
    refresh?: boolean;
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
      return Object.entries(results).map(([methodKey, result]) => ({
        uid: `${methodKey}-${Date.now()}`,
        prompt: query,
        vars: {},
        metavars: {
          method: methodKey,
          ...result.metavars,
        },
        responses: result.retrieved.map(
          (item: any) => `[Score: ${item.similarity.toFixed(4)}] ${item.text}`,
        ),
        llm: methodKey,
      }));
    },
    [query],
  );

  // Helper to build request body
  const buildRequestBody = useCallback(
    (query: string, chunks: TemplateVarInfo[], method: RetrievalMethodSpec) => {
      const baseRequest = {
        query,
        chunks,
        top_k: method.settings?.top_k || 5,
      };

      // Check if it's a keyword-based method
      const keywordMethods = ["bm25", "tfidf", "boolean", "overlap"];
      if (keywordMethods.includes(method.baseMethod)) {
        return {
          ...baseRequest,
          method_type: "retrieval",
          library: method.library,
        };
      }

      // It's a vector-based method
      const request: any = {
        ...baseRequest,
        method_type: "vectorization",
        library: method.embeddingModel,
        method: method.baseMethod,
      };

      // Add FAISS settings if it's a FAISS method
      if (method.baseMethod === "faiss") {
        request.vectorStore = {
          type: "faiss",
          mode: method.settings?.mode || "create",
          path: method.settings?.path || "",
        };
      }

      return request;
    },
    [],
  );

  // Main retrieval function
  const runRetrieval = useCallback(async () => {
    if (!query.trim()) {
      showAlert?.("Please enter a search query");
      return;
    }

    if (methodItems.length === 0) {
      showAlert?.("Please add at least one retrieval method");
      return;
    }

    setLoading(true);

    try {
      // Get input chunks
      const inputData = pullInputData(["fields"], id) as {
        fields?: TemplateVarInfo[];
      };
      console.log("Input Data:", inputData);
      console.log("Fields:", inputData.fields);

      if (!inputData.fields || inputData.fields.length === 0) {
        throw new Error("No input chunks found");
      }

      // Log method items to verify they're structured correctly
      console.log("Method Items:", methodItems);

      // Process each method
      const allResults: Record<string, any> = {};

      for (const method of methodItems) {
        // Skip if embedding model is needed but not selected
        if (method.needsEmbeddingModel && !method.embeddingModel) {
          continue;
        }

        try {
          const requestBody = buildRequestBody(query, inputData.fields, method);

          const response = await fetch("http://localhost:5000/retrieve", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestBody),
          });

          if (!response.ok) {
            throw new Error(`Retrieval failed: ${response.statusText}`);
          }

          const data = await response.json();

          // Handle both standard and FAISS responses
          allResults[method.key] = {
            retrieved: data.retrieved || [], // Ensure we always have an array
            metavars: {
              method: method.methodName,
              library: method.library,
              embeddingModel: method.embeddingModel,
              vectorstoreStatus: data.vectorstore_status, // Include FAISS status if present
            },
          };
        } catch (error) {
          console.error(`Error with method ${method.methodName}:`, error);
          allResults[method.key] = {
            error: error instanceof Error ? error.message : "Unknown error",
          };
        }
      }

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
    id,
    pullInputData,
    buildRequestBody,
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
    });
  }, [id, query, methodItems, results, setDataPropsForNode]);

  return (
    <BaseNode
      nodeId={id}
      classNames="retrieval-node"
      style={{ backgroundColor: "rgba(255,255,255,0.9)" }}
    >
      <Handle type="target" position={Position.Left} id="fields" />

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

        <Textarea
          label="Search Query"
          placeholder="Enter your query..."
          className="prompt-field-fixed nodrag nowheel"
          autosize
          minRows={4}
          maxRows={12}
          value={query}
          onChange={(e) => setQuery(e.currentTarget.value)}
          mb="sm"
        />

        <RetrievalMethodListContainer
          initMethodItems={methodItems}
          onItemsChange={handleMethodsChange}
        />
      </div>

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
