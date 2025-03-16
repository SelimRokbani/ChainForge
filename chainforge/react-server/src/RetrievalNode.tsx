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

  // Main retrieval function
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
      // Normalize queries to handle both string arrays and object arrays
      const normalizedQueries = inputData.queries.map((query) => {
        // If query is an object with a text property, extract it
        if (typeof query === "object" && query !== null && "text" in query) {
          return query.text;
        }
        // Otherwise, return as is (assuming it's a string)
        return query;
      });
      // Make the API request
      const response = await fetch("http://localhost:5000/retrieve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          methods: formattedMethods,
          chunks: inputData.chunks,
          queries: normalizedQueries,
        }),
      });

      if (!response.ok) {
        throw new Error(`Retrieval failed: ${response.statusText}`);
      }

      // The response is now a flat array of objects
      const retrievalResults = await response.json();

      // Convert to proper LLMResponse objects
      const llmResponses: LLMResponse[] = retrievalResults.map(
        (result: any) => ({
          uid: result.uid || `retrieval-${Date.now()}-${Math.random()}`,
          prompt: result.prompt,
          vars: result.vars || {},
          metavars: result.metavars || {},
          responses: [result.text],
          llm: result.llm || "retrieval",
        }),
      );

      // Set the responses for the inspector
      setJsonResponses(llmResponses);

      // Group results by method for the node's internal state
      const resultsByMethod: Record<string, any> = {};

      // Process each result to organize by method
      retrievalResults.forEach((result: any) => {
        const methodId = result.fill_history?.methodId;
        const methodName = result.metavars?.method;

        if (!resultsByMethod[methodId]) {
          resultsByMethod[methodId] = {
            retrieved: {},
            metavars: {
              method: methodName,
              baseMethod: result.metavars?.baseMethod,
              embeddingModel: result.fill_history?.embeddingModel,
            },
          };
        }

        // Group by query
        const query = result.prompt;
        if (!resultsByMethod[methodId].retrieved[query]) {
          resultsByMethod[methodId].retrieved[query] = [];
        }

        // Add this result to the appropriate query group
        resultsByMethod[methodId].retrieved[query].push({
          text: result.text,
          similarity: result.metavars?.similarity,
          docTitle: result.metavars?.docTitle,
          chunkId: result.metavars?.chunkId,
        });
      });

      // Update results state
      setResults(resultsByMethod);

      // Prepare output chunks (combined from all methods)
      const outputChunks = retrievalResults
        .sort(
          (a: any, b: any) =>
            (b.metavars?.similarity || 0) - (a.metavars?.similarity || 0),
        )
        .filter(
          (result: any, index: number, self: any[]) =>
            index ===
            self.findIndex(
              (r: any) => r.metavars?.chunkId === result.metavars?.chunkId,
            ),
        )
        .slice(0, 10)
        .map((result: any) => ({
          text: result.text,
          similarity: result.metavars?.similarity || 0,
          docTitle: result.metavars?.docTitle || "",
          chunkId: result.metavars?.chunkId || "",
          method: result.metavars?.method || "",
        }));

      const outputForDownstream: TemplateVarInfo[] = retrievalResults.map(
        (result: any) => ({
          text: result.text,
          prompt: result.prompt,
          fill_history: result.fill_history || {},
          metavars: result.metavars || {},
          llm: result.llm,
          uid: result.uid || `chunk-${Date.now()}-${Math.random()}`,
        }),
      );

      // Update node data
      setDataPropsForNode(id, {
        query,
        methods: methodItems,
        results: resultsByMethod,
        output: outputForDownstream,
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
        handleRunClick={runRetrieval}
        runButtonTooltip="Run Retrieval"
      />

      <div style={{ padding: 8, position: "relative" }}>
        <LoadingOverlay visible={loading} />

        <Textarea
          ref={setRef}
          label="Search Query"
          placeholder="Enter your query..."
          className="prompt-field-fixed nodrag nowheel"
          autosize
          minRows={4}
          maxRows={12}
          value={query}
          onChange={handleQueryChange}
          mb="sm"
        />
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
    </BaseNode>
  );
};
export default RetrievalNode;
