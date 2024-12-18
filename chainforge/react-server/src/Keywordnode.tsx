import React, {
  useEffect,
  useState,
  useRef,
  useCallback,
  useContext,
} from "react";
import { Handle, Position } from "reactflow";
import { Progress } from "@mantine/core";
import useStore from "./store";
import BaseNode from "./BaseNode";
import NodeLabel from "./NodeLabelComponent";
import { LLMListContainer, LLMListContainerRef } from "./LLMListComponent";
import {
  ChatHistoryInfo,
  Dict,
  LLMSpec,
  LLMResponse,
  TemplateVarInfo,
  ChatMessage,
} from "./backend/typing";
import { queryLLM } from "./backend/backend";
import { AlertModalContext } from "./AlertModal";
import { Status } from "./StatusIndicatorComponent";
import LLMResponseInspectorModal, {
  LLMResponseInspectorModalRef,
} from "./LLMResponseInspectorModal";
import InspectFooter from "./InspectFooter";
import TemplateHooks, {
  extractBracketedSubstrings,
} from "./TemplateHooksComponent";
import { setsAreEqual, genDebounceFunc } from "./backend/utils";

export interface keywordNodeProps {
  data: {
    title: string;
    llms: LLMSpec[];
    vars: string[];
  };
  id: string;
  type: string;
}

const keywordnode: React.FC<keywordNodeProps> = ({
  data,
  id,
  type: node_type,
}) => {
  const node_icon = "🗒️";
  const node_default_title = "Text Keyword & Context Extractor Node";

  // Zustand state
  const setDataPropsForNode = useStore((state) => state.setDataPropsForNode);
  const pullInputData = useStore((state) => state.pullInputData);

  // State for LLMs, progress, responses, and template variables
  const llmListContainer = useRef<LLMListContainerRef>(null);
  const [llmItemsCurrState, setLLMItemsCurrState] = useState<LLMSpec[]>([]);
  const [progress, setProgress] = useState<number | undefined>(undefined);
  const [status, setStatus] = useState(Status.NONE);
  const [jsonResponses, setJSONResponses] = useState<LLMResponse[] | null>(
    null,
  );
  const [templateVars, setTemplateVars] = useState<string[]>(data.vars ?? []);

  // Display alert
  const showAlert = useContext(AlertModalContext);

  // For inspecting the responses
  const inspectModal = useRef<LLMResponseInspectorModalRef>(null);
  const [showDrawer, setShowDrawer] = useState(false);

  // Debounce helpers
  const debounceTimeoutRef = useRef(null);
  const debounce = genDebounceFunc(debounceTimeoutRef);

  // Trigger alert to indicate an error
  const triggerAlert = useCallback(
    (msg: string) => {
      setProgress(undefined);
      if (showAlert) showAlert(msg);
    },
    [showAlert],
  );

  // Handle changes to LLM list items
  const onLLMListItemsChange = useCallback(
    (new_items: LLMSpec[], old_items: LLMSpec[]) => {
      setLLMItemsCurrState(new_items);
      setDataPropsForNode(id, { llms: new_items });
    },
    [id, setDataPropsForNode],
  );

  // Refresh template hooks (variables in curly braces)
  const refreshTemplateHooks = useCallback(
    (text: string) => {
      const found_template_vars = new Set(extractBracketedSubstrings(text));
      if (!setsAreEqual(found_template_vars, new Set(templateVars))) {
        setTemplateVars(Array.from(found_template_vars));
        setDataPropsForNode(id, { vars: Array.from(found_template_vars) });
      }
    },
    [templateVars, setDataPropsForNode, id],
  );

  // Handle running the keyword and context extraction query
  const handleRunClick = () => {
    if (llmItemsCurrState.length === 0) {
      triggerAlert(
        "Please select at least one LLM to run the extraction query.",
      );
      return;
    }

    // Initialize an array to collect texts
    const texts: string[] = [];

    try {
      // Attempt to pull data
      const pulled_template_data: Dict<(string | TemplateVarInfo)[]> =
        pullInputData(templateVars.concat("text"), id);

      if (pulled_template_data.text) {
        const pulled_texts = pulled_template_data.text;
        pulled_texts.forEach((text) => {
          if (typeof text === "string") {
            texts.push(text);
          }
        });
      }

      console.log("Pulled Data:", pulled_template_data); // Add logging to ensure data is correct
    } catch (err) {
      triggerAlert("Error: Missing inputs for required template variables.");
      console.error(err);
      return; // early exit
    }

    if (texts.length === 0) {
      triggerAlert("No text provided for extraction.");
      return;
    }

    // Prepare to query the selected LLMs with each input text
    setStatus(Status.LOADING);
    setProgress(0);

    let responseCount = 0;
    const allResponses: LLMResponse[] = [];

    // Loop through each text and query the LLM for extraction
    texts.forEach((text, textIndex) => {
      // Create the prompt for each entire text
      const prompt = `Extract key concepts or keywords from the following text, focusing on phrases that are critical for understanding the whole content. Additionally, provide a brief two-sentence summary highlighting the central message of the text.\n\n"${text}"`;

      queryLLM(
        id,
        llmItemsCurrState,
        1,
        prompt,
        {},
        undefined,
        {},
        false,
        undefined,
        undefined,
        Date.now(),
      )
        .then((json) => {
          if (json?.responses) {
            allResponses.push(...json.responses);
          } else {
            triggerAlert(
              `Failed to retrieve responses for text ${textIndex + 1}`,
            );
          }

          responseCount++;
          setProgress((responseCount / texts.length) * 100);

          if (responseCount === texts.length) {
            setJSONResponses(allResponses);
            setStatus(Status.READY);
          }
        })
        .catch((err) => {
          console.error(
            `Error during LLM query for text ${textIndex + 1}:`,
            err,
          );
          setStatus(Status.ERROR);
          triggerAlert(`Error during LLM query for text ${textIndex + 1}`);
        });
    });
  };

  // Show response inspector
  const showResponseInspector = useCallback(() => {
    if (inspectModal && inspectModal.current && jsonResponses) {
      inspectModal.current?.trigger();
    }
  }, [inspectModal, jsonResponses]);

  return (
    <BaseNode classNames="keyword-node" nodeId={id}>
      <NodeLabel
        title={data.title || node_default_title}
        nodeId={id}
        icon={node_icon}
        status={status}
        isRunning={status === "loading"}
        handleRunClick={handleRunClick}
      />

      {/* Add an input handle for accepting text input from connected nodes */}
      <Handle
        type="target"
        position={Position.Left}
        id="text"
        className="grouped-handle"
        style={{ top: "50%", background: "#555" }}
      />

      <TemplateHooks
        vars={templateVars}
        nodeId={id}
        startY={160} // This sets the Y position for the input handles
        position={Position.Left}
      />

      <Handle
        type="source"
        position={Position.Right}
        id="prompt"
        className="grouped-handle"
        style={{ top: "50%" }}
      />

      <div style={{ marginTop: "10px" }}>
        <LLMListContainer
          ref={llmListContainer}
          initLLMItems={data.llms}
          onItemsChange={onLLMListItemsChange}
        />
        {progress !== undefined && (
          <Progress
            value={progress}
            color="blue"
            animate={true}
            size="sm"
            mt="sm"
          />
        )}
        {jsonResponses && jsonResponses.length > 0 && status !== "loading" ? (
          <InspectFooter
            onClick={showResponseInspector}
            isDrawerOpen={showDrawer}
            showDrawerButton={true}
            onDrawerClick={() => {
              setShowDrawer(!showDrawer);
            }}
          />
        ) : (
          <></>
        )}
      </div>

      <LLMResponseInspectorModal
        ref={inspectModal}
        jsonResponses={jsonResponses ?? []}
      />
    </BaseNode>
  );
};

export default keywordnode;
