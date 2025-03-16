import React, { useState, useEffect, useCallback } from "react";
import { Handle, Position } from "reactflow";
import { v4 as uuid } from "uuid";
import useStore from "./store";
import BaseNode from "./BaseNode";
import NodeLabel from "./NodeLabelComponent";
import { IconArrowMerge, IconList } from "@tabler/icons-react";
import {
  Divider,
  NativeSelect,
  Text,
  Popover,
  Tooltip,
  Center,
  Modal,
  Box,
  MultiSelect,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { escapeBraces } from "./backend/template";
import {
  countNumLLMs,
  toStandardResponseFormat,
  tagMetadataWithLLM,
  extractLLMLookup,
  removeLLMTagFromMetadata,
  groupResponsesBy,
  getVarsAndMetavars,
  cleanMetavarsFilterFunc,
} from "./backend/utils";
import StorageCache from "./backend/cache";
import { ResponseBox } from "./ResponseBoxes";
import {
  Dict,
  JSONCompatible,
  LLMResponsesByVarDict,
  TemplateVarInfo,
} from "./backend/typing";
import { generatePrompts } from "./backend/backend";

enum JoinFormat {
  DubNewLine = "\n\n",
  NewLine = "\n",
  DashedList = "-",
  NumList = "1.",
  PyArr = "[]",
}

const formattingOptions = [
  { value: JoinFormat.DubNewLine, label: "double newline \\n\\n" },
  { value: JoinFormat.NewLine, label: "newline \\n" },
  { value: JoinFormat.DashedList, label: "- dashed list" },
  { value: JoinFormat.NumList, label: "1. numbered list" },
  { value: JoinFormat.PyArr, label: '["list", "of", "strings"]' },
];

const joinTexts = (texts: string[], format: JoinFormat): string => {
  const escaped_texts = texts.map((t) => escapeBraces(t));

  if (format === JoinFormat.DubNewLine || format === JoinFormat.NewLine)
    return escaped_texts.join(format);
  else if (format === JoinFormat.DashedList)
    return escaped_texts.map((t) => "- " + t).join("\n");
  else if (format === JoinFormat.NumList)
    return escaped_texts.map((t, i) => `${i + 1}. ${t}`).join("\n");
  else if (format === JoinFormat.PyArr) return JSON.stringify(escaped_texts);

  console.error(`Could not join: Unknown formatting option: ${format}`);
  return escaped_texts[0];
};

const DEFAULT_GROUPBY_VAR_ALL = { label: "all text", value: "A" };

const displayJoinedTexts = (
  textInfos: (TemplateVarInfo | string)[],
  getColorForLLM: (llm_name: string) => string,
) => {
  const color_for_llm = (llm_name: string) => getColorForLLM(llm_name) + "99";
  return textInfos.map((info, idx) => {
    const llm_name =
      typeof info !== "string"
        ? typeof info.llm === "string"
          ? info.llm
          : info.llm?.name
        : "";
    const ps = (
      <pre className="small-response">
        {typeof info === "string" ? info : info.text}
      </pre>
    );
    return (
      <ResponseBox
        key={"r" + idx}
        boxColor={
          typeof info !== "string" && info.llm && llm_name
            ? color_for_llm(llm_name)
            : "#ddd"
        }
        width="100%"
        vars={typeof info === "string" ? {} : info.fill_history ?? {}}
        truncLenForVars={72}
        llmName={llm_name ?? ""}
      >
        {ps}
      </ResponseBox>
    );
  });
};

interface JoinedTextsPopoverProps {
  textInfos: (TemplateVarInfo | string)[];
  onHover: () => void;
  onClick: () => void;
  getColorForLLM: (llm_name: string) => string;
}

const JoinedTextsPopover: React.FC<JoinedTextsPopoverProps> = ({
  textInfos,
  onHover,
  onClick,
  getColorForLLM,
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
      key="query-info"
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
        <Tooltip label="Click to view all joined inputs" withArrow>
          <button
            className="custom-button"
            onMouseEnter={_onHover}
            onMouseLeave={close}
            onClick={onClick}
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
        <Center>
          <Text size="xs" fw={500} color="#666">
            Preview of joined inputs ({textInfos?.length} total)
          </Text>
        </Center>
        {displayJoinedTexts(textInfos, getColorForLLM)}
      </Popover.Dropdown>
    </Popover>
  );
};

export interface JoinNodeProps {
  data: {
    input: JSONCompatible;
    title: string;
    refresh: boolean;
  };
  id: string;
}

const JoinNode: React.FC<JoinNodeProps> = ({ data, id }) => {
  const [joinedTexts, setJoinedTexts] = useState<(TemplateVarInfo | string)[]>(
    [],
  );

  // For an info pop-up that previews all the joined inputs
  const [infoModalOpened, { open: openInfoModal, close: closeInfoModal }] =
    useDisclosure(false);

  const [pastInputs, setPastInputs] = useState<JSONCompatible>([]);
  const pullInputData = useStore((state) => state.pullInputData);
  const setDataPropsForNode = useStore((state) => state.setDataPropsForNode);

  // Global lookup for what color to use per LLM
  const getColorForLLMAndSetIfNotFound = useStore(
    (state) => state.getColorForLLMAndSetIfNotFound,
  );

  const [inputHasLLMs, setInputHasLLMs] = useState(false);

  // Changed: Store available variables as objects for MultiSelect
  const [groupByVarOptions, setGroupByVarOptions] = useState([
    DEFAULT_GROUPBY_VAR_ALL,
  ]);
  // Changed: Use array to store multiple selections
  const [selectedGroupVars, setSelectedGroupVars] = useState<string[]>(["A"]);

  const [groupByLLM, setGroupByLLM] = useState("within");
  const [formatting, setFormatting] = useState(formattingOptions[0].value);

  // New recursive grouping function similar to LLMResponseInspector
  const groupAndJoinByVars = useCallback(
    (
      items: TemplateVarInfo[],
      varnames: string[],
      eatenvars: string[] = [],
    ): (TemplateVarInfo | string)[] => {
      if (items.length === 0) return [];

      // Base case: no more variables to group by, join the texts
      if (varnames.length === 0) {
        // Join all texts in this group according to formatting option
        const joined_text = joinTexts(
          items.map((item) =>
            typeof item === "string" ? item : item.text ?? "",
          ),
          formatting,
        );

        // If all items have the same LLM, preserve it
        const llm = countNumLLMs(items) === 1 ? items[0].llm : undefined;

        // Create a new TemplateVarInfo with the joined text and preserved metadata
        return [
          {
            text: joined_text,
            fill_history: {},
            metavars: {},
            llm,
            uid: uuid(),
          },
        ];
      }

      // Get the current variable to group by
      const currentVar = varnames[0];
      const isMetavar = currentVar[0] === "M";
      const varname = currentVar.substring(1);

      // Group items by the current variable
      const [groupedItems, ungroupedItems] = groupResponsesBy(
        items,
        isMetavar
          ? (r) => (r.metavars ? r.metavars[varname] : undefined)
          : (r) => (r.fill_history ? r.fill_history[varname] : undefined),
      );

      // Process each group recursively
      const result: (TemplateVarInfo | string)[] = [];

      // Add groups for each value of the current variable
      Object.entries(groupedItems).forEach(([value, groupItems]) => {
        // Recursively process this group with remaining variables
        const groupResult = groupAndJoinByVars(groupItems, varnames.slice(1), [
          ...eatenvars,
          varname,
        ]);

        // Add metadata about this grouping level to the result items
        groupResult.forEach((item) => {
          if (typeof item !== "string") {
            // Add the current variable and its value to the appropriate metadata field
            if (isMetavar) {
              item.metavars = { ...item.metavars, [varname]: value };
            } else {
              item.fill_history = { ...item.fill_history, [varname]: value };
            }
          }
          result.push(item);
        });
      });

      // Handle ungrouped items if any
      if (ungroupedItems.length > 0) {
        const ungroupedResults = groupAndJoinByVars(
          ungroupedItems,
          varnames.slice(1),
          [...eatenvars, varname],
        );
        result.push(...ungroupedResults);
      }

      return result;
    },
    [formatting],
  );

  const handleOnConnect = useCallback(() => {
    let input_data: LLMResponsesByVarDict = pullInputData(["__input"], id);
    if (!input_data?.__input) {
      // soft fail
      return;
    }

    // Find all vars and metavars in the input data (if any):
    const { vars, metavars } = getVarsAndMetavars(input_data);

    // Create lookup table for LLMs in input, indexed by llm key
    const llm_lookup = extractLLMLookup(input_data);

    // Refresh the dropdown list with available vars/metavars:
    setGroupByVarOptions(
      [DEFAULT_GROUPBY_VAR_ALL]
        .concat(
          vars.map((varname) => ({
            label: `by ${varname}`,
            value: `V${varname}`,
          })),
        )
        .concat(
          metavars.filter(cleanMetavarsFilterFunc).map((varname) => ({
            label: `by ${varname} (meta)`,
            value: `M${varname}`,
          })),
        ),
    );

    // Check whether more than one LLM is present in the inputs:
    const numLLMs = countNumLLMs(input_data);
    setInputHasLLMs(numLLMs > 1);

    // Tag all response objects in the input data with a metavar for their LLM (using the llm key as a uid)
    input_data = tagMetadataWithLLM(input_data);

    // Generate (flatten) the inputs, which could be recursively chained templates
    // and a mix of LLM resp objects, templates, and strings
    generatePrompts(
      "{__input}",
      input_data as Dict<(TemplateVarInfo | string)[]>,
    )
      .then((promptTemplates) => {
        // Convert the templates into response objects
        const resp_objs = promptTemplates.map(
          (p) =>
            ({
              text: p.toString(),
              fill_history: p.fill_history,
              llm:
                "__LLM_key" in p.metavars
                  ? llm_lookup[p.metavars.__LLM_key]
                  : undefined,
              metavars: removeLLMTagFromMetadata(p.metavars),
              uid: uuid(),
            }) as TemplateVarInfo,
        );

        let joined_texts: (TemplateVarInfo | string)[] = [];

        // If all selected is the only option or we're just grouping by "all text"
        if (
          selectedGroupVars.length === 0 ||
          (selectedGroupVars.length === 1 && selectedGroupVars[0] === "A")
        ) {
          // Simple join all texts
          let joined_text: string | TemplateVarInfo = joinTexts(
            resp_objs.map((r) => (typeof r === "string" ? r : r.text ?? "")),
            formatting,
          );

          // If there is exactly 1 LLM and it's present across all inputs, keep track of it
          if (numLLMs === 1 && resp_objs.every((r) => r.llm !== undefined))
            joined_text = {
              text: joined_text,
              fill_history: {},
              llm: resp_objs[0].llm,
              uid: uuid(),
            };

          joined_texts = [joined_text];
        } else {
          // Multiple grouping variables selected

          // If there's multiple LLMs and groupByLLM is 'within', first group by LLM
          if (numLLMs > 1 && groupByLLM === "within") {
            const [groupedRespsByLLM, nonLLMRespGroup] = groupResponsesBy(
              resp_objs,
              (r) => (typeof r.llm === "string" ? r.llm : r.llm?.key),
            );

            // For each LLM group, apply the variable grouping
            Object.values(groupedRespsByLLM).forEach((llmGroup) => {
              const grouped = groupAndJoinByVars(
                llmGroup,
                selectedGroupVars.filter((v) => v !== "A"),
              );
              joined_texts = joined_texts.concat(grouped);
            });

            // Handle items without LLM
            if (nonLLMRespGroup.length > 0) {
              const text = joinTexts(
                nonLLMRespGroup.map((t) => t.text ?? ""),
                formatting,
              );
              joined_texts.push(text);
            }
          } else {
            // Group across LLMs by the selected variables
            joined_texts = groupAndJoinByVars(
              resp_objs,
              selectedGroupVars.filter((v) => v !== "A"),
            );
          }
        }

        setJoinedTexts(joined_texts);
        setDataPropsForNode(id, { fields: joined_texts });
      })
      .catch(console.error);
  }, [
    formatting,
    pullInputData,
    selectedGroupVars,
    groupByLLM,
    groupAndJoinByVars,
    id,
    setDataPropsForNode,
  ]);

  if (data.input) {
    // If there's a change in inputs...
    if (data.input !== pastInputs) {
      setPastInputs(data.input);
      handleOnConnect();
    }
  }

  // Refresh join output anytime the dropdowns change
  useEffect(() => {
    handleOnConnect();
  }, [selectedGroupVars, groupByLLM, formatting, handleOnConnect]);

  // Store the outputs to the cache whenever they change
  useEffect(() => {
    StorageCache.store(`${id}.json`, joinedTexts.map(toStandardResponseFormat));
  }, [joinedTexts, id]);

  useEffect(() => {
    if (data.refresh && data.refresh === true) {
      // Recreate the visualization:
      setDataPropsForNode(id, { refresh: false });
      handleOnConnect();
    }
  }, [data, id, handleOnConnect, setDataPropsForNode]);

  return (
    <BaseNode classNames="join-node" nodeId={id}>
      <NodeLabel
        title={data.title || "Join Node"}
        nodeId={id}
        icon={<IconArrowMerge size="12pt" />}
        customButtons={[
          <JoinedTextsPopover
            key="joined-text-previews"
            textInfos={joinedTexts}
            onHover={handleOnConnect}
            onClick={openInfoModal}
            getColorForLLM={getColorForLLMAndSetIfNotFound}
          />,
        ]}
      />
      <Modal
        title={"List of joined inputs (" + joinedTexts.length + " total)"}
        size="xl"
        opened={infoModalOpened}
        onClose={closeInfoModal}
        styles={{
          header: { backgroundColor: "#FFD700" },
          root: { position: "relative", left: "-5%" },
        }}
      >
        <Box m="lg" mt="xl">
          {displayJoinedTexts(joinedTexts, getColorForLLMAndSetIfNotFound)}
        </Box>
      </Modal>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "left",
          maxWidth: "100%",
          marginBottom: "10px",
        }}
      >
        <Text mb="xs">Group by (in order):</Text>
        <MultiSelect
          data={groupByVarOptions}
          value={selectedGroupVars}
          onChange={setSelectedGroupVars}
          className="nodrag nowheel"
          placeholder="Select grouping variables"
          size="xs"
          clearable
          searchable
          mb="sm"
        />
      </div>
      {inputHasLLMs ? (
        <div
          style={{
            display: "flex",
            justifyContent: "left",
            maxWidth: "100%",
            marginBottom: "10px",
          }}
        >
          <NativeSelect
            onChange={(e) => setGroupByLLM(e.target.value)}
            className="nodrag nowheel"
            data={["within", "across"]}
            size="xs"
            value={groupByLLM}
            maw="80px"
            mr="xs"
          />
          <Text mt="3px">LLMs</Text>
        </div>
      ) : (
        <></>
      )}
      <Divider my="xs" label="formatting" labelPosition="center" />
      <NativeSelect
        onChange={(e) => setFormatting(e.target.value as JoinFormat)}
        className="nodrag nowheel"
        data={formattingOptions}
        size="xs"
        value={formatting}
        miw="80px"
      />
      <Handle
        type="target"
        position={Position.Left}
        id="__input"
        className="grouped-handle"
        style={{ top: "50%" }}
        onConnect={handleOnConnect}
      />
      <Handle
        type="source"
        position={Position.Right}
        id="output"
        className="grouped-handle"
        style={{ top: "50%" }}
      />
    </BaseNode>
  );
};

export default JoinNode;
