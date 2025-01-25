import React, {
  useEffect,
  useState,
  useRef,
  useCallback,
  useMemo,
  useContext,
} from "react";
import { Handle, Position } from "reactflow";
import { v4 as uuid } from "uuid";
import {
  Button,
  Group,
  Text,
  Modal,
  Box,
  Tooltip,
  List,
  ThemeIcon,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { IconUpload, IconTrash, IconList } from "@tabler/icons-react";
import useStore from "./store";
import BaseNode from "./BaseNode";
import NodeLabel from "./NodeLabelComponent";
import { AlertModalContext } from "./AlertModal";
import { Status } from "./StatusIndicatorComponent";
import { TemplateVarInfo } from "./backend/typing";

// Define the properties for the UploadNode component
export interface UploadNodeProps {
  data: {
    title: string;
    fields: TemplateVarInfo[];
    refresh: boolean;
  };
  id: string;
  type: string;
}

const UploadNode: React.FC<UploadNodeProps> = ({ data, id, type }) => {
  const nodeIcon = useMemo(() => "📁", []);
  const nodeDefaultTitle = useMemo(() => "Upload Node", []);
  const setDataPropsForNode = useStore((state) => state.setDataPropsForNode);
  const [fields, setFields] = useState<TemplateVarInfo[]>(data.fields || []);
  const [status, setStatus] = useState<Status>(Status.READY);
  const [
    fileListModalOpened,
    { open: openFileListModal, close: closeFileListModal },
  ] = useDisclosure(false);
  const showAlert = useContext(AlertModalContext);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Handle file uploads
  const handleFilesUpload = useCallback(
    async (files: FileList) => {
      if (files.length === 0) return;

      setStatus(Status.LOADING);
      const updatedFields = [...fields];

      for (const file of Array.from(files)) {
        const formData = new FormData();
        formData.append("file", file);

        try {
          const res = await fetch("http://localhost:5000/upload", {
            method: "POST",
            body: formData,
          });
          if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || "Failed to process file");
          }

          const json = await res.json();
          const textContent = json.text || "";

          // Add filename to the output along with the text content
          updatedFields.push({
            text: textContent,
            prompt: "",
            fill_history: {},
            llm: {
              name: "None",
              emoji: "",
              base_model: "",
              model: "",
              temp: 0,
            },
            metavars: {
              size: file.size.toString(),
              type: file.type,
              filename: file.name,
              id: uuid(),
            },
          });
        } catch (error: any) {
          console.error("Error uploading file:", error);
          showAlert &&
            showAlert(`Error uploading ${file.name}: ${error.message}`);
          setStatus(Status.ERROR);
        }
      }

      setFields(updatedFields);
      // Here we modify the output to include the filename:
      const outputFields = updatedFields.map((field) => ({
        text: field.text,
        filename: field.metavars?.filename || "Untitled file",
      }));
      setDataPropsForNode(id, { fields: updatedFields, output: outputFields });
      setStatus(Status.READY);
    },
    [fields, id, setDataPropsForNode, showAlert],
  );

  // Handle file input change event
  const handleFileInputChange = (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    if (event.target.files) {
      handleFilesUpload(event.target.files);
      event.target.value = "";
    }
  };

  // Handle file drop event
  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (event.dataTransfer.files) {
      handleFilesUpload(event.dataTransfer.files);
    }
  };

  // Handle drag over event
  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  // Remove a specific file from the list
  const handleRemoveFile = (index: number) => {
    const updatedFields = fields.filter((_, i) => i !== index);
    setFields(updatedFields);
    // Update output accordingly
    const outputFields = updatedFields.map((field) => ({
      text: field.text,
      filename: field.metavars?.filename || "Untitled file",
    }));
    setDataPropsForNode(id, { fields: updatedFields, output: outputFields });
  };

  // Clear all uploaded files
  const handleClearUploads = useCallback(() => {
    setFields([]);
    setDataPropsForNode(id, { fields: [], output: [] });
    setStatus(Status.READY);
  }, [id, setDataPropsForNode]);

  // Effect to handle refresh
  useEffect(() => {
    if (data.refresh) {
      handleClearUploads();
      setDataPropsForNode(id, { refresh: false });
    }
  }, [data.refresh, id, setDataPropsForNode, handleClearUploads]);

  return (
    <BaseNode classNames="upload-node" nodeId={id}>
      <NodeLabel
        title={data.title || nodeDefaultTitle}
        nodeId={id}
        icon={nodeIcon}
        status={status}
        customButtons={[
          <Tooltip label="View Uploaded Files" withArrow key="view-files">
            <button
              className="custom-button"
              onClick={openFileListModal}
              style={{ border: "none", background: "none" }}
            >
              <IconList size="16px" color="gray" />
            </button>
          </Tooltip>,
        ]}
      />

      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        style={{
          border: "2px dashed #ccc",
          borderRadius: "8px",
          padding: "20px",
          textAlign: "center",
          margin: "10px",
          backgroundColor: "#f9f9f9",
          cursor: "pointer",
        }}
        onClick={() => fileInputRef.current?.click()}
      >
        <IconUpload size={40} color="#888" />
        <Text size="sm" color="dimmed">
          Drag & drop files here or click to upload
        </Text>
        <input
          type="file"
          multiple
          accept=".pdf,.docx,.txt"
          ref={fileInputRef}
          style={{ display: "none" }}
          onChange={handleFileInputChange}
        />
      </div>

      <Handle
        type="source"
        position={Position.Right}
        id="text"
        className="grouped-handle"
        style={{ top: "50%" }}
      />

      <Modal
        title={`Uploaded Files (${fields.length})`}
        size="md"
        opened={fileListModalOpened}
        onClose={closeFileListModal}
      >
        <Box>
          {fields.length === 0 ? (
            <Text color="dimmed">No files uploaded.</Text>
          ) : (
            <List spacing="xs" size="sm" center>
              {fields.map((field, index) => (
                <List.Item key={field.metavars?.id || index}>
                  <Group position="apart">
                    <Group>
                      <ThemeIcon color="blue" variant="light">
                        <IconUpload size={16} />
                      </ThemeIcon>
                      <div>
                        <Text size="sm" weight={500}>
                          {field.metavars?.filename || "Untitled file"}
                        </Text>
                        {field.text && (
                          <Text size="xs" color="dimmed">
                            {field.text.substring(0, 50)}
                            {field.text.length > 50 ? "..." : ""}
                          </Text>
                        )}
                      </div>
                    </Group>
                    <Button
                      variant="subtle"
                      color="red"
                      compact
                      onClick={() => handleRemoveFile(index)}
                    >
                      Remove
                    </Button>
                  </Group>
                </List.Item>
              ))}
            </List>
          )}
        </Box>
      </Modal>
    </BaseNode>
  );
};

export default UploadNode;
