// UploadNode.tsx

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

interface TemplateVarInfo {
  text?: string;
  image?: string;
  url?: string;
  metavars?: Record<string, any>;
}

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

  const handleFilesUpload = useCallback(
    async (files: FileList) => {
      if (files.length === 0) return;

      setStatus(Status.LOADING);
      const filePromises: Promise<TemplateVarInfo>[] = Array.from(files).map(
        (file) => {
          return new Promise<TemplateVarInfo>((resolve, reject) => {
            const reader = new FileReader();

            reader.onload = () => {
              const result = reader.result;
              if (typeof result === "string") {
                if (file.type.startsWith("image/")) {
                  resolve({
                    image: result,
                    metavars: {
                      size: file.size,
                      type: file.type,
                      id: uuid(),
                    },
                  });
                } else if (
                  file.type.startsWith("text/") ||
                  file.type === "application/json"
                ) {
                  resolve({
                    text: result,
                    metavars: {
                      size: file.size,
                      type: file.type,
                      id: uuid(),
                    },
                  });
                } else {
                  resolve({
                    url: result,
                    metavars: {
                      size: file.size,
                      type: file.type,
                      id: uuid(),
                    },
                  });
                }
              } else {
                reject(new Error("Failed to read file."));
              }
            };

            reader.onerror = () => {
              reject(new Error("Error reading file."));
            };

            if (file.type.startsWith("image/")) {
              reader.readAsDataURL(file);
            } else if (
              file.type.startsWith("text/") ||
              file.type === "application/json"
            ) {
              reader.readAsText(file);
            } else {
              reader.readAsDataURL(file);
            }
          });
        },
      );

      try {
        const newFields = await Promise.all(filePromises);
        const updatedFields = [...fields, ...newFields];
        setFields(updatedFields);
        setDataPropsForNode(id, { fields: updatedFields });
        setStatus(Status.READY);
      } catch (error: any) {
        console.error("Error uploading files:", error);
        setStatus(Status.ERROR);
        showAlert && showAlert(`Error uploading files: ${error.message}`);
      }
    },
    [fields, id, setDataPropsForNode, showAlert],
  );

  const handleFileInputChange = (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    if (event.target.files) {
      handleFilesUpload(event.target.files);
      event.target.value = "";
    }
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (event.dataTransfer.files) {
      handleFilesUpload(event.dataTransfer.files);
    }
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  const handleRemoveFile = (index: number) => {
    const updatedFields = fields.filter((_, i) => i !== index);
    setFields(updatedFields);
    setDataPropsForNode(id, { fields: updatedFields });
  };

  const handleClearUploads = useCallback(() => {
    setFields([]);
    setDataPropsForNode(id, { fields: [] });
    setStatus(Status.READY);
  }, [id, setDataPropsForNode]);

  useEffect(() => {
    if (data.refresh) {
      handleClearUploads();
      setDataPropsForNode(id, { refresh: false });
    }
  }, [data.refresh, id, setDataPropsForNode, handleClearUploads]);

  useEffect(() => {
    return () => {
      fields.forEach((field) => {
        if (field.url) {
          URL.revokeObjectURL(field.url);
        }
      });
    };
  }, [fields]);

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
          ref={fileInputRef}
          style={{ display: "none" }}
          onChange={handleFileInputChange}
        />
      </div>

      <Handle
        type="source"
        position={Position.Right}
        id="upload"
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
                          {field.text || "Image File" || "Other File"}
                        </Text>
                        {field.text && (
                          <Text size="xs" color="dimmed">
                            {field.text.length > 50
                              ? `${field.text.substring(0, 50)}...`
                              : field.text}
                          </Text>
                        )}
                        {field.url && (
                          <Text size="xs" color="dimmed">
                            URL:{" "}
                            <a
                              href={field.url}
                              target="_blank"
                              rel="noopener noreferrer"
                            >
                              Download
                            </a>
                          </Text>
                        )}
                        {field.image && (
                          <img
                            src={field.image}
                            alt={field.text || "Uploaded Image"}
                            style={{
                              maxWidth: "100px",
                              maxHeight: "100px",
                              marginTop: "5px",
                            }}
                          />
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
