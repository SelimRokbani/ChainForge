import React, {
  useState,
  useEffect,
  forwardRef,
  useImperativeHandle,
} from "react";
import {
  Modal,
  Button,
  TextInput,
  Text,
  Group,
  Stack,
  Title,
  Divider,
  Alert,
} from "@mantine/core";
import { IconAlertCircle } from "@tabler/icons-react";

export interface ColumnMapping {
  responseField: string;
  displayName: string;
}

export interface ColumnMappingModalRef {
  openModal: (
    detectedColumns: string[],
    onConfirm: (mappings: ColumnMapping[]) => void,
  ) => void;
}

interface ColumnMappingModalProps {
  title?: string;
}

const ColumnMappingModal = forwardRef<
  ColumnMappingModalRef,
  ColumnMappingModalProps
>((props, ref) => {
  const [isOpen, setIsOpen] = useState(false);
  const [columns, setColumns] = useState<ColumnMapping[]>([]);
  const [onConfirmCallback, setOnConfirmCallback] = useState<
    ((mappings: ColumnMapping[]) => void) | null
  >(null);

  // Function to open the modal with detected column names
  const openModal = (
    detectedColumns: string[],
    onConfirm: (mappings: ColumnMapping[]) => void,
  ) => {
    // Initialize the mappings with the detected columns
    const initialMappings: ColumnMapping[] = detectedColumns.map((col) => ({
      responseField: col,
      displayName: formatColumnName(col), // Format the column name for display
    }));

    setColumns(initialMappings);
    setOnConfirmCallback(() => onConfirm);
    setIsOpen(true);
  };

  // Helper function to format column names from raw response fields
  const formatColumnName = (fieldName: string): string => {
    // Convert camelCase or snake_case to Title Case
    return fieldName
      .replace(/([A-Z])/g, " $1") // Insert space before capital letters
      .replace(/_/g, " ") // Replace underscores with spaces
      .replace(/^\w/, (c) => c.toUpperCase()) // Capitalize first letter
      .trim();
  };

  // Handle input change
  const handleColumnChange = (
    index: number,
    field: "responseField" | "displayName",
    value: string,
  ) => {
    const updatedColumns = [...columns];
    updatedColumns[index][field] = value;
    setColumns(updatedColumns);
  };

  // Handle form submission
  const handleSubmit = () => {
    if (onConfirmCallback) {
      onConfirmCallback(columns);
    }
    setIsOpen(false);
  };

  useImperativeHandle(ref, () => ({
    openModal,
  }));

  return (
    <Modal
      opened={isOpen}
      onClose={() => setIsOpen(false)}
      title={props.title || "Configure Column Mappings"}
      size="lg"
    >
      <Stack spacing="md">
        <Alert
          icon={<IconAlertCircle size="1rem" />}
          title="Column Mapping"
          color="blue"
        >
          Map response data fields to display names. This affects how evaluation
          results are displayed.
        </Alert>

        <Divider label="Field Mappings" labelPosition="center" />

        {columns.map((column, index) => (
          <Group key={index} position="apart" grow>
            <TextInput
              label="Response Field"
              value={column.responseField}
              onChange={(e) =>
                handleColumnChange(index, "responseField", e.target.value)
              }
              placeholder="Original field name"
              readOnly // Usually we don't want users changing the actual field names
            />
            <TextInput
              label="Display Name"
              value={column.displayName}
              onChange={(e) =>
                handleColumnChange(index, "displayName", e.target.value)
              }
              placeholder="Human-readable column name"
            />
          </Group>
        ))}

        <Group position="right" mt="md">
          <Button variant="outline" onClick={() => setIsOpen(false)}>
            Cancel
          </Button>
          <Button onClick={handleSubmit}>Confirm</Button>
        </Group>
      </Stack>
    </Modal>
  );
});

ColumnMappingModal.displayName = "ColumnMappingModal";

export default ColumnMappingModal;
