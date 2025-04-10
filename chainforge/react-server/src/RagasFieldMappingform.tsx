import React, { useState, useEffect } from "react";
import {
  TextInput,
  Text,
  Group,
  Paper,
  Title,
  Button,
  Collapse,
  Stack,
  Divider,
  Alert,
  Box,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { IconChevronDown, IconChevronRight, IconInfoCircle } from "@tabler/icons-react";

export interface RagasFieldMappings {
  questionField: string;
  answerField: string;
  contextField: string;
  groundTruthField: string;
}

interface RagasFieldMappingFormProps {
  mappings: RagasFieldMappings;
  onChange: (mappings: RagasFieldMappings) => void;
  availableFields?: string[];
}

const RagasFieldMappingForm: React.FC<RagasFieldMappingFormProps> = ({
  mappings,
  onChange,
  availableFields = [],
}) => {
  const [opened, { toggle }] = useDisclosure(true);
  const [localMappings, setLocalMappings] = useState<RagasFieldMappings>(mappings);
  
  // Update local state when props change
  useEffect(() => {
    setLocalMappings(mappings);
  }, [mappings]);

  // Handle field changes
  const handleChange = (field: keyof RagasFieldMappings, value: string) => {
    const newMappings = { ...localMappings, [field]: value };
    setLocalMappings(newMappings);
    onChange(newMappings);
  };

  return (
    <Paper withBorder p="xs" mb="md">
      <Group position="apart" mb={opened ? "xs" : 0}>
        <Button 
          variant="subtle" 
          color="gray" 
          compact 
          onClick={toggle}
          leftIcon={opened ? <IconChevronDown size={16} /> : <IconChevronRight size={16} />}
        >
          RAGAS Field Mappings
        </Button>
      </Group>

      <Collapse in={opened}>
        <Alert 
          color="blue" 
          mb="sm" 
          icon={<IconInfoCircle size="1rem" />}
        >
          Map your data columns to the fields required by RAGAS evaluators.
        </Alert>

        <Stack spacing="xs">
          <TextInput
            label="Question/Query Field"
            description="Column containing the questions/queries"
            placeholder="e.g., Question, query, queryText"
            value={localMappings.questionField}
            onChange={(e) => handleChange("questionField", e.target.value)}
          />
          
          <TextInput
            label="Expected Answer Field"
            description="Column containing the expected answers"
            placeholder="e.g., Answer, ExpectedAnswer, expected"
            value={localMappings.answerField}
            onChange={(e) => handleChange("answerField", e.target.value)}
          />
          
          <TextInput
            label="Context Field (optional)"
            description="Column containing context keywords or content"
            placeholder="e.g., context, keywords, information"
            value={localMappings.contextField}
            onChange={(e) => handleChange("contextField", e.target.value)}
          />
          
          <TextInput
            label="Ground Truth Field (optional)"
            description="Column containing ground truth information"
            placeholder="e.g., groundTruth, GroundTruth, truth"
            value={localMappings.groundTruthField}
            onChange={(e) => handleChange("groundTruthField", e.target.value)}
          />
        </Stack>

        {availableFields.length > 0 && (
          <Box mt="xs">
            <Text size="xs" color="dimmed">Available fields in your data:</Text>
            <Text size="xs" color="blue">
              {availableFields.join(', ')}
            </Text>
          </Box>
        )}
      </Collapse>
    </Paper>
  );
};

export default RagasFieldMappingForm;
