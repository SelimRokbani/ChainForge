/**
 * Library of RAGAS-style evaluator functions for RAG evaluation
 */

export interface RagasEvaluator {
  name: string;
  description: string;
  code: string;
  language: "javascript" | "python";
}

export const ragasEvaluators: RagasEvaluator[] = [
  {
    name: "Answer Correctness",
    description:
      "Evaluates if the retrieved document contains the correct answer",
    language: "javascript",
    code: `function evaluate(response) {
  // Validate required metadata fields
  const question = response.meta?.queryGroup || response.meta?.Question || "Unknown question";
  const expectedAnswer = 
    response.meta?.Answer || 
    response.meta?.ExpectedAnswer || 
    response.var?.Answer || 
    response.var?.ExpectedAnswer ||
    null;
  
  if (!expectedAnswer) {
    return 0;
  }

  // Normalize text for comparison
  const normalizeText = (text) => text
    .toLowerCase()
    .replace(/[.,/#!$%^&*;:{}=-_~()]/g, '') // Remove punctuation
    .replace(/\\s+/g, ' ') // Normalize spaces
    .trim();

  const normalizedResponse = normalizeText(response.text);
  const normalizedAnswer = normalizeText(expectedAnswer);

  // Check if the retrieved context contains the expected answer using word boundaries
  const answerWords = normalizedAnswer.split(' ');
  const containsAnswer = answerWords.every(word => {
    const wordBoundaryRegex = new RegExp(\`\\\\b\${word}\\\\b\`);
    return wordBoundaryRegex.test(normalizedResponse);
  });

  // Calculate score based on exact match or partial match
  let score = 0;
  if (containsAnswer) {
    score = 1.0; // Exact match
  } else {
    // Partial match - calculate how many words from the expected answer are in the text
    const matchedWords = answerWords.filter(word => {
      const wordBoundaryRegex = new RegExp(\`\\\\b\${word}\\\\b\`);
      return wordBoundaryRegex.test(normalizedResponse);
    }).length;
    score = matchedWords / answerWords.length;
  }

  return score;
}`,
  },
  {
    name: "Context Relevance",
    description:
      "Evaluates how relevant the retrieved document is to the question",
    language: "javascript",
    code: `function evaluate(response) {
  // Validate required metadata fields
  const question = response.meta?.queryGroup || response.meta?.Question || "Unknown question";
  const contextKeywords = (response.meta?.context || "").split(',').map(k => k.trim().toLowerCase()).filter(k => k);

  // Get retrieval metadata
  const similarity = parseFloat(response.meta?.similarity || "0");

  // Preprocess text
  const normalizeText = (text) => text
    .toLowerCase()
    .replace(/[.,/#!$%^&*;:{}=-_~()]/g, '')
    .replace(/\\s+/g, ' ')
    .trim();

  const normalizedResponse = normalizeText(response.text);

  // Extract meaningful words from the question (length > 3, exclude stop words)
  const stopWords = new Set(["what", "when", "where", "which", "whose", "does", "that", "this", "have", "from", "with", "about", "into", "during", "before", "after"]);
  const questionKeywords = normalizeText(question)
    .split(' ')
    .filter(word => word.length > 3 && !stopWords.has(word));

  // Combine question keywords and context keywords
  const allKeywords = [...questionKeywords, ...contextKeywords];
  let matchCount = 0;

  // Single pass to check for keyword matches
  allKeywords.forEach(keyword => {
    if (keyword && normalizedResponse.includes(keyword)) {
      matchCount++;
    }
  });

  // Calculate context relevance score
  const totalKeywords = allKeywords.length;
  const relevanceScore = totalKeywords > 0 ? Math.min(1.0, matchCount / totalKeywords) : 0;

  // Adjust score based on vector similarity
  const combinedScore = (relevanceScore * 0.7) + (similarity * 0.3);

  return combinedScore;
}`,
  },
  {
    name: "Context Faithfulness",
    description:
      "Evaluates if the retrieved document faithfully supports the expected answer",
    language: "javascript",
    code: `function evaluate(response) {
  // Validate required metadata fields
  const expectedAnswer = 
    response.meta?.Answer || 
    response.meta?.ExpectedAnswer || 
    response.var?.Answer || 
    response.var?.ExpectedAnswer ||
    null;
  const groundTruth = 
    response.meta?.ground_truth || 
    response.meta?.GroundTruth || 
    response.var?.ground_truth || 
    response.var?.GroundTruth || 
    "";

  if (!expectedAnswer) {
    return 0;
  }

  // Normalize text
  const normalizeText = (text) => text
    .toLowerCase()
    .replace(/[.,/#!$%^&*;:{}=-_~()]/g, '')
    .replace(/\\s+/g, ' ')
    .trim();

  const normalizedResponse = normalizeText(response.text);
  const normalizedAnswer = normalizeText(expectedAnswer);
  const normalizedGroundTruth = normalizeText(groundTruth);

  // Check if the retrieved text contains the ground truth
  const containsGroundTruth = groundTruth && normalizedResponse.includes(normalizedGroundTruth);

  // Check if the document supports the expected answer
  const answerWords = normalizedAnswer.split(' ');
  const supportsAnswer = answerWords.every(word => {
    const wordBoundaryRegex = new RegExp(\`\\\\b\${word}\\\\b\`);
    return wordBoundaryRegex.test(normalizedResponse);
  });

  // Calculate faithfulness score
  let faithfulnessScore = 0;
  if (containsGroundTruth && supportsAnswer) {
    faithfulnessScore = 1.0; // Perfect score if contains both ground truth and answer
  } else if (supportsAnswer) {
    faithfulnessScore = 0.7; // Good score if supports answer but may not have complete ground truth
  } else if (containsGroundTruth) {
    faithfulnessScore = 0.5; // Partial score for having ground truth but not explicitly supporting answer
  }

  return faithfulnessScore;
}`,
  },
  {
    name: "Combined RAGAS Evaluation",
    description:
      "Provides a comprehensive evaluation combining all RAGAS metrics",
    language: "javascript",
    code: `function evaluate(response) {
  // Get weights from metadata, if provided
  const weights = response.meta?.weights || {
    answer: 0.4,
    relevance: 0.3,
    faithfulness: 0.2,
    retrieval: 0.1
  };

  // Validate required metadata fields
  const question = response.meta?.queryGroup || response.meta?.Question || "Unknown question";
  const expectedAnswer = 
    response.meta?.Answer || 
    response.meta?.ExpectedAnswer || 
    response.var?.Answer || 
    response.var?.ExpectedAnswer ||
    null;
  const groundTruth = 
    response.meta?.ground_truth || 
    response.meta?.GroundTruth || 
    response.var?.ground_truth || 
    response.var?.GroundTruth || 
    "";
  const contextKeywords = (response.meta?.context || "").split(',').map(k => k.trim().toLowerCase()).filter(k => k);

  if (!expectedAnswer) {
    return 0;
  }

  // Get retrieval metadata
  const similarity = parseFloat(response.meta?.similarity || "0");

  // Normalize text
  const normalizeText = (text) => text
    .toLowerCase()
    .replace(/[.,/#!$%^&*;:{}=-_~()]/g, '')
    .replace(/\\s+/g, ' ')
    .trim();

  const normalizedResponse = normalizeText(response.text);
  const normalizedAnswer = normalizeText(expectedAnswer);
  const normalizedGroundTruth = normalizeText(groundTruth);

  // 1. ANSWER CORRECTNESS
  const answerWords = normalizedAnswer.split(' ');
  const containsAnswer = answerWords.every(word => {
    const wordBoundaryRegex = new RegExp(\`\\\\b\${word}\\\\b\`);
    return wordBoundaryRegex.test(normalizedResponse);
  });

  // 2. CONTEXT RELEVANCE
  const stopWords = new Set(["what", "when", "where", "which", "whose", "does", "that", "this", "have", "from", "with", "about", "into", "during", "before", "after"]);
  const questionKeywords = normalizeText(question)
    .split(' ')
    .filter(word => word.length > 3 && !stopWords.has(word));

  let keywordMatchCount = 0;
  const allKeywords = [...questionKeywords, ...contextKeywords];
  allKeywords.forEach(keyword => {
    if (keyword && normalizedResponse.includes(keyword)) {
      keywordMatchCount++;
    }
  });

  const totalKeywords = allKeywords.length;
  const relevanceScore = totalKeywords > 0 ? Math.min(1.0, keywordMatchCount / totalKeywords) : 0;

  // 3. CONTEXT FAITHFULNESS
  const containsGroundTruth = groundTruth && normalizedResponse.includes(normalizedGroundTruth);

  // 4. Calculate combined RAGAS score
  const answerScore = containsAnswer ? 1.0 : 0.0;
  const faithfulnessScore = containsGroundTruth ? 1.0 : (containsAnswer ? 0.7 : 0.0);

  const ragasScore = (
    (answerScore * weights.answer) +
    (relevanceScore * weights.relevance) +
    (faithfulnessScore * weights.faithfulness) +
    (similarity * weights.retrieval)
  );

  return ragasScore;
}`,
  },
  {
    name: "Groundedness Evaluation",
    description:
      "Evaluates how well-grounded a response is in the source document",
    language: "javascript",
    code: `function evaluate(response) {
  // Get retrieval metadata
  const similarity = parseFloat(response.meta?.similarity || "0");

  // Normalize text
  const normalizeText = (text) => text
    .toLowerCase()
    .replace(/[.,/#!$%^&*;:{}=-_~()]/g, '')
    .replace(/\\s+/g, ' ')
    .trim();

  const normalizedResponse = normalizeText(response.text);

  // Extract key factual claims from document
  const sentences = response.text.split(/[.!?]+\\s+/).filter(s => s.length > 10);
  const factPattern = /([A-Z][a-z]+(?: [A-Z][a-z]+)*)|\\d+(?:st|nd|rd|th)?|\\$\\d+|\\d+%|\\d+\\.\\d+/g;
  const keyFacts = new Set();

  sentences.forEach(sentence => {
    const matches = sentence.match(factPattern);
    if (matches) {
      matches.forEach(match => keyFacts.add(normalizeText(match)));
    }
  });

  // Calculate groundedness score based on fact density and similarity
  const factDensity = keyFacts.size / (response.text.length / 100); // Facts per 100 chars

  const groundednessScore = Math.min(1.0, (factDensity * 0.5) + (similarity * 0.5));

  return groundednessScore;
}`,
  },
  {
    name: "Citation Precision",
    description:
      "Evaluates if the retrieved document contains information that would be cited",
    language: "javascript",
    code: `function evaluate(response) {
  // Get retrieval metadata
  const similarity = parseFloat(response.meta?.similarity || "0");

  // Check for citation-worthy content
  const citationPatterns = [
    /\\d+%/g,                   // Percentages
    /\\$\\d+/g,                  // Dollar amounts
    /"([^"]+)"/g,              // Quoted text
    /according to ([^,.]+)/gi, // Attribution phrases
    /in \\d{4}/g,              // Years
    /\\d+ (million|billion|trillion)/gi, // Large numbers
  ];

  let citationWorthy = 0;

  citationPatterns.forEach(pattern => {
    const matches = [...response.text.matchAll(pattern)];
    citationWorthy += matches.length;
  });

  // Calculate precision score
  const citationPrecision = Math.min(1.0, (citationWorthy / 3) * 0.7 + (similarity * 0.3));

  return citationPrecision;
}`,
  },
  {
    name: "Query-Document Alignment",
    description: "Evaluates how well the document aligns with the query intent",
    language: "javascript",
    code: `function evaluate(response) {
  // Get question from metadata
  const question = response.meta?.queryGroup || response.meta?.Question || "Unknown question";

  // Get retrieval metadata
  const similarity = parseFloat(response.meta?.similarity || "0");

  // Normalize text
  const normalizeText = (text) => text
    .toLowerCase()
    .replace(/[.,/#!$%^&*;:{}=-_~()]/g, '')
    .replace(/\\s+/g, ' ')
    .trim();

  const normalizedResponse = normalizeText(response.text);

  // Extract key entities from question
  const stopWords = new Set(["what", "where", "when", "which", "whose", "does", "that", "this", "have"]);
  const questionWords = normalizeText(question)
    .split(' ')
    .filter(w => w.length > 3 && !stopWords.has(w));

  // Check how many key question words appear in document
  const matchingWords = questionWords.filter(word => normalizedResponse.includes(word));

  // Calculate query-document alignment score
  const wordMatchRatio = questionWords.length > 0 ? matchingWords.length / questionWords.length : 0;

  // Adjust score based on question type
  const alignmentScore = wordMatchRatio * 0.7 + similarity * 0.3;

  return alignmentScore;
}`,
  },
  {
    name: "Answer Completeness",
    description: "Evaluates if the response fully answers the question",
    language: "javascript",
    code: `function evaluate(response) {
  // Validate required metadata fields
  const expectedAnswer = 
    response.meta?.Answer || 
    response.meta?.ExpectedAnswer || 
    response.var?.Answer || 
    response.var?.ExpectedAnswer ||
    null;

  if (!expectedAnswer) {
    return 0;
  }

  // Normalize text
  const normalizeText = (text) => text
    .toLowerCase()
    .replace(/[.,/#!$%^&*;:{}=-_~()]/g, '')
    .replace(/\\s+/g, ' ')
    .trim();

  const normalizedResponse = normalizeText(response.text);
  const normalizedAnswer = normalizeText(expectedAnswer);

  // Tokenize expected answer into key components
  const stopWords = new Set(["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"]);
  const expectedComponents = normalizedAnswer
    .split(' ')
    .filter(word => word.length > 2 && !stopWords.has(word));

  // Check how many components are present in the response
  const matchedComponents = expectedComponents.filter(component => {
    const componentRegex = new RegExp(\`\\\\b\${component}\\\\b\`);
    return componentRegex.test(normalizedResponse);
  });

  // Calculate completeness score
  const completenessScore = expectedComponents.length > 0 ?
    matchedComponents.length / expectedComponents.length : 0;

  return completenessScore;
}`,
  },
];
