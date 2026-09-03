// Temporary sample data so the layout can be built and reviewed before the
// webcam (Week 2) and backend (Week 5) are wired in. Delete once real data flows.

export const MOCK_PREDICTION = { word: 'hello', confidence: 0.78 }

export const MOCK_HISTORY = [
  { word: 'hello', confidence: 0.92, time: '12:04:01' },
  { word: 'dog', confidence: 0.85, time: '12:03:44' },
  { word: 'cat', confidence: 0.61, time: '12:03:20' },
  { word: 'you', confidence: 0.44, time: '12:02:58' },
]

// Draft vocabulary from PRD.md section 4 (final list comes from the dataset).
export const MOCK_VOCAB = [
  'a', 'bye', 'can', 'cat', 'demo', 'dog', 'hello',
  'here', 'is', 'lips', 'my', 'read', 'you',
]

// Low-confidence cutoff — matches DESIGN.md "Confidence display rule".
export const CONFIDENCE_THRESHOLD = 0.6
