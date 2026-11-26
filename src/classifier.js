const FEATURE_COUNT = 8;
const MIN_FEATURE_SCALE = 0.04;
const MIN_CLASS_DISTANCE = 2.25;
const MAX_CLASS_DISTANCE = 4;
const MIN_CONFIDENCE = 0.3;
const MIN_CLASS_MARGIN = 0.08;

const point = (face, index) => face?.keypoints?.[index];

const distance = (first, second) => {
  if (!first || !second) return null;
  return Math.hypot(first.x - second.x, first.y - second.y);
};

const validVector = vector => Array.isArray(vector)
  && vector.length === FEATURE_COUNT
  && vector.every(value => Number.isFinite(value));

const euclideanDistance = (left, right) => Math.sqrt(
  left.reduce((sum, value, index) => sum + (value - right[index]) ** 2, 0)
);

export const extractFaceFeatures = face => {
  if (!face?.keypoints) return null;
  const faceWidth = distance(point(face, 234), point(face, 454));
  if (!faceWidth || faceWidth <= 1) return null;

  const vector = [
    distance(point(face, 61), point(face, 291)) / faceWidth,
    distance(point(face, 13), point(face, 14)) / faceWidth,
    distance(point(face, 159), point(face, 145)) / faceWidth,
    distance(point(face, 386), point(face, 374)) / faceWidth,
    distance(point(face, 105), point(face, 159)) / faceWidth,
    distance(point(face, 334), point(face, 386)) / faceWidth,
    distance(point(face, 61), point(face, 13)) / faceWidth,
    distance(point(face, 291), point(face, 14)) / faceWidth
  ];
  return validVector(vector) ? vector : null;
};

export const trainClassifier = trainingExamples => {
  const samples = trainingExamples.filter(example =>
    typeof example?.label === 'string'
    && example.label.length > 0
    && validVector(example.features)
  );
  if (!samples.length) return null;

  const means = Array.from({ length: FEATURE_COUNT }, (_, index) =>
    samples.reduce((sum, sample) => sum + sample.features[index], 0) / samples.length
  );
  const scales = means.map((mean, index) => {
    const variance = samples.reduce((sum, sample) =>
      sum + (sample.features[index] - mean) ** 2, 0
    ) / samples.length;
    return Math.max(Math.sqrt(variance), MIN_FEATURE_SCALE);
  });
  const normalize = features => features.map((value, index) =>
    (value - means[index]) / scales[index]
  );
  const examplesByLabel = new Map();
  for (const sample of samples) {
    const examples = examplesByLabel.get(sample.label) || [];
    examples.push(normalize(sample.features));
    examplesByLabel.set(sample.label, examples);
  }

  const classes = [...examplesByLabel.entries()].map(([label, examples]) => {
    const centroid = Array.from({ length: FEATURE_COUNT }, (_, index) =>
      examples.reduce((sum, sample) => sum + sample[index], 0) / examples.length
    );
    const distances = examples
      .map(example => euclideanDistance(example, centroid))
      .sort((left, right) => left - right);
    const percentile90 = distances[Math.floor((distances.length - 1) * 0.9)] || 0;
    return {
      label,
      centroid,
      maxDistance: Math.min(MAX_CLASS_DISTANCE, Math.max(MIN_CLASS_DISTANCE, percentile90 * 1.5))
    };
  });

  return { means, scales, classes };
};

export const predictExpression = (features, classifier) => {
  if (!validVector(features) || !classifier?.classes?.length) {
    return { label: null, confidence: 0, state: 'uncertain' };
  }

  const normalized = features.map((value, index) =>
    (value - classifier.means[index]) / classifier.scales[index]
  );
  const ranked = classifier.classes
    .map(candidate => ({
      label: candidate.label,
      distance: euclideanDistance(normalized, candidate.centroid),
      maxDistance: candidate.maxDistance
    }))
    .sort((left, right) => left.distance - right.distance);
  const best = ranked[0];
  const second = ranked[1];
  const margin = second
    ? (second.distance - best.distance) / Math.max(second.distance, Number.EPSILON)
    : 1;
  const distanceConfidence = Math.max(0, 1 - best.distance / best.maxDistance);
  const confidence = distanceConfidence * (second ? 0.65 + 0.35 * margin : 1);

  if (confidence < MIN_CONFIDENCE || (second && margin < MIN_CLASS_MARGIN)) {
    return { label: null, confidence, state: 'uncertain', nearestLabel: best.label };
  }
  return { label: best.label, confidence, state: 'recognized' };
};
