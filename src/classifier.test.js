import assert from 'node:assert/strict';
import test from 'node:test';
import { extractFaceFeatures, predictExpression, trainClassifier } from './classifier.js';

const happy = [0.42, 0.08, 0.035, 0.036, 0.05, 0.052, 0.24, 0.25];
const serious = [0.26, 0.025, 0.03, 0.031, 0.045, 0.044, 0.16, 0.17];

const faceAt = (scale, offsetX, offsetY) => {
  const keypoints = Array.from({ length: 468 }, () => ({ x: 0, y: 0 }));
  const points = {
    234: [0, 0],
    454: [200, 0],
    61: [60, 120],
    291: [140, 120],
    13: [95, 115],
    14: [105, 120],
    159: [70, 80],
    145: [70, 90],
    386: [130, 80],
    374: [130, 90],
    105: [68, 70],
    334: [132, 70]
  };
  for (const [index, [x, y]] of Object.entries(points)) {
    keypoints[index] = { x: x * scale + offsetX, y: y * scale + offsetY };
  }
  return { keypoints };
};

test('FaceMesh distance features are invariant to image position and scale', () => {
  const baseline = extractFaceFeatures(faceAt(1, 0, 0));
  const transformed = extractFaceFeatures(faceAt(2.4, 310, 170));

  assert.ok(baseline);
  assert.ok(transformed);
  baseline.forEach((value, index) => {
    assert.ok(Math.abs(value - transformed[index]) < 1e-10);
  });
});

test('classifier keeps an earlier emotion available when a participant skips it', () => {
  const combinedExamples = [
    { label: 'happy', features: happy },
    { label: 'happy', features: happy.map((value, index) => value + (index % 2 ? 0.002 : -0.002)) },
    { label: 'serious', features: serious },
    { label: 'serious', features: serious.map((value, index) => value + (index % 2 ? -0.002 : 0.002)) }
  ];
  const classifier = trainClassifier(combinedExamples);

  assert.deepEqual(classifier.classes.map(item => item.label).sort(), ['happy', 'serious']);
  assert.equal(predictExpression(happy, classifier).label, 'happy');
});

test('classifier reports uncertainty for ambiguous or out-of-range expressions', () => {
  const ambiguous = trainClassifier([
    { label: 'happy', features: happy },
    { label: 'serious', features: happy }
  ]);
  const outOfRange = trainClassifier([{ label: 'happy', features: happy }]);

  assert.equal(predictExpression(happy, ambiguous).state, 'uncertain');
  assert.equal(predictExpression(happy.map(value => value + 5), outOfRange).state, 'uncertain');
});
