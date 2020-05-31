/*
 * @module Pitch Type
 * @desc   Model for predicting the pitch type given pitch sensor data.
 *
 */
const tf = require('@tensorflow/tfjs-node')

// @func normalize()
function normalize(val, min, max) {
  if (min === undefined || max === undefined) {
    return value;
  }
  return (value - min) / (max - min);
}

// @data pitch data
const TRAIN_DATA_PATH = '../data/pitch_type_training_data.csv';
const TEST_DATA_PATH = '../data/pitch_type_test_data.csv';

// @data training data
const VX0_MIN = -18.885;
const VX0_MAX = 18.065;
const VY0_MIN = -152.463;
const VY0_MAX = -86.374;
const VZ0_MIN = -15.5146078412997;
const VZ0_MAX = 9.974;
const AX_MIN = -48.0287647107959;
const AX_MAX = 30.592;
const AY_MIN = 9.397;
const AY_MAX = 49.18;
const AZ_MIN = -49.339;
const AZ_MAX = 2.95522851438373;
const START_SPEED_MIN = 59;
const START_SPEED_MAX = 104.4;
const NUM_PITCH_CLASSES = 7;
const TRAINING_DATA_LENGTH = 7000;
const TEST_DATA_LENGTH = 700;

// @func csvTransform
const csvTransform = ({xs, ys}) => {
  const values = [
    normalize(xs.vx0, VX0_MIN, VX0_MAX),
    normalize(xs.vy0, VY0_MIN, VY0_MAX),
    normalize(xs.vz0, VZ0_MIN, VZ0_MAX), normalize(xs.ax, AX_MIN, AX_MAX),
    normalize(xs.ay, AY_MIN, AY_MAX), normalize(xs.az, AZ_MIN, AZ_MAX),
    normalize(xs.start_speed, START_SPEED_MIN, START_SPEED_MAX),
    xs.left_handed_pitcher
  ];
  return {xs: values, ys: ys.pitch_code};
};

// @desc load training data
const trainingData = tf.data.csv(TRAIN_DATA_PATH, {columnConfigs: {pitch_code: {isLabel: true}}})
  .map(csvTransform)
  .shuffle(TRAINING_DATA_LENGTH)
  .batch(100);

// @desc evaluate training data
const trainingValidationData = tf.data.csv(TRAIN_DATA_PATH, {columnConfigs: {pitch_code: {isLabel: true}}})
  .map(csvTransform)
  .batch(TRAINING_DATA_LENGTH);

  // @desc test training data
const testValidationData = tf.data.csv(TEST_DATA_PATH, {columnConfigs: {pitch_code: {isLabel: true}}})
  .map(csvTransform)
  .batch(TEST_DATA_LENGTH);

/*
 * @module Model
 * @desc   ML Model
 */
const model = tf.sequential();

// @desc add layers
model.add(tf.layers.dense({ units: 250, activation: 'relu', inputShape: [8] }));
model.add(tf.layers.dense({ units: 175, activation: 'relu' }));
model.add(tf.layers.dense({ units: 150, activation: 'relu' }));
model.add(tf.layers.dense({ units: NUM_PITCH_CLASSES, activation: 'softmax' }));

// @desc compile data
model.compile({
  optimizer: tf.train.adam(),
  loss: 'sparseCategoricalCrossentropy',
  metrics: ['accuracy']
});

// @return pitch class evaluation percentages for training data
async function evaluate(useTestData) {
  let results = {};
  await trainingValidationData.forEachAsync(pitchTypeBatch => {
    const values = model.predict(pitchTypeBatch.xs).dataSync();
    const classSize = TRAINING_DATA_LENGTH / NUM_PITCH_CLASSES;
    for (let i = 0; i < NUM_PITCH_CLASSES; i++) {
      results[pitchFromClassNum(i)] = {
        training: calcPitchClassEval(i, classSize, values)
      };
    }
  });

  if (useTestData) {
    await testValidationData.forEachAsync(pitchTypeBatch => {
      const values = model.predict(pitchTypeBatch.xs).dataSync();
      const classSize = TEST_DATA_LENGTH / NUM_PITCH_CLASSES;
      for (let i = 0; i < NUM_PITCH_CLASSES; i++) {
        results[pitchFromClassNum(i)].validation =
            calcPitchClassEval(i, classSize, values);
      }
    });
  }
  return results;
}

// @desc predict samples
async function predictSample(sample) {
  let result = model.predict(tf.tensor(sample, [1,sample.length])).arraySync();
  var maxValue = 0;
  var predictedPitch = 7;
  for (var i = 0; i < NUM_PITCH_CLASSES; i++) {
    if (result[0][i] > maxValue) {
      predictedPitch = i;
      maxValue = result[0][i];
    }
  }
  return pitchFromClassNum(predictedPitch);
}

// @desc  accuracy evaluation for a given pitch class by index
function calcPitchClassEval(pitchIndex, classSize, values) {
  // Output has 7 different class values for each pitch, offset based on
  // which pitch class (ordered by i)
  let index = (pitchIndex * classSize * NUM_PITCH_CLASSES) + pitchIndex;
  let total = 0;
  for (let i = 0; i < classSize; i++) {
    total += values[index];
    index += NUM_PITCH_CLASSES;
  }
  return total / classSize;
}

// @return string value for pitch labels
function pitchFromClassNum(classNum) {
  switch (classNum) {
    case 0:
      return 'Fastball (2-seam)';
    case 1:
      return 'Fastball (4-seam)';
    case 2:
      return 'Fastball (sinker)';
    case 3:
      return 'Fastball (cutter)';
    case 4:
      return 'Slider';
    case 5:
      return 'Changeup';
    case 6:
      return 'Curveball';
    default:
      return 'Unknown';
  }
}

module.exports = {
  evaluate,
  model,
  pitchFromClassNum,
  predictSample,
  testValidationData,
  trainingData,
  TEST_DATA_LENGTH
}