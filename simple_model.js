const tf = require('@tensorflow/tfjs-node');

const xTrain = tf.tensor2d([[1], [2], [3], [4], [5]], [5, 1]);
const yTrain = tf.tensor2d([[3], [5], [7], [9], [11]], [5, 1]);

const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

async function trainModel() {
  await model.fit(xTrain, yTrain, { epochs: 100 });
  console.log("Training completed!");
}

trainModel();

// Test
const xTest = tf.tensor2d([[6], [7], [8]], [3, 1]);
const predictions = model.predict(xTest);

predictions.print();
