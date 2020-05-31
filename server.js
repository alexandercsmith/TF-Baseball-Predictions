/*
 * @module server
 * @desc   ML server training
 */
require('@tensorflow/tfjs-node');

// @imports
const http = require('http');
const socketio = require('socket.io');
const pitch_type = require('./src/pitch_type.js');

// @config
const TIMEOUT_BETWEEN_EPOCHS_MS = 500;
const PORT = 8001;

// @func sleep()
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// @func run()
// @desc server - perform model training, and emit stats via socket
async function run() {
  const port = process.env.PORT || PORT;
  const server = http.createServer();
  const io = socketio(server);

  server.listen(port, () => {
    console.log(`  > Running socket on port: ${port}`);
  });

  io.on('connection', (socket) => {
    socket.on('predictSample', async (sample) => {
      io.emit('predictResult', await pitch_type.predictSample(sample));
    });
  });

  let numTrainingIterations = 10;
  for (var i = 0; i < numTrainingIterations; i++) {
    console.log(`Training iteration : ${i+1} / ${numTrainingIterations}`);
    await pitch_type.model.fitDataset(pitch_type.trainingData, {epochs: 1});
    console.log('accuracyPerClass', await pitch_type.evaluate(true));
    await sleep(TIMEOUT_BETWEEN_EPOCHS_MS);
  }

  io.emit('trainingComplete', true);
}

run();