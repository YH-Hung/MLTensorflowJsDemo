require('@tensorflow/tfjs-node');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');
const _ = require('lodash');
const mnist = require('mnist-data')

function loadData() {
    const mnistData = mnist.training(0, 60000)

    const features = mnistData.images.values.map(image => _.flatMap(image));
    const encodedLabels = mnistData.labels.values.map(label => {
        const row = new Array(10).fill(0)
        row[label] = 1
        return row
    })

    return {features, encodedLabels};
}

const { features, encodedLabels } = loadData();

const regression = new LogisticRegression(features, encodedLabels, {
    learningRate: 0.1,
    iterations: 50,
    batchSize: 10,
});

regression.train();

const testMnistData = mnist.testing(0, 1000)

const testFeatures = testMnistData.images.values.map(image => _.flatMap(image));
const testEncodedLabels = testMnistData.labels.values.map(label => {
    const row = new Array(10).fill(0)
    row[label] = 1
    return row
})

const accu = regression.test(testFeatures, testEncodedLabels)

console.log(accu)