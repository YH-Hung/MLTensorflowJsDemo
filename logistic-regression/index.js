require('@tensorflow/tfjs-node');
const loadCSV = require('../load-csv');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');

let { features, labels, testFeatures, testLabels } = loadCSV('./data/cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['passedemissions'],
    converters: {
        passedemissions: (value) => value === 'TRUE' ? 1 : 0
    }
});

const regression = new LogisticRegression(features, labels, {
    learningRate: 0.05,
    iterations: 20,
    batchSize: 10,
    decisionBoundary: 0.6,
});

regression.train();
const accuracy = regression.test(testFeatures, testLabels);

console.log(accuracy)

plot({
    x: regression.costHistory.reverse(),
    xLabel: 'Iteration #',
    yLabel: 'Cost'
});
