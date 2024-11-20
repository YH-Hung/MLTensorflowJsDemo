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
    learningRate: 0.01,
    iterations: 20,
    batchSize: 10
});

regression.train();
const r2 = regression.test(testFeatures, testLabels);

plot({
    x: regression.mseHistory.reverse(),
    xLabel: 'Iteration #',
    yLabel: 'Mean Squared Error'
});

console.log('R2 is', r2);

regression.predict([[120, 2, 380]]).print();
