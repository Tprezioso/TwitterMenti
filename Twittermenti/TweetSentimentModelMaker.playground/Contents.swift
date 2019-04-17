import Cocoa
import CreateML

let data = try MLDataTable(contentsOf: URL(fileURLWithPath: "/Users/ThomasPrezioso/Downloads/twitter-sanders-apple3.csv"))

// Split data 80% to 20% for testing
let(trainingData, testingData) = data.randomSplit(by: 0.8, seed: 5)

let sentimentClassifier = try MLTextClassifier(trainingData: trainingData, textColumn: "text", labelColumn: "class")

let evaluationMetrics = sentimentClassifier.evaluation(on: testingData)

let evaluationAccuracy = (1.0 - evaluationMetrics.classificationError) * 100

let metaData = MLModelMetadata(author: "Thomas Prezioso", shortDescription: "A model trained to classify sentiment on Tweets", version: "1.0")

try sentimentClassifier.write(to: URL(fileURLWithPath: "/Users/ThomasPrezioso/Downloads/TweetSentimentClassifier.mlmodel"))

try sentimentClassifier.prediction(from: "@Apple is a terrible company!")
try sentimentClassifier.prediction(from: "AWESOME! @Apple")
