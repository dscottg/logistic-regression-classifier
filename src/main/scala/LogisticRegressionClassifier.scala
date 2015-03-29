import scala.io.Source
import scala.math
import breeze.linalg._
import math._

object LogisticRegressionClassifier {

    // Calculates the element-wise sigmoid of the given vector
    def sigmoid(t : DenseVector[Double]) : DenseVector[Double] = {
        t.map{ t : Double => (1 / (1 + (pow(E, -1 * t)))) }
    }

    /** Calculate the error using the given feature weights
    *
    * @param X matrix containing the training examples
    * @param Y vector containing the actual results
    * @param weights weight given to each feature
    *
    * @return double giving the amount of error
    */
    def calculateCost(X : DenseMatrix[Double], Y : DenseVector[Double], weights : DenseVector[Double]) = {
        val h_x = sigmoid((X * weights).toDenseVector)
        val term_one = Y :* h_x.map(x=>log(x))
        val term_two = Y.map(1.0-_) :* (h_x.map(1.0-_).map(x=>log(x)))
        val num_features = X.rows

        (term_one + term_two).sum * (-1.0 / num_features).toDouble
    }

    /** Use gradient descent to get the new feature weights.
    *
    * @param X matrix containing the training examples
    * @param Y vector containing the actual results
    * @param weights weight given to each feature
    * @param step_size how large a step we want to take down the error function
    *
    * @return the new weights
    */
    def calculateUpdatedWeights(
        X : DenseMatrix[Double],
        Y : DenseVector[Double],
        current_weights : DenseVector[Double],
        step_size : Double
    ) : DenseVector[Double] = {

        val h_x = sigmoid((X * current_weights).toDenseVector)
        current_weights - ((X.t * (h_x - Y)) :* step_size)
    }

    /** Evaluate the accuracy of the feature weights.
    *
    * @param weights vector containing the feature weights
    * @param examples testing examples
    * @param correct_values the correct binary values we should be predicting
    *
    * @return the true positive rate
    */
    def predict(
        weights : DenseVector[Double], 
        examples : DenseMatrix[Double], 
        correct_values : DenseVector[Double]
    ) : Double = {

        var predicted_values = sigmoid(examples * weights)
        predicted_values = predicted_values.map( x => if (x < 0.5) 0.0 else 1.0 )

        var correct = 0
        for (i <- 0 to predicted_values.length-1) {
            if (predicted_values(i) == correct_values(i)) correct += 1
        }

        correct.toDouble / predicted_values.length.toDouble
    }

    def main(args: Array[String]) {
        val data_filename = "src/main/resources/data/census_data.csv"

        // How much of the data we should use for training vs testing
        val test_train_split = 0.7

        // I tried a few different step sizes and this is one that works well.
        val step_size = 0.001

        // How often we display progress updates about the training
        val output_interval = 20

        // How many steps we should take
        val iterations = 500

        val data_formatter = new FeatureFormatter(data_filename)

        var data_matrix = data_formatter.complete_matrix
        var data_matrix_column_names = data_formatter.complete_matrix_column_names

        // The >50k column is kept as our expected classification values
        val Y = data_matrix(::, data_matrix.cols-1)

        // The last two rows tell whether a person is in the <=50K or the >50k salary range.
        // These rows are redundant because the category is binary, so discard the <=50K column.
        val X = data_matrix(::, 0 to data_matrix.cols-3)

        var num_features = X.cols
        var num_examples = X.rows

        // Set the inital weights to 1
        var weights = DenseVector.ones[Double](num_features)

        // Partition into test and training sets
        var training_cutoff_row = (data_matrix.rows * test_train_split).toInt

        var X_train = X(0 until training_cutoff_row, ::)
        var Y_train = Y(0 until training_cutoff_row)

        var X_test = X(training_cutoff_row until num_examples, ::)
        var Y_test = Y(training_cutoff_row until num_examples)

        println("\n######## TRAINING #########\n")
        for (i <- 1 to iterations) {
            var cost = calculateCost(X_train, Y_train, weights)
            weights = calculateUpdatedWeights(X_train, Y_train, weights, step_size)

            if (i == 1 || (i % output_interval) == 0) {
                val true_positive_rate = predict(weights, X_test, Y_test)
                println("Iteration " + i + ": cost: " + cost + ", accuracy: " + true_positive_rate)
            }
        }

        println("\n######## PREDICTION ACCURACY #########\n")
        val true_positive_rate = predict(weights, X_test, Y_test)
        println(true_positive_rate * 100 + "%")
    }
}
