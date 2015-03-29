import scala.io.Source
import scala.math
import breeze.linalg._
import scala.collection.mutable._
import math._

/*

Takes CSV data read in line-by-line and turns it into a
scaled matrix containing values between 0 and 1

NOTE: First row must contain the column names.

1. All numeric values are assumed to continuous and are scaled relative 
to the largest value for that feature. 

2. Categorical data is encoded using one-hot encoding, and new columns are added
so every category is represented by a feature containing a 1 if the example 
belongs to that category, or a 0 if not. See code comments for an example.

*/

class InconsistentRowSizeException (message : String) extends Exception (message : String) {}

class FeatureFormatter (filename : String) {

    val csv_lines = new ArrayBuffer[String]()

    // The original matrix loaded in from our file.
    // Matrix is M x N where M = number of examples and N = number of features.
    var original_matrix : DenseMatrix[String] = null
    var original_column_names : Array[String] = null

    // The column indexes of orignal matrix that contain numeric values.
    // Because these values are numeric, we assume they are continuous.
    var numeric_columns = new ArrayBuffer[Int]()

    // The column indexes of orignal matrix that have non-numeric values.
    // Because these values are non-numeric we assume they are categorical.
    var categorical_columns = new ArrayBuffer[Int]()

    // A matrix holding our scaled numeric values
    var numeric_matrix : DenseMatrix[Double] = null
    var numeric_matrix_column_names = new ArrayBuffer[String]()

    // A matrix holding one-hot encoded categorical data
    var one_hot_categorical_matrix : DenseMatrix[Double] = null
    var one_hot_categorical_matrix_column_names = new ArrayBuffer[String]()

    // The numeric and one-hot matrices combined together.
    // This is the matrix we will use for our training.
    var complete_matrix : DenseMatrix[Double] = null
    var complete_matrix_column_names : Array[String] = null

    var row_count = 0
    var column_count = 0

    init()

    def getDimensions() : Map[String, Int] = {
        Map("rows" -> row_count, "columns" -> column_count)
    }

    def init() {
        loadLinesFromFile()
        loadLinesIntoMatrix()
        categorizeColumns()
        createScaledNumericMatrix()
        createdOneHotEncodedCategoricalMatrix()
        combineCategoricalAndNumeric()
        addInterceptTerms()
    }

    // Load the lines into an array as strings
    private def loadLinesFromFile() {
        var first_line = true

        for (line <- Source.fromFile(filename).getLines()) {
            var columns = line.split(", *")
            if (first_line) { // first row holds the column names
                original_column_names = columns
                column_count = columns.length
                first_line = false
            } else if (columns.length > 1) { // skip rows without any commas
                csv_lines += line

                val current_line_column_count = line.split(", *").length // numer of columns in this row
                if (column_count != current_line_column_count) {
                    // every row should have the same number of columns
                    throw new InconsistentRowSizeException("All rows should have the same number of columns. Found mismatch at line " +
                        row_count+ ". Expected " + column_count + " columns, found " + current_line_column_count)
                }
            }
        }

        row_count = csv_lines.length
    }

    // Split the lines apart on commas and use these values as the matrix rows
    private def loadLinesIntoMatrix() {
        original_matrix = new DenseMatrix[String](row_count, column_count)

        for (i <- 0 to row_count - 1; columns = csv_lines(i).split(", *")) {
            for (j <- 0 to column_count - 1; data = columns(j)) {
                original_matrix(i, j) = data
            }
        }
    }

    // Figure out if the column is numeric or categorical
    // eg. 1, 2, dog, cat, 4 will create two arrays
    // numeric_columns => [0, 1, 4]
    // categorical_columns => [2, 3]
    private def categorizeColumns() {
        // We'll assume the first training example is an indicator of the type
        // for each of the remaining examples
        val first_row_features = original_matrix(0, ::).t.toArray
        first_row_features.indices.foreach{
            i  =>
                if (isNumber(first_row_features(i))) {
                    numeric_columns += i
                } else {
                    categorical_columns += i
                }
        }
    }

    // Create a matrix with the scaled numeric values. Scale each feature so it's never
    // larger than one (divide every element by the max for that column)
    private def createScaledNumericMatrix() {
        numeric_matrix = new DenseMatrix[Double](row_count, numeric_columns.length)

        // add column names
        numeric_matrix_column_names = numeric_columns.map{ original_column_names(_) }

        for (row_index <- 0 until row_count) {
            for (col_index <- 0 until numeric_columns.length; old_matrix_col_index = numeric_columns(col_index)) {
                numeric_matrix(row_index, col_index) = original_matrix(row_index, old_matrix_col_index).toDouble
            }
        }

        // scale each value to the largest value for that column
        for (column_index <- 0 until numeric_matrix.cols) {
            val max_value : Double = numeric_matrix(::, column_index).max

            for (row_index <- 0 until row_count) {
                numeric_matrix(row_index, column_index) /= max_value
            }
        }
    }

    // Example one-hot matrix conversion:
    // animal, habitat       animal_zebra, animal_tiger, habitat_savannah, habitat_woods
    // zebra,  savannah  =>             1,            0                 1,             0
    // tiger,  savannah                 0             1,                0,             0
    // tiger,  forest                   0             1,                0,             1
    //
    // That's what this function does, takes a categorical column and turns it into one-hot encoded columns
    private def createdOneHotEncodedCategoricalMatrix() {
        one_hot_categorical_matrix = null // in case there's already data in here (which there shouldn't be)

        categorical_columns.foreach{ i =>
            // get just the unique values for this column (the categories)
            val column = original_matrix(::, i).toArray
            val column_categories : Array[String] = column.distinct
            column_categories.foreach{ category =>
                one_hot_categorical_matrix_column_names += original_column_names(i) + "_" + category
            }

            var temp_one_hot_matrix = DenseMatrix.zeros[Double](row_count, column_categories.length)

            var row_index = 0
            column.foreach{ entry =>
                temp_one_hot_matrix(row_index, column_categories.indexOf(entry)) = 1.0
                row_index += 1
            }

            if (one_hot_categorical_matrix == null) {
                one_hot_categorical_matrix = temp_one_hot_matrix
            } else {
                one_hot_categorical_matrix = DenseMatrix.horzcat(one_hot_categorical_matrix, temp_one_hot_matrix)
            }
        }
    }

    // Combine the separate categorical and numeric matrices into one matrix
    private def combineCategoricalAndNumeric() {
        if (numeric_matrix == null) {
            complete_matrix_column_names = one_hot_categorical_matrix_column_names.toArray
            complete_matrix = one_hot_categorical_matrix
        } else if (one_hot_categorical_matrix == null) {
            complete_matrix_column_names = numeric_matrix_column_names.toArray
            complete_matrix = numeric_matrix
        } else {
            complete_matrix = DenseMatrix.horzcat(numeric_matrix, one_hot_categorical_matrix)
            complete_matrix_column_names = (numeric_matrix_column_names ++ one_hot_categorical_matrix_column_names).toArray
        }
    }

    // Prepend the array with 1s for the intercept term; update the column names appropriately
    private def addInterceptTerms() {
        complete_matrix = DenseMatrix.horzcat(DenseMatrix.ones[Double](row_count, 1), complete_matrix)
        complete_matrix_column_names = "intercept" +: complete_matrix_column_names
    }

    private def isNumber(x: String) = x forall { x => x.isDigit || x == '.' || x == '-' }

}
