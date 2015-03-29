import org.scalatest._
import breeze.linalg._

class FeatureFormatterSpec extends FlatSpec with Matchers {

  "the feature formatter" should "throw an exception for inconsistent row lengths" in {
    evaluating {
        new FeatureFormatter("src/test/scala/data/inconsistent_row_lengths.txt")
    } should produce [InconsistentRowSizeException]
  }

  "the feature formatter" should "give the correct dimensions of the data" in {
    val formatted = new FeatureFormatter("src/test/scala/data/dimension_test_data.txt")
    val dimensions = formatted.getDimensions()

    dimensions("rows") should equal (10)
    dimensions("columns") should equal (5)
  }

  "the feature formatter" should "return the correct original column names" in {
    val formatted = new FeatureFormatter("src/test/scala/data/dimension_test_data.txt")
    formatted.original_column_names should equal (Array("a", "b", "c", "d", "e"))
  }

  "the feature formatter" should "load all the data into a densematrix of strings" in {
    val formatted = new FeatureFormatter("src/test/scala/data/mixed_data.txt")
    var expected_matrix : DenseMatrix[String] = DenseMatrix(
        Array("1", "visa", "3", "8", "5", "cat"),
        Array("1", "mastercard", "6", "4", "25", "cat")
    )

    formatted.original_matrix should equal (expected_matrix)
  }

  "the feature formatter" should "categorize the columns and numeric or categorical values" in {
    val formatted = new FeatureFormatter("src/test/scala/data/mixed_data.txt")
    formatted.numeric_columns should equal (Array(0, 2, 3, 4))
    formatted.categorical_columns should equal (Array(1, 5))
  }

  "the feature formatter" should "create a matrix of Doubles and scale the values" in {
    val formatted = new FeatureFormatter("src/test/scala/data/scaled_numeric_matrix_data.txt")
    var expected_matrix : DenseMatrix[Double] = DenseMatrix(
        Array(0.25, .25),
        Array(.25, .50),
        Array(.50, .50),
        Array(1.0, 1.0)
    )

    formatted.numeric_matrix should equal (expected_matrix)
  }

  "the feature formatter" should "create a one-hot encoded categorical matrix" in {
    val formatted = new FeatureFormatter("src/test/scala/data/one_hot_encoded_categorical_data.txt")
    var expected_matrix = DenseMatrix(
        Array(1.0, 0.0, 0.0, 1.0, 0.0),
        Array(0.0, 1.0, 0.0, 0.0, 1.0),
        Array(0.0, 0.0, 1.0, 0.0, 1.0)
    )
    formatted.one_hot_categorical_matrix should equal (expected_matrix)
  }

  "the feature formatter" should "combine encoded categorical data and numeric data into one matrix" in {
    val formatted = new FeatureFormatter("src/test/scala/data/mixed_data.txt")
    var expected_matrix = DenseMatrix(
        Array(1.0, 1.0, 0.5, 1.0, 0.2, 1.0, 0.0, 1.0),
        Array(1.0, 1.0, 1.0, 0.5, 1.0, 0.0, 1.0, 1.0)
    )

    // a, credit_card-visa, credit_card-mastercard, c, d, e, animal-cat
    formatted.complete_matrix should equal (expected_matrix)
  }

  "the feature formatter" should "return the correct column names after the data is combined" in {
    val formatted = new FeatureFormatter("src/test/scala/data/mixed_data.txt")
    var expected_column_names = Array("intercept", "quantity", "width", "height", "length", "credit_card_visa", "credit_card_mastercard", "animal_cat")

    formatted.complete_matrix_column_names should equal (expected_column_names)
  }

}
