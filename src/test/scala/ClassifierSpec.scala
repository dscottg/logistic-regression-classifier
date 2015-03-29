import org.scalatest._
import breeze.linalg._

class RichStringSpec extends FlatSpec with Matchers {
  "the sigmoid function" should "return the elementwise sigmoid of an array" in {
      val v = DenseVector[Double](0.0, 1.0, 2.0)
      LogisticRegressionClassifier.sigmoid(v) should be (DenseVector(0.5, 0.7310585786300049, 0.8807970779778823))
  }

  "calculate cost" should "calculate the total error using the given weights" in {
      var X = DenseMatrix((0.0, 1.0), (4.0, 5.0))
      var Y = DenseVector(0.0 ,1.0)
      var weights = DenseVector[Double](2.0 ,2.0)

      LogisticRegressionClassifier.calculateCost(X, Y, weights) should equal (1.0634640131364754)
  }
}

