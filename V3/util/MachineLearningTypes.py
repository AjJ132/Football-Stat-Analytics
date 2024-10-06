from enum import Enum

class MachineLearningType(Enum):
    LINEAR_REGRESSION = "Linear Regression"
    RANDOM_FOREST = "Random Forest"
    GRADIENT_BOOSTING = "Gradient Boosting"
    RIDGE_REGRESSION = "Ridge Regression"
    LASSO_REGRESSION = "Lasso Regression"
    SVR = "Support Vector Regression"
    NEURAL_NETWORKS = "Neural Networks"


class Type:
    def __init__(self, ml_type: MachineLearningType):
        if not isinstance(ml_type, MachineLearningType):
            raise ValueError("Invalid machine learning type")
        self.ml_type = ml_type

    def __str__(self):
        return f"Selected Machine Learning Type: {self.ml_type.value}"
