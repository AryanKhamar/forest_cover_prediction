from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_random_forest, train_xgboost
from src.model_evaluation import evaluate_model

def main():
    print("Loading dataset...")
    df = load_data("data/train.csv")

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    print("Evaluating Random Forest:")
    evaluate_model(rf_model, X_test, y_test)

    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    print("Evaluating XGBoost:")
    evaluate_model(xgb_model, X_test, y_test)

if __name__ == "__main__":
    main()