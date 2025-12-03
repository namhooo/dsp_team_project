"""
모델 평가 관련 함수
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)


def evaluate_model(y_true, y_pred, model_name='Model'):
    """
    모델 성능 종합 평가

    Parameters:
    -----------
    y_true : array-like
        실제값
    y_pred : array-like
        예측값
    model_name : str
        모델 이름

    Returns:
    --------
    dict : 평가 지표 딕셔너리
    """
    metrics = {
        'model': model_name,
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100
    }

    print(f"\n=== {model_name} Performance ===")
    print(f"RMSE: {metrics['RMSE']:,.0f} 명")
    print(f"MAE: {metrics['MAE']:,.0f} 명")
    print(f"R²: {metrics['R2']:.3f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")

    return metrics


def compare_models(results_dict):
    """
    여러 모델의 성능 비교

    Parameters:
    -----------
    results_dict : dict
        {model_name: metrics_dict} 형태의 딕셔너리

    Returns:
    --------
    pd.DataFrame : 모델 비교 결과 테이블
    """
    comparison_df = pd.DataFrame(results_dict).T
    comparison_df = comparison_df.sort_values('R2', ascending=False)

    print("\n=== Model Comparison ===")
    print(comparison_df.to_string())

    return comparison_df


def plot_actual_vs_predicted(y_true, y_pred, model_name='Model', figsize=(10, 6)):
    """
    실제값 vs 예측값 산점도

    Parameters:
    -----------
    y_true : array-like
        실제값
    y_pred : array-like
        예측값
    model_name : str
        모델 이름
    figsize : tuple
        그래프 크기
    """
    plt.figure(figsize=figsize)

    # 산점도
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)

    # 완벽한 예측선 (y=x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    plt.xlabel('Actual Admissions', fontsize=12)
    plt.ylabel('Predicted Admissions', fontsize=12)
    plt.title(f'{model_name}: Actual vs Predicted', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, y_pred, model_name='Model', figsize=(12, 5)):
    """
    잔차(오차) 분석 그래프

    Parameters:
    -----------
    y_true : array-like
        실제값
    y_pred : array-like
        예측값
    model_name : str
        모델 이름
    figsize : tuple
        그래프 크기
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. 잔차 분포 (히스토그램)
    axes[0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Residuals (Actual - Predicted)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title(f'{model_name}: Residual Distribution', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # 2. 잔차 vs 예측값
    axes[1].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Admissions', fontsize=11)
    axes[1].set_ylabel('Residuals', fontsize=11)
    axes[1].set_title(f'{model_name}: Residuals vs Predicted', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, top_n=15, figsize=(10, 6)):
    """
    Feature Importance 시각화

    Parameters:
    -----------
    model : sklearn model
        학습된 모델 (feature_importances_ 속성 필요)
    feature_names : list
        Feature 이름 리스트
    top_n : int
        표시할 상위 Feature 개수
    figsize : tuple
        그래프 크기
    """
    if not hasattr(model, 'feature_importances_'):
        print("This model does not have feature_importances_ attribute")
        return

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # 상위 N개만 표시
    top_features = feature_importance_df.head(top_n)

    plt.figure(figsize=figsize)
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return feature_importance_df


def plot_learning_curve(train_scores, val_scores, metric_name='RMSE', figsize=(10, 6)):
    """
    학습 곡선 시각화

    Parameters:
    -----------
    train_scores : list
        훈련 세트 점수
    val_scores : list
        검증 세트 점수
    metric_name : str
        메트릭 이름
    figsize : tuple
        그래프 크기
    """
    plt.figure(figsize=figsize)

    epochs = range(1, len(train_scores) + 1)

    plt.plot(epochs, train_scores, 'b-o', label=f'Train {metric_name}', linewidth=2)
    plt.plot(epochs, val_scores, 'r-o', label=f'Validation {metric_name}', linewidth=2)

    plt.xlabel('Epoch / Iteration', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'Learning Curve: {metric_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def analyze_prediction_errors(y_true, y_pred, movie_names=None, top_n=10):
    """
    예측 오차가 큰 영화 분석

    Parameters:
    -----------
    y_true : array-like
        실제값
    y_pred : array-like
        예측값
    movie_names : list
        영화 이름 리스트
    top_n : int
        표시할 영화 개수

    Returns:
    --------
    pd.DataFrame : 오차 분석 결과
    """
    errors = np.abs(y_true - y_pred)
    error_df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred,
        'error': errors,
        'error_percentage': (errors / y_true) * 100
    })

    if movie_names is not None:
        error_df['movie'] = movie_names

    # 오차가 큰 순서대로 정렬
    error_df = error_df.sort_values('error', ascending=False)

    print(f"\n=== Top {top_n} Movies with Largest Prediction Errors ===")
    print(error_df.head(top_n).to_string())

    return error_df


def calculate_percentage_within_range(y_true, y_pred, percentage=20):
    """
    특정 오차 범위 내 예측 비율 계산

    Parameters:
    -----------
    y_true : array-like
        실제값
    y_pred : array-like
        예측값
    percentage : float
        허용 오차 비율 (%)

    Returns:
    --------
    float : 범위 내 예측 비율
    """
    errors = np.abs(y_true - y_pred)
    threshold = y_true * (percentage / 100)
    within_range = np.sum(errors <= threshold)
    total = len(y_true)
    ratio = (within_range / total) * 100

    print(f"\nPredictions within {percentage}% error: {ratio:.2f}% ({within_range}/{total})")

    return ratio


def cross_validation_evaluation(model, X, y, cv=5, scoring=None):
    """
    Cross-Validation 평가

    Parameters:
    -----------
    model : sklearn model
        평가할 모델
    X : array-like
        Feature 데이터
    y : array-like
        Target 데이터
    cv : int
        Fold 개수
    scoring : list
        평가 지표 리스트

    Returns:
    --------
    dict : Cross-Validation 결과
    """
    from sklearn.model_selection import cross_validate

    if scoring is None:
        scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']

    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True
    )

    print(f"\n=== {cv}-Fold Cross-Validation Results ===")

    for metric in scoring:
        test_key = f'test_{metric}'
        if test_key in cv_results:
            scores = cv_results[test_key]

            # MSE는 음수로 반환되므로 처리
            if 'neg_' in metric:
                scores = -scores
                metric_name = metric.replace('neg_', '').upper()

                # RMSE 계산
                if 'mean_squared_error' in metric:
                    scores = np.sqrt(scores)
                    metric_name = 'RMSE'
            else:
                metric_name = metric.upper()

            print(f"{metric_name}: {scores.mean():,.2f} (+/- {scores.std():,.2f})")

    return cv_results


def save_evaluation_results(results_dict, filepath):
    """
    평가 결과를 CSV로 저장

    Parameters:
    -----------
    results_dict : dict
        평가 결과 딕셔너리
    filepath : str
        저장 경로
    """
    results_df = pd.DataFrame([results_dict])
    results_df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"\nEvaluation results saved to {filepath}")


def create_model_comparison_plot(comparison_df, metric='R2', figsize=(10, 6)):
    """
    모델 성능 비교 그래프

    Parameters:
    -----------
    comparison_df : pd.DataFrame
        모델 비교 결과 데이터프레임
    metric : str
        비교할 메트릭
    figsize : tuple
        그래프 크기
    """
    plt.figure(figsize=figsize)

    models = comparison_df.index
    values = comparison_df[metric]

    bars = plt.bar(range(len(models)), values, color='skyblue', edgecolor='black')

    # 최고 성능 모델 강조
    best_idx = values.argmax() if metric == 'R2' else values.argmin()
    bars[best_idx].set_color('green')

    plt.xlabel('Model', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.title(f'Model Comparison: {metric}', fontsize=14, fontweight='bold')
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Model Evaluation Module")
    print("Usage:")
    print("  from utils.evaluation import evaluate_model, plot_actual_vs_predicted")
