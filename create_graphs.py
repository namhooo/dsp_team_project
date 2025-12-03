"""
영화 관객 수 예측 프로젝트 - 결과 시각화 그래프 생성
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Apple SD Gothic Neo'
plt.rcParams['axes.unicode_minus'] = False

# 스타일 설정
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# 데이터 로드
baseline_data = pd.read_csv('results/tables/baseline_results.csv')
model_comparison = pd.read_csv('results/tables/model_comparison.csv')

# 모델명 정리
model_comparison.iloc[:, 0] = model_comparison.iloc[:, 0].str.replace(' (기본)', '')

print("그래프 생성 중...\n")

# ============================================
# 1. 모델별 MAE 비교 (Bar Chart)
# ============================================
plt.figure(figsize=(12, 7))
colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(model_comparison))]
bars = plt.bar(range(len(model_comparison)), 
               model_comparison['Val_MAE'], 
               color=colors, 
               edgecolor='black', 
               linewidth=1.5,
               alpha=0.8)

plt.xticks(range(len(model_comparison)), 
           model_comparison.iloc[:, 0], 
           rotation=45, 
           ha='right',
           fontsize=11)
plt.ylabel('Validation MAE', fontsize=13, fontweight='bold')
plt.title('Model-wise Validation MAE Comparison', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, axis='y')

# 값 표시
for i, (bar, val) in enumerate(zip(bars, model_comparison['Val_MAE'])):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:,.0f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# 최고 성능 표시
plt.text(0, model_comparison['Val_MAE'].iloc[0] * 1.05, '⭐ Best', 
         ha='center', fontsize=12, fontweight='bold', color='#27ae60')

plt.tight_layout()
plt.savefig('results/figures/model_mae_comparison.png', dpi=300, bbox_inches='tight')
print("✅ 저장: results/figures/model_mae_comparison.png")

# ============================================
# 2. 모델별 R² 비교 (Bar Chart)
# ============================================
plt.figure(figsize=(12, 7))
colors_r2 = ['#e74c3c' if val < 0.4 else '#f39c12' if val < 0.5 else '#2ecc71' 
             for val in model_comparison['Val_R2']]
bars = plt.bar(range(len(model_comparison)), 
               model_comparison['Val_R2'], 
               color=colors_r2, 
               edgecolor='black', 
               linewidth=1.5,
               alpha=0.8)

plt.xticks(range(len(model_comparison)), 
           model_comparison.iloc[:, 0], 
           rotation=45, 
           ha='right',
           fontsize=11)
plt.ylabel('Validation R² Score', fontsize=13, fontweight='bold')
plt.title('모델별 Validation R² 비교 (높을수록 좋음)', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, axis='y')
plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='R²=0.5 기준선')

# 값 표시
for bar, val in zip(bars, model_comparison['Val_R2']):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.3f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('results/figures/model_r2_comparison.png', dpi=300, bbox_inches='tight')
print("✅ 저장: results/figures/model_r2_comparison.png")

# ============================================
# 3. Baseline vs Best Model 비교
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# MAE 비교
baseline_mae = baseline_data['val_mae'][0]
best_mae = model_comparison['Val_MAE'].iloc[0]
improvement = (baseline_mae - best_mae) / baseline_mae * 100

axes[0].bar(['Baseline\n(Linear Regression)', f'Best Model\n({model_comparison.iloc[0, 0]})'], 
            [baseline_mae, best_mae],
            color=['#95a5a6', '#2ecc71'],
            edgecolor='black',
            linewidth=2,
            alpha=0.8,
            width=0.6)
axes[0].set_ylabel('Validation MAE', fontsize=12, fontweight='bold')
axes[0].set_title('Baseline vs Best Model (MAE)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

for i, (label, val) in enumerate(zip(['Baseline', 'Best'], [baseline_mae, best_mae])):
    axes[0].text(i, val, f'{val:,.0f}명', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')

# 개선율 표시
axes[0].annotate('', xy=(1, best_mae), xytext=(0, baseline_mae),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
axes[0].text(0.5, (baseline_mae + best_mae) / 2, f'↓ {improvement:.1f}% improvement',
            ha='center', fontsize=12, fontweight='bold', color='red',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

# R² 비교
baseline_r2 = baseline_data['val_r2'][0]
best_r2 = model_comparison['Val_R2'].iloc[0]

axes[1].bar(['Baseline\n(Linear Regression)', f'Best Model\n({model_comparison.iloc[0, 0]})'], 
            [baseline_r2, best_r2],
            color=['#95a5a6', '#3498db'],
            edgecolor='black',
            linewidth=2,
            alpha=0.8,
            width=0.6)
axes[1].set_ylabel('Validation R²', fontsize=12, fontweight='bold')
axes[1].set_title('Baseline vs Best Model (R²)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5)

for i, (label, val) in enumerate(zip(['Baseline', 'Best'], [baseline_r2, best_r2])):
    axes[1].text(i, val, f'{val:.3f}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/baseline_vs_best.png', dpi=300, bbox_inches='tight')
print("✅ 저장: results/figures/baseline_vs_best.png")

# ============================================
# 4. 모델 성능 종합 비교 (MAE + R² 함께)
# ============================================
fig, ax1 = plt.subplots(figsize=(14, 7))

x = np.arange(len(model_comparison))
width = 0.35

# MAE (왼쪽 축)
ax1.set_xlabel('model', fontsize=12, fontweight='bold')
ax1.set_ylabel('Validation MAE', fontsize=12, fontweight='bold', color='#2c3e50')
bars1 = ax1.bar(x - width/2, model_comparison['Val_MAE'], width, 
                label='MAE', color='#3498db', alpha=0.8, edgecolor='black')
ax1.tick_params(axis='y', labelcolor='#2c3e50')
ax1.grid(True, alpha=0.3, axis='y')

# R² (오른쪽 축)
ax2 = ax1.twinx()
ax2.set_ylabel('Validation R²', fontsize=12, fontweight='bold', color='#e74c3c')
bars2 = ax2.bar(x + width/2, model_comparison['Val_R2'], width, 
                label='R²', color='#e74c3c', alpha=0.8, edgecolor='black')
ax2.tick_params(axis='y', labelcolor='#e74c3c')

# X축 레이블
ax1.set_xticks(x)
ax1.set_xticklabels(model_comparison.iloc[:, 0], rotation=45, ha='right', fontsize=11)

# 제목과 범례
ax1.set_title('모델별 MAE와 R² 종합 비교', fontsize=16, fontweight='bold', pad=20)
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9), fontsize=11)

plt.tight_layout()
plt.savefig('results/figures/model_combined_comparison.png', dpi=300, bbox_inches='tight')
print("✅ 저장: results/figures/model_combined_comparison.png")

# ============================================
# 5. 학습 시간 비교 (있는 경우)
# ============================================
if 'Training_Time' in model_comparison.columns:
    # NaN이 아닌 값만 필터링
    time_data = model_comparison[model_comparison['Training_Time'].notna()].copy()
    
    if len(time_data) > 0:
        plt.figure(figsize=(12, 7))
        colors_time = plt.cm.viridis(np.linspace(0, 1, len(time_data)))
        bars = plt.bar(range(len(time_data)), 
                      time_data['Training_Time'], 
                      color=colors_time, 
                      edgecolor='black', 
                      linewidth=1.5,
                      alpha=0.8)
        
        plt.xticks(range(len(time_data)), 
                  time_data.iloc[:, 0], 
                  rotation=45, 
                  ha='right',
                  fontsize=11)
        plt.ylabel('학습 시간 (초)', fontsize=13, fontweight='bold')
        plt.title('모델별 학습 시간 비교', fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 값 표시
        for bar, val in zip(bars, time_data['Training_Time']):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}s',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/figures/model_training_time.png', dpi=300, bbox_inches='tight')
        print("✅ 저장: results/figures/model_training_time.png")

# ============================================
# 6. 성능-시간 트레이드오프 (산점도)
# ============================================
if 'Training_Time' in model_comparison.columns:
    time_data = model_comparison[model_comparison['Training_Time'].notna()].copy()
    
    if len(time_data) > 0:
        plt.figure(figsize=(12, 8))
        
        # 산점도
        scatter = plt.scatter(time_data['Training_Time'], 
                            time_data['Val_MAE'],
                            s=300,
                            c=time_data['Val_R2'],
                            cmap='RdYlGn',
                            edgecolors='black',
                            linewidth=2,
                            alpha=0.8)
        
        # 컬러바
        cbar = plt.colorbar(scatter)
        cbar.set_label('R² Score', fontsize=12, fontweight='bold')
        
        # 모델명 표시
        for idx, row in time_data.iterrows():
            plt.annotate(row.iloc[0],
                        (row['Training_Time'], row['Val_MAE']),
                        fontsize=10,
                        fontweight='bold',
                        xytext=(10, 10),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        plt.xlabel('학습 시간 (초)', fontsize=13, fontweight='bold')
        plt.ylabel('Validation MAE (명)', fontsize=13, fontweight='bold')
        plt.title('모델 성능 vs 학습 시간 (색상: R² Score)', fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/figures/performance_time_tradeoff.png', dpi=300, bbox_inches='tight')
        print("✅ 저장: results/figures/performance_time_tradeoff.png")

print("\n" + "="*60)
print("✅ 모든 그래프 생성 완료!")
print("="*60)
print("\n저장 위치: results/figures/")
print("  - model_mae_comparison.png")
print("  - model_r2_comparison.png")
print("  - baseline_vs_best.png")
print("  - model_combined_comparison.png")
if 'Training_Time' in model_comparison.columns:
    print("  - model_training_time.png")
    print("  - performance_time_tradeoff.png")
