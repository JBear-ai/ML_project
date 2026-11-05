# - 여러 개의 결정 트리를 무작위로 구성해 예측을 종합하는 앙상블 모델입니다.
# - 분류 문제에서는 다수결 투표, 회귀 문제에서는 평균으로 예측을 결정합니다.
# - 결정 트리의 과적합, 민감함, 일반화 부족 문제를 해결
# - 강한 특성 하나가 모든 트리를 똑같이 만드는 것을 막아, 트리들이 서로 다르게 만들어 짐
# - 배깅: 데이터를 여러 번 샘플링에 각 모델을 독립적으로 학습 (랜덤 포레스트)
# - 부스팅: 이전 모델의 오류를 보완하면서 순차적으로 학습 (AdaBoost, Gradient Boosting, XGBoost)


# 데이터 로딩 및 분할
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
train_input, test_input, train_target, test_target = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

# 랜덤 포레스트 모델 훈련
from sklearn.ensemble import RandomForestClassifier
# n_estimators= 100 : 100개의 결정 트리를 만듬, 각 트리는 데이터와 특성 모두 무작위로 선택해 구성
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(train_input, train_target)

# 성능 평가
print(f"훈련 정확도: {rf.score(train_input, train_target):.3f}")
print(f"테스트 정확도: {rf.score(test_input, test_target):.3f}")

# 특성 중요도 확인
for name, score in zip(wine.feature_names, rf.feature_importances_):
    print(f"{name:30} : {score:.3f}")

# 하나의 트리 시각화
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
plot_tree(rf.estimators_[0], filled=True, feature_names=wine.feature_names, class_names=wine.target_names)
plt.title("랜덤 포레스트 내부 트리 예시")
plt.show()
