import pandas as pd

df = pd.read_csv("product_quality_data.csv")

print("최초 2개 출력(조건1):\n", df.head(2)) # 조건 1: 최초 2개 데이터 출력
print("행 개수(조건2):",df.shape[0]) # 조건 2: 전체 데이터 수 출력하기


from sklearn.model_selection import train_test_split

input_data = df[['length', 'weight','thickness']]
target_data = df['result']

train_input, test_input, train_target,test_target =train_test_split(
    input_data,
    target_data,
    test_size=0.2,
    random_state=42
)

# K-최근접 이웃 모델 만들기
from sklearn.neighbors import KNeighborsClassifier

# K-NN 분류기 객체 생성 (k=5 기본값)
# 기본 이웃 수(k)는 5개
kn = KNeighborsClassifier()

kn.fit(train_input, train_target)

# 테스트 세트
print("테스트 세트:",kn.score(test_input, test_target))
print()

pred = kn.predict(test_input)

# 조건 3
from sklearn.metrics import f1_score

print("(조건3)")
print("F1 점수:",f1_score(test_target, pred, pos_label='good'))


from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

f1_scorer = make_scorer(f1_score, pos_label='good')

scores = cross_val_score(
    kn,
    input_data,
    target_data,
    cv=5,
    scoring=f1_scorer,
)

print("5-Fold 교차검증 평균 F1 점수:", scores.mean())
print()

# 조건 4
# “학습할 때는 length, weight, thickness라는 이름이 있었는데, 지금 예측 데이터에는 그 이름이 없네요?”
# DataFrame 추가
new_data = pd.DataFrame(
    [[100.5,50.0,5.0]],
    columns=['length','weight','thickness']
)

result = kn.predict(new_data)

print("새로운 제품 예측 결과(조건4):",result[0])
