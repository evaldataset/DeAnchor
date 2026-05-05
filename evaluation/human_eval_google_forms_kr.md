# 금융 사기 탐지 AI 설명 품질 평가

## Google Forms 구성 가이드

아래 내용을 Google Forms에 복사하여 설문을 만드세요.
- Google Forms: https://docs.google.com/forms

---

## [섹션 1] 안내 및 동의

**양식 제목:** 금융 사기 탐지 AI 설명 품질 평가

**양식 설명:**
> 본 설문은 금융 사기 탐지 AI 시스템이 생성한 설명의 품질을 평가하기 위한 학술 연구입니다.
> 머신러닝 모델이 의심스러운 거래로 플래깅한 거래에 대해, 세 가지 다른 방식으로 생성된 설명을 읽고 품질을 평가해 주세요.
>
> - 소요시간: 약 15-20분
> - 총 10개 거래, 각 거래당 3개 설명 평가
> - 모든 응답은 익명 처리됩니다
> - 정답은 없습니다. 전문가로서의 직관적 판단을 부탁드립니다.
>
> **평가 기준 (1-5점):**
> - 1점 = 매우 나쁨: 도움 안 됨, 혼란스러움
> - 2점 = 나쁨: 정보는 있으나 핵심 누락
> - 3점 = 보통: 합리적이나 구체성 부족
> - 4점 = 좋음: 구체적이고 유용, 사소한 문제만 있음
> - 5점 = 매우 좋음: 즉시 조사에 활용 가능

---

## [섹션 2] 평가자 정보

**Q. 주요 직무 분야** (객관식, 필수)
- 금융 사기 조사 / BSA / AML
- 금융 데이터 분석
- ML/AI 엔지니어링
- 금융학 연구/교수
- 데이터 과학 (비금융)
- 기타

**Q. 금융 사기 탐지 관련 경력** (객관식, 필수)
- 경험 없음
- 1년 미만
- 1-3년
- 3-10년
- 10년 이상

**Q. SHAP/LIME 등 모델 설명 도구 사용 경험** (객관식, 필수)
- 전혀 모른다
- 들어본 적 있다
- 사용해 본 적 있다
- 업무에서 자주 사용한다

---

## [섹션 3] 거래 1 / 10

**거래 정보** (이 텍스트를 섹션 설명에 삽입)

> ML 사기 확률 점수: **0.8384**
>
> Transaction Amount: $335.00 (typical range), round number
> Time: evening (18:00), weekday
> Product: W
> Email domain: verizon.net
> Time deltas: 4 available

**설명 A** (아래 내용을 질문 설명에 삽입)

```
ML Fraud Score: 0.8384

FACTORS INCREASING FRAUD RISK:
  - D2 = 0.00 (SHAP: +0.647, increases fraud risk)
  - card6 = 1.00 (SHAP: +0.529, increases fraud risk)
  - TransactionAmt = 335.00 (SHAP: +0.430, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - C14 = 3.00 (SHAP: -0.487, decreases fraud risk)
  - D10 = 286.00 (SHAP: -0.439, decreases fraud risk)
  - P_emaildomain = 49.00 (SHAP: -0.3
```

**Q. 설명 A의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**설명 B** (아래 내용을 질문 설명에 삽입)

```
The evidence for fraud is moderate, primarily due to the round number and the high fraud score. However, the timing and typical transaction amount provide some counter-evidence suggesting legitimacy.
Recommendation: HOLD_FOR_REVIEW
Fraud likelihood: 0.8385
```

**Q. 설명 B의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**설명 C** (아래 내용을 질문 설명에 삽입)

```
### Quantitative Evidence (SHAP)
ML Fraud Score: 0.8384

FACTORS INCREASING FRAUD RISK:
  - D2 = 0.00 (SHAP: +0.647, increases fraud risk)
  - card6 = 1.00 (SHAP: +0.529, increases fraud risk)
  - TransactionAmt = 335.00 (SHAP: +0.430, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - C14 = 3.00 (SHAP: -0.487, decreases fraud risk)
  - D10 = 286.00 (SHAP: -0.439, decreases fraud risk)
  - P_emaildomain = 49.00 (SHAP: -0.399, decreases fraud risk)

Overall: 5 features increase risk, 5 fea
```

**Q. 설명 C의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 종합 순위** (객관식 그리드, 필수)
  - 질문: "세 가지 설명 중 가장 유용한 것부터 순위를 매겨 주세요"
  - 1순위 (가장 유용): A / B / C
  - 2순위: A / B / C
  - 3순위: A / B / C

**Q. 추가 의견** (단답형, 선택)
  - "순위를 매긴 이유나 특정 설명에 대한 의견을 자유롭게 적어 주세요."

---

## [섹션 4] 거래 2 / 10

**거래 정보** (이 텍스트를 섹션 설명에 삽입)

> ML 사기 확률 점수: **0.9267**
>
> Transaction Amount: $23.20 (typical range)
> Time: late night/early morning (0:00), weekday
> Product: C
> Email domain: gmail.com
> Time deltas: 5 available

**설명 A** (아래 내용을 질문 설명에 삽입)

```
ML Fraud Score: 0.9267

FACTORS INCREASING FRAUD RISK:
  - C14 = 0.00 (SHAP: +1.165, increases fraud risk)
  - C13 = 0.00 (SHAP: +0.619, increases fraud risk)
  - card6 = 1.00 (SHAP: +0.435, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - TransactionAmt = 23.20 (SHAP: -0.482, decreases fraud risk)
  - C1 = 1.00 (SHAP: -0.380, decreases fraud risk)
  - card1 = 3867.00 (SHAP: -0.149, decre
```

**Q. 설명 A의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**설명 B** (아래 내용을 질문 설명에 삽입)

```
The evidence against fraud is weak and does not outweigh the high fraud score provided by the ML model.
Recommendation: HOLD_FOR_REVIEW
Fraud likelihood: 0.9274
```

**Q. 설명 B의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**설명 C** (아래 내용을 질문 설명에 삽입)

```
### Quantitative Evidence (SHAP)
ML Fraud Score: 0.9267

FACTORS INCREASING FRAUD RISK:
  - C14 = 0.00 (SHAP: +1.165, increases fraud risk)
  - C13 = 0.00 (SHAP: +0.619, increases fraud risk)
  - card6 = 1.00 (SHAP: +0.435, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - TransactionAmt = 23.20 (SHAP: -0.482, decreases fraud risk)
  - C1 = 1.00 (SHAP: -0.380, decreases fraud risk)
  - card1 = 3867.00 (SHAP: -0.149, decreases fraud risk)

Overall: 5 features increase risk, 5 features dec
```

**Q. 설명 C의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 종합 순위** (객관식 그리드, 필수)
  - 질문: "세 가지 설명 중 가장 유용한 것부터 순위를 매겨 주세요"
  - 1순위 (가장 유용): A / B / C
  - 2순위: A / B / C
  - 3순위: A / B / C

**Q. 추가 의견** (단답형, 선택)
  - "순위를 매긴 이유나 특정 설명에 대한 의견을 자유롭게 적어 주세요."

---

## [섹션 5] 거래 3 / 10

**거래 정보** (이 텍스트를 섹션 설명에 삽입)

> ML 사기 확률 점수: **0.8799**
>
> Transaction Amount: $117.00 (typical range), round number
> Time: morning (9:00), weekday
> Product: W
> Email domain: aol.com
> Time deltas: 7 available

**설명 A** (아래 내용을 질문 설명에 삽입)

```
ML Fraud Score: 0.8799

FACTORS INCREASING FRAUD RISK:
  - card6 = 1.00 (SHAP: +0.643, increases fraud risk)
  - hour = 9.00 (SHAP: +0.640, increases fraud risk)
  - C6 = 3.00 (SHAP: +0.400, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - P_emaildomain = 2.00 (SHAP: -0.307, decreases fraud risk)
  - card1 = 15066.00 (SHAP: -0.249, decreases fraud risk)
  - C1 = 1.00 (SHAP: -0.244, decrea
```

**Q. 설명 A의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**설명 B** (아래 내용을 질문 설명에 삽입)

```
The evidence against fraud is relatively strong, indicating that this transaction may be legitimate. However, the high fraud score from the ML model suggests caution.
Recommendation: HOLD_FOR_REVIEW
Fraud likelihood: 0.8796
```

**Q. 설명 B의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**설명 C** (아래 내용을 질문 설명에 삽입)

```
### Quantitative Evidence (SHAP)
ML Fraud Score: 0.8799

FACTORS INCREASING FRAUD RISK:
  - card6 = 1.00 (SHAP: +0.643, increases fraud risk)
  - hour = 9.00 (SHAP: +0.640, increases fraud risk)
  - C6 = 3.00 (SHAP: +0.400, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - P_emaildomain = 2.00 (SHAP: -0.307, decreases fraud risk)
  - card1 = 15066.00 (SHAP: -0.249, decreases fraud risk)
  - C1 = 1.00 (SHAP: -0.244, decreases fraud risk)

Overall: 5 features increase risk, 5 features decr
```

**Q. 설명 C의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 종합 순위** (객관식 그리드, 필수)
  - 질문: "세 가지 설명 중 가장 유용한 것부터 순위를 매겨 주세요"
  - 1순위 (가장 유용): A / B / C
  - 2순위: A / B / C
  - 3순위: A / B / C

**Q. 추가 의견** (단답형, 선택)
  - "순위를 매긴 이유나 특정 설명에 대한 의견을 자유롭게 적어 주세요."

---

## [섹션 6] 거래 4 / 10

**거래 정보** (이 텍스트를 섹션 설명에 삽입)

> ML 사기 확률 점수: **0.9624**
>
> Transaction Amount: $22.75 (typical range)
> Time: late night/early morning (3:00), weekend
> Product: C
> Email domain: hotmail.com
> Unusual count features (>95th pct): {'C12': 2}
> Time deltas: 5 available

**설명 A** (아래 내용을 질문 설명에 삽입)

```
ML Fraud Score: 0.9624

FACTORS INCREASING FRAUD RISK:
  - C8 = 2.00 (SHAP: +0.646, increases fraud risk)
  - C13 = 1.00 (SHAP: +0.468, increases fraud risk)
  - card3 = 185.00 (SHAP: +0.334, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - TransactionAmt = 22.75 (SHAP: -0.307, decreases fraud risk)
  - P_emaildomain = 19.00 (SHAP: -0.185, decreases fraud risk)
  - card6 = 2.00 (SHAP: -0.
```

**Q. 설명 A의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**설명 B** (아래 내용을 질문 설명에 삽입)

```
The evidence for fraud is strong due to the high fraud score, late-night timing, and multiple anomalous features. However, the typical transaction amount and common email domain provide some counter-evidence.
Recommendation: HOLD_FOR_REVIEW
Fraud likelihood: 0.95
```

**Q. 설명 B의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**설명 C** (아래 내용을 질문 설명에 삽입)

```
### Quantitative Evidence (SHAP)
ML Fraud Score: 0.9624

FACTORS INCREASING FRAUD RISK:
  - C8 = 2.00 (SHAP: +0.646, increases fraud risk)
  - C13 = 1.00 (SHAP: +0.468, increases fraud risk)
  - card3 = 185.00 (SHAP: +0.334, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - TransactionAmt = 22.75 (SHAP: -0.307, decreases fraud risk)
  - P_emaildomain = 19.00 (SHAP: -0.185, decreases fraud risk)
  - card6 = 2.00 (SHAP: -0.167, decreases fraud risk)

Overall: 5 features increase risk, 5 fe
```

**Q. 설명 C의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 종합 순위** (객관식 그리드, 필수)
  - 질문: "세 가지 설명 중 가장 유용한 것부터 순위를 매겨 주세요"
  - 1순위 (가장 유용): A / B / C
  - 2순위: A / B / C
  - 3순위: A / B / C

**Q. 추가 의견** (단답형, 선택)
  - "순위를 매긴 이유나 특정 설명에 대한 의견을 자유롭게 적어 주세요."

---

## [섹션 7] 거래 5 / 10

**거래 정보** (이 텍스트를 섹션 설명에 삽입)

> ML 사기 확률 점수: **0.8674**
>
> Transaction Amount: $50.00 (typical range)
> Time: late night/early morning (22:00), weekday
> Product: H
> Email domain: hotmail.com
> Time deltas: 1 available

**설명 A** (아래 내용을 질문 설명에 삽입)

```
ML Fraud Score: 0.8674

FACTORS INCREASING FRAUD RISK:
  - C14 = 0.00 (SHAP: +1.143, increases fraud risk)
  - C13 = 0.00 (SHAP: +0.701, increases fraud risk)
  - card6 = 1.00 (SHAP: +0.385, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - C1 = 1.00 (SHAP: -0.442, decreases fraud risk)
  - addr1 = 469.00 (SHAP: -0.261, decreases fraud risk)
  - TransactionAmt = 50.00 (SHAP: -0.203, decrea
```

**Q. 설명 A의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**설명 B** (아래 내용을 질문 설명에 삽입)

```
The evidence for fraud is strengthened by the late-night timing and the round number transaction amount. However, the typical range of the transaction amount and the use of a common email domain provide some counter-evidence. Overall, the high fraud score indicates a significant likelihood of fraud.
Recommendation: HOLD_FOR_REVIEW
Fraud likelihood: 0.8674
```

**Q. 설명 B의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**설명 C** (아래 내용을 질문 설명에 삽입)

```
### Quantitative Evidence (SHAP)
ML Fraud Score: 0.8674

FACTORS INCREASING FRAUD RISK:
  - C14 = 0.00 (SHAP: +1.143, increases fraud risk)
  - C13 = 0.00 (SHAP: +0.701, increases fraud risk)
  - card6 = 1.00 (SHAP: +0.385, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - C1 = 1.00 (SHAP: -0.442, decreases fraud risk)
  - addr1 = 469.00 (SHAP: -0.261, decreases fraud risk)
  - TransactionAmt = 50.00 (SHAP: -0.203, decreases fraud risk)

Overall: 5 features increase risk, 5 features decr
```

**Q. 설명 C의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 종합 순위** (객관식 그리드, 필수)
  - 질문: "세 가지 설명 중 가장 유용한 것부터 순위를 매겨 주세요"
  - 1순위 (가장 유용): A / B / C
  - 2순위: A / B / C
  - 3순위: A / B / C

**Q. 추가 의견** (단답형, 선택)
  - "순위를 매긴 이유나 특정 설명에 대한 의견을 자유롭게 적어 주세요."

---

## [섹션 8] 거래 6 / 10

**거래 정보** (이 텍스트를 섹션 설명에 삽입)

> ML 사기 확률 점수: **0.8800**
>
> Transaction Amount: $914.00 (extremely large (>3 std above average)), round number
> Time: evening (18:00), weekday
> Product: W
> Email domain: comcast.net
> Time deltas: 8 available

**설명 A** (아래 내용을 질문 설명에 삽입)

```
ML Fraud Score: 0.8800

FACTORS INCREASING FRAUD RISK:
  - TransactionAmt = 914.00 (SHAP: +1.267, increases fraud risk)
  - D2 = 2.00 (SHAP: +0.552, increases fraud risk)
  - D3 = 2.00 (SHAP: +0.322, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - card6 = 2.00 (SHAP: -0.213, decreases fraud risk)
  - C1 = 1.00 (SHAP: -0.199, decreases fraud risk)
  - C11 = 1.00 (SHAP: -0.158, decreases f
```

**Q. 설명 A의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**설명 B** (아래 내용을 질문 설명에 삽입)

```
The evidence against fraud is relatively strong, indicating that this transaction may be legitimate. However, the high fraud score from the ML model suggests caution.
Recommendation: HOLD_FOR_REVIEW
Fraud likelihood: 0.8796
```

**Q. 설명 B의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**설명 C** (아래 내용을 질문 설명에 삽입)

```
### Quantitative Evidence (SHAP)
ML Fraud Score: 0.8800

FACTORS INCREASING FRAUD RISK:
  - TransactionAmt = 914.00 (SHAP: +1.267, increases fraud risk)
  - D2 = 2.00 (SHAP: +0.552, increases fraud risk)
  - D3 = 2.00 (SHAP: +0.322, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - card6 = 2.00 (SHAP: -0.213, decreases fraud risk)
  - C1 = 1.00 (SHAP: -0.199, decreases fraud risk)
  - C11 = 1.00 (SHAP: -0.158, decreases fraud risk)

Overall: 5 features increase risk, 5 features decrease 
```

**Q. 설명 C의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 종합 순위** (객관식 그리드, 필수)
  - 질문: "세 가지 설명 중 가장 유용한 것부터 순위를 매겨 주세요"
  - 1순위 (가장 유용): A / B / C
  - 2순위: A / B / C
  - 3순위: A / B / C

**Q. 추가 의견** (단답형, 선택)
  - "순위를 매긴 이유나 특정 설명에 대한 의견을 자유롭게 적어 주세요."

---

## [섹션 9] 거래 7 / 10

**거래 정보** (이 텍스트를 섹션 설명에 삽입)

> ML 사기 확률 점수: **0.8305**
>
> Transaction Amount: $53.05 (typical range)
> Time: late night/early morning (23:00), weekday
> Product: C
> Email domain: gmail.com
> Unusual count features (>95th pct): {'C4': 2, 'C7': 2, 'C12': 2}
> Time deltas: 5 available

**설명 A** (아래 내용을 질문 설명에 삽입)

```
ML Fraud Score: 0.8305

FACTORS INCREASING FRAUD RISK:
  - C13 = 1.00 (SHAP: +0.519, increases fraud risk)
  - card6 = 1.00 (SHAP: +0.485, increases fraud risk)
  - C4 = 2.00 (SHAP: +0.474, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - card1 = 16062.00 (SHAP: -0.435, decreases fraud risk)
  - addr1 = 130.00 (SHAP: -0.415, decreases fraud risk)
  - C1 = 2.00 (SHAP: -0.359, decreases fra
```

**Q. 설명 A의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**설명 B** (아래 내용을 질문 설명에 삽입)

```
The evidence suggests a moderate likelihood of fraud due to the combination of late-night timing and multiple anomalous features, despite the typical transaction amount and common email domain.
Recommendation: HOLD_FOR_REVIEW
Fraud likelihood: 0.83
```

**Q. 설명 B의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**설명 C** (아래 내용을 질문 설명에 삽입)

```
### Quantitative Evidence (SHAP)
ML Fraud Score: 0.8305

FACTORS INCREASING FRAUD RISK:
  - C13 = 1.00 (SHAP: +0.519, increases fraud risk)
  - card6 = 1.00 (SHAP: +0.485, increases fraud risk)
  - C4 = 2.00 (SHAP: +0.474, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - card1 = 16062.00 (SHAP: -0.435, decreases fraud risk)
  - addr1 = 130.00 (SHAP: -0.415, decreases fraud risk)
  - C1 = 2.00 (SHAP: -0.359, decreases fraud risk)

Overall: 5 features increase risk, 5 features decrease ri
```

**Q. 설명 C의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 종합 순위** (객관식 그리드, 필수)
  - 질문: "세 가지 설명 중 가장 유용한 것부터 순위를 매겨 주세요"
  - 1순위 (가장 유용): A / B / C
  - 2순위: A / B / C
  - 3순위: A / B / C

**Q. 추가 의견** (단답형, 선택)
  - "순위를 매긴 이유나 특정 설명에 대한 의견을 자유롭게 적어 주세요."

---

## [섹션 10] 거래 8 / 10

**거래 정보** (이 텍스트를 섹션 설명에 삽입)

> ML 사기 확률 점수: **0.9523**
>
> Transaction Amount: $56.55 (typical range)
> Time: afternoon (17:00), weekday
> Product: C
> Email domain: gmail.com
> Unusual count features (>95th pct): {'C1': 25, 'C2': 111, 'C4': 19, 'C6': 19, 'C7': 19, 'C8': 4, 'C10': 4, 'C11': 27, 'C12': 27}
> Time deltas: 7 available

**설명 A** (아래 내용을 질문 설명에 삽입)

```
ML Fraud Score: 0.9523

FACTORS INCREASING FRAUD RISK:
  - C1 = 25.00 (SHAP: +2.131, increases fraud risk)
  - C6 = 19.00 (SHAP: +0.397, increases fraud risk)
  - C12 = 27.00 (SHAP: +0.339, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - C14 = 9.00 (SHAP: -0.819, decreases fraud risk)
  - C13 = 17.00 (SHAP: -0.811, decreases fraud risk)
  - card1 = 15885.00 (SHAP: -0.497, decreases fraud
```

**Q. 설명 A의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**설명 B** (아래 내용을 질문 설명에 삽입)

```
The evidence for fraud is strong due to the high fraud score and weekend timing, but the typical transaction amount and morning timing provide some counter-evidence.
Recommendation: HOLD_FOR_REVIEW
Fraud likelihood: 0.85
```

**Q. 설명 B의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**설명 C** (아래 내용을 질문 설명에 삽입)

```
### Quantitative Evidence (SHAP)
ML Fraud Score: 0.9523

FACTORS INCREASING FRAUD RISK:
  - C1 = 25.00 (SHAP: +2.131, increases fraud risk)
  - C6 = 19.00 (SHAP: +0.397, increases fraud risk)
  - C12 = 27.00 (SHAP: +0.339, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - C14 = 9.00 (SHAP: -0.819, decreases fraud risk)
  - C13 = 17.00 (SHAP: -0.811, decreases fraud risk)
  - card1 = 15885.00 (SHAP: -0.497, decreases fraud risk)

Overall: 5 features increase risk, 5 features decrease risk
```

**Q. 설명 C의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 종합 순위** (객관식 그리드, 필수)
  - 질문: "세 가지 설명 중 가장 유용한 것부터 순위를 매겨 주세요"
  - 1순위 (가장 유용): A / B / C
  - 2순위: A / B / C
  - 3순위: A / B / C

**Q. 추가 의견** (단답형, 선택)
  - "순위를 매긴 이유나 특정 설명에 대한 의견을 자유롭게 적어 주세요."

---

## [섹션 11] 거래 9 / 10

**거래 정보** (이 텍스트를 섹션 설명에 삽입)

> ML 사기 확률 점수: **0.8508**
>
> Transaction Amount: $60.90 (typical range)
> Time: late night/early morning (2:00), weekday
> Product: C
> Email domain: hotmail.com
> Unusual count features (>95th pct): {'C4': 2, 'C7': 2, 'C12': 2}
> Time deltas: 4 available

**설명 A** (아래 내용을 질문 설명에 삽입)

```
ML Fraud Score: 0.8508

FACTORS INCREASING FRAUD RISK:
  - C4 = 2.00 (SHAP: +0.575, increases fraud risk)
  - card6 = 1.00 (SHAP: +0.518, increases fraud risk)
  - C13 = 1.00 (SHAP: +0.513, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - C1 = 2.00 (SHAP: -0.281, decreases fraud risk)
  - P_emaildomain = 19.00 (SHAP: -0.164, decreases fraud risk)
  - C8 = 1.00 (SHAP: -0.146, decreases fra
```

**Q. 설명 A의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**설명 B** (아래 내용을 질문 설명에 삽입)

```
The evidence for fraud is moderate, primarily based on the ML model's score and the round number of the transaction. However, the timing and typical amount provide some counterbalance, suggesting that while there are indicators of potential fraud, they are not overwhelmingly strong.
Recommendation: HOLD_FOR_REVIEW
Fraud likelihood: 0.8514
```

**Q. 설명 B의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**설명 C** (아래 내용을 질문 설명에 삽입)

```
### Quantitative Evidence (SHAP)
ML Fraud Score: 0.8508

FACTORS INCREASING FRAUD RISK:
  - C4 = 2.00 (SHAP: +0.575, increases fraud risk)
  - card6 = 1.00 (SHAP: +0.518, increases fraud risk)
  - C13 = 1.00 (SHAP: +0.513, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - C1 = 2.00 (SHAP: -0.281, decreases fraud risk)
  - P_emaildomain = 19.00 (SHAP: -0.164, decreases fraud risk)
  - C8 = 1.00 (SHAP: -0.146, decreases fraud risk)

Overall: 5 features increase risk, 5 features decrease ri
```

**Q. 설명 C의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 종합 순위** (객관식 그리드, 필수)
  - 질문: "세 가지 설명 중 가장 유용한 것부터 순위를 매겨 주세요"
  - 1순위 (가장 유용): A / B / C
  - 2순위: A / B / C
  - 3순위: A / B / C

**Q. 추가 의견** (단답형, 선택)
  - "순위를 매긴 이유나 특정 설명에 대한 의견을 자유롭게 적어 주세요."

---

## [섹션 12] 거래 10 / 10

**거래 정보** (이 텍스트를 섹션 설명에 삽입)

> ML 사기 확률 점수: **0.8899**
>
> Transaction Amount: $300.00 (typical range), round number
> Time: evening (18:00), weekend
> Product: H
> Email domain: anonymous.com
> Time deltas: 1 available

**설명 A** (아래 내용을 질문 설명에 삽입)

```
ML Fraud Score: 0.8899

FACTORS INCREASING FRAUD RISK:
  - C14 = 0.00 (SHAP: +1.038, increases fraud risk)
  - TransactionAmt = 300.00 (SHAP: +0.613, increases fraud risk)
  - C13 = 0.00 (SHAP: +0.607, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - P_emaildomain = 1.00 (SHAP: -0.590, decreases fraud risk)
  - card1 = 16659.00 (SHAP: -0.422, decreases fraud risk)
  - C1 = 1.00 (SHAP: -0.
```

**Q. 설명 A의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 A의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**설명 B** (아래 내용을 질문 설명에 삽입)

```
The evidence for fraud is strengthened by the late-night timing and the round number amount, but the typical amount and established email domain provide some counter-evidence. However, the overall indicators lean towards fraud.
Recommendation: HOLD_FOR_REVIEW
Fraud likelihood: 0.8897
```

**Q. 설명 B의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 B의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**설명 C** (아래 내용을 질문 설명에 삽입)

```
### Quantitative Evidence (SHAP)
ML Fraud Score: 0.8899

FACTORS INCREASING FRAUD RISK:
  - C14 = 0.00 (SHAP: +1.038, increases fraud risk)
  - TransactionAmt = 300.00 (SHAP: +0.613, increases fraud risk)
  - C13 = 0.00 (SHAP: +0.607, increases fraud risk)

FACTORS DECREASING FRAUD RISK:
  - P_emaildomain = 1.00 (SHAP: -0.590, decreases fraud risk)
  - card1 = 16659.00 (SHAP: -0.422, decreases fraud risk)
  - C1 = 1.00 (SHAP: -0.352, decreases fraud risk)

Overall: 5 features increase risk, 5 fe
```

**Q. 설명 C의 유용성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 거래 조사에 얼마나 도움이 됩니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 명확성** (선형 척도 1-5, 필수)
  - 질문: "이 설명이 이해하기 쉽습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 구체성** (선형 척도 1-5, 필수)
  - 질문: "일반적인 내용이 아닌 구체적 근거를 제시합니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 설명 C의 신뢰성** (선형 척도 1-5, 필수)
  - 질문: "이 설명을 신뢰할 수 있습니까?"
  - 1 = 매우 나쁨 ... 5 = 매우 좋음

**Q. 종합 순위** (객관식 그리드, 필수)
  - 질문: "세 가지 설명 중 가장 유용한 것부터 순위를 매겨 주세요"
  - 1순위 (가장 유용): A / B / C
  - 2순위: A / B / C
  - 3순위: A / B / C

**Q. 추가 의견** (단답형, 선택)
  - "순위를 매긴 이유나 특정 설명에 대한 의견을 자유롭게 적어 주세요."

---

## [마지막 섹션] 종합 의견

**Q. 전반적으로 어떤 유형의 설명이 사기 조사에 가장 유용합니까?** (객관식)
- A 유형 (수치적 feature 기여도)
- B 유형 (AI 서술형 분석)
- C 유형 (수치 + 서술 결합)
- 모두 비슷하다

**Q. AI 사기 설명 시스템에 대한 추가 의견** (장문형, 선택)
  - "실무에서 이런 AI 설명 시스템이 도움이 될지, 어떤 정보가 추가되면 좋을지 의견을 주세요."

---

## 설정 참고사항

1. **섹션 나누기**: 각 거래를 별도 섹션으로 (한 페이지에 1개 거래)
2. **필수 응답**: 1-5점 평가와 순위는 필수, 자유 의견은 선택
3. **선형 척도**: 1-5점 질문은 Google Forms의 '선형 척도' 유형 사용
4. **설명 삽입**: 각 설명 텍스트는 질문의 '설명' 필드에 넣거나 이미지 캡처 삽입
5. **순서 랜덤화**: 평가자별로 A/B/C가 가리키는 실제 방법을 바꾸려면 양식 2-3개 버전 생성
6. **응답 제한**: '1인 1회 응답'으로 설정
7. **설명 A = SHAP feature attribution, B = LLM narrative, C = SHAP+LLM combined** (평가자에게는 알리지 마세요)