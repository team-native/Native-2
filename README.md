# 안전 송금 금융사기 탐지 AI

AI와 디지털 금융 환경에 익숙하지 않은 고연령층 사용자가 송금 전에 금융사기 가능성을 확인할 수 있도록 돕는 Python 프로젝트입니다. 핵심은 규칙 기반 점수화가 아니라 검증·익명화된 데이터로 재학습되는 머신러닝 파이프라인입니다.

## AI 아키텍처

```text
Text → TF-IDF + Logistic Regression → text_fraud_probability
Transaction / Behavior → feature engineering ───────────────┐
                                                          ↓
                    Main Fraud Model (XGBoost) → fraud_probability (0~1)
                                                          ↓
                  향후 Risk Engine (미구현) → 향후 LLM 설명 (미구현)
```

Text Model은 문장 전체 n-gram 패턴을 학습하며 단순 키워드 일치로 최종 판단하지 않습니다. Main Model은 텍스트 확률과 거래·행동 feature를 함께 학습합니다. 향후 Transformer Text Model로 교체할 수 있도록 `TextPredictor` 인터페이스를 분리했습니다.

## Dataset schema와 Feature

필수 컬럼: `case_id`, `event_time`, `text`, `amount`, `average_amount`, `recipient_is_new`, `recipient_transfer_count`, `transfers_last_hour`, `average_transfers_per_hour`, `transfer_hour`, `label`, `label_source`.

`label`은 정상 `0` 또는 확인된 사기 `1`이며, `label_source`는 `verified_case`, `public_dataset`, `manual_review`, `synthetic_demo` 등 출처를 기록합니다. 모델 예측을 자동 label로 사용하지 않으며, 승인되지 않은 사용자 입력을 즉시 학습 데이터로 넣지 않습니다.

생성 feature: `text_fraud_probability`, `amount_ratio`, `average_amount_missing`, `recipient_is_new`, `recipient_transfer_count`, `has_transaction_history`, `transfer_frequency_ratio`, `has_behavior_history`, `transfer_hour`, `is_late_night`.

평균 이력이 0이거나 없을 때 비율을 임의의 위험값으로 만들지 않고, 0과 missing/history flag를 함께 전달합니다. 신규 수취인·심야 시간도 점수가 아니라 모델 학습 feature일 뿐입니다.

## 설치 및 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m training.pipeline
python3 -m uvicorn app.main:app --reload
```

Swagger: <http://127.0.0.1:8000/docs>

`data/raw/`에 승인된 CSV가 있으면 pipeline이 이를 사용하고, 비어 있으면 `data/sample/sample_training.csv`로 동작을 검증합니다. raw/validated/processed/rejected와 model binary는 Git에서 제외됩니다.

### 스미싱 JSON/JSONL 학습

`text`, `send_hour`, `has_url`, `has_phone_num`, `char_len`, `category`, `label` 형식의 스미싱 데이터는 거래 모델과 섞지 않고 별도 경량 분류기로 학습합니다.

```bash
python3 -m training.smishing_pipeline data/raw/smishing_messages.jsonl
```

외부에서 미리 나눈 데이터는 순서를 보존해 학습합니다. `test`는 최종 평가에만 사용합니다.

```bash
python3 -m training.smishing_pipeline --train data/raw/train.jsonl --validation data/raw/validation.jsonl --test data/raw/test.jsonl
python3 -m training.model_registry promote-smishing <version>
```

이 경로는 한국어 문자 문맥(char n-gram), URL·전화번호 존재 여부, 발송 시간, 분류 category를 사용합니다. 텍스트의 전화번호 등 식별자는 학습 전 마스킹됩니다. 생성 결과는 candidate일 뿐이며, 실제 송금 사기 production 모델로 자동 승격되지 않습니다.

## 자동 학습 Pipeline 및 평가

`python3 -m training.pipeline`은 raw 읽기 → schema validation → rejected 분리 → 중복 제거 → 익명화 → feature engineering → case_id group 기반 train/validation/test split → Text Model 학습 → text 확률 생성 → Main Model 학습 → 평가 → candidate/metadata/report 저장을 수행합니다.

동일 `case_id`가 train/test에 함께 포함되지 않도록 `StratifiedGroupKFold`를 사용합니다. `event_time`을 보존해 향후 time-based split으로 확장할 수 있습니다. 평가는 Accuracy, Precision, **Recall**, F1, ROC-AUC, PR-AUC, TP/TN/FP/FN을 JSON과 confusion matrix 이미지로 저장합니다. 금융사기 탐지에서는 사기를 정상으로 놓치는 False Negative가 중요하므로 Recall이 핵심 지표입니다. threshold는 `configs/training.toml`에서 모델 파라미터와 분리해 관리합니다.

## Model Registry와 Inference

학습은 `models/candidates/`에만 모델·metadata를 생성합니다. production이 있으면 Recall, Precision, F1, PR-AUC를 비교해 추천만 기록하며 자동 배포하지 않습니다. 사람이 검토한 뒤에만 승격합니다.

```bash
python3 -m training.model_registry promote <version>
```

승격 후 `/analyze`는 `models/production/`의 모델을 이용합니다. 운영 모델이 없으면 `503 model_not_ready`를 반환합니다.

스미싱 production 모델은 거래 사기 모델과 분리되어 `/analyze/smishing`에서 사용합니다. 요청 본문은 `text`, `send_hour`, 선택 `category`이며 URL·전화번호·긴급성·문자 길이는 원문에서 서버가 계산합니다.

### 공주 LLM 챗봇

`/chat/smishing`은 스미싱 확률을 먼저 로컬에서 계산하고, OpenAI `gpt-5-mini`가 그 판정과 안전 조치만 쉬운 한국어로 설명합니다. 원문 메시지와 개인정보는 LLM에 보내지 않습니다. API 키는 Git에 넣지 말고 실행 환경에만 설정합니다.

```bash
export OPENAI_API_KEY="your_api_key"
python3 -m pip install -r requirements.txt
python3 -m uvicorn app.main:app --reload
```

```json
{
  "text": "검찰입니다. 지금 바로 안전계좌로 돈을 보내주세요.",
  "amount": 3000000,
  "average_amount": 300000,
  "recipient_is_new": true,
  "recipient_transfer_count": 0,
  "transfers_last_hour": 4,
  "average_transfers_per_hour": 0.4,
  "transfer_hour": 14
}
```

## 개인정보·Data Poisoning 방지

학습 전 전화번호, 계좌번호 형태, 주민등록번호, 이메일, 긴 숫자 문자열을 마스킹합니다. 요청 원문과 계좌번호·인증번호·비밀번호를 로그로 출력하거나 저장하지 않습니다. 데이터는 검증 후보 → 개인정보 제거 → 사실·label 확인 → validated → 재학습 절차를 따라야 합니다. 실제 서비스 전에는 데이터 최소화, 접근통제, 보존정책, 라이선스·출처·검증 상태 관리와 법률 검토가 필요합니다.

## Sample 데이터의 한계와 향후 계획

sample CSV는 **synthetic demo**이며 pipeline·학습·추론 검증용입니다. 여기서 나온 성능 수치는 실제 금융사기 탐지 성능을 의미하지 않습니다. 실제 데이터 확보 후에는 label 품질 검토, 누수·중복 검사 강화, 시간 기반 검증, 불균형·threshold 재조정, 편향·드리프트 모니터링을 수행해야 합니다.

Risk Engine의 0~100 점수화와 XGBoost feature importance·SHAP 설명 가능 AI는 향후 확장 예정입니다. 공주 LLM은 판정을 내리지 않고, 로컬 모델의 결과를 고연령층 친화적인 설명·대응 방법으로만 바꿉니다.
