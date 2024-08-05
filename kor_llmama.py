from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# Hugging Face 토큰
token = "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"

# 예시: 한국어 텍스트 데이터셋 로드 (KLUE의 MRC 데이터셋)
dataset = load_dataset("klue", "mrc")

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token=token)

# EOS 토큰을 패딩 토큰으로 설정
tokenizer.pad_token = tokenizer.eos_token

# 전처리 함수 정의
def preprocess_function(examples):
    inputs = [context + " " + question for context, question in zip(examples['context'], examples['question'])]
    return tokenizer(inputs, truncation=True, padding="max_length")

# 데이터 전처리 적용
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 모델 로드
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token=token)

# 훈련 설정 정의
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer 객체 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# 모델 훈련
trainer.train()

# 모델 저장
model.save_pretrained("path_to_save_model")
tokenizer.save_pretrained("path_to_save_tokenizer")
