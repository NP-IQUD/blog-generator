from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import torch

# Force CPU usage
device = torch.device("cpu")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="blog_dataset.txt",
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="./tinyllama-blog-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=1,  # Start with 1 epoch due to CPU
    per_device_train_batch_size=1,
    save_steps=100,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()

model.save_pretrained("./tinyllama-blog-finetuned")
tokenizer.save_pretrained("./tinyllama-blog-finetuned")
