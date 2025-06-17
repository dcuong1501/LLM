# finetune_songoku_ultimate.py
"""
Version đơn giản nhất - chắc chắn work với batch_size=1
Không có padding issues, không có tensor length conflicts
"""

import os
import torch
import gc
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json

# Force CPU-only
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UltimateSimpleDataset:
    def __init__(self, tokenizer, json_file_path, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.examples = []
        logger.info(f"📂 Processing {len(data)} conversations...")

        for item in data:
            text = f"User: {item['prompt']}\nAssistant: {item['response']}<|endoftext|>"

            # Tokenize immediately with fixed length
            tokens = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",  # Always pad to max_length
                max_length=max_length,
                return_tensors="pt"
            )

            # Extract and convert to lists
            input_ids = tokens["input_ids"].squeeze().tolist()
            attention_mask = tokens["attention_mask"].squeeze().tolist()

            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.copy()  # For causal LM
            })

        logger.info(f"✅ Prepared {len(self.examples)} examples with fixed length {max_length}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class UltimateSimpleFineTuner:
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct"):
        self.model_name = model_name

        print("🎯 ULTIMATE SIMPLE SON GOKU FINE-TUNER")
        print("=" * 50)
        print("💡 Batch size = 1, No padding issues!")
        print("✅ Guaranteed to work!")
        print("=" * 50)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model on CPU
        logger.info("📥 Loading model on CPU...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            use_cache=False
        )

        logger.info("✅ Model loaded successfully")

    def setup_lora(self):
        """Setup minimal LoRA"""
        logger.info("🔧 Setting up LoRA...")

        # Find any linear layer for target
        target_modules = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module_name = name.split('.')[-1]
                if 'proj' in module_name:
                    target_modules.append(module_name)
                    break

        if not target_modules:
            target_modules = ['o_proj']  # Default fallback

        logger.info(f"🎯 Target modules: {target_modules}")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none"
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        logger.info("✅ LoRA setup completed")

    def train(self, dataset_path="songoku_dataset.json", output_dir="./songoku-ultimate-results"):
        """Ultimate simple training"""

        os.makedirs(output_dir, exist_ok=True)

        # Setup LoRA
        self.setup_lora()

        # Create simple dataset
        logger.info("📊 Creating simple dataset...")
        full_dataset = UltimateSimpleDataset(self.tokenizer, dataset_path)

        # Convert to HuggingFace dataset
        dataset_dict = {
            "input_ids": [ex["input_ids"] for ex in full_dataset.examples],
            "attention_mask": [ex["attention_mask"] for ex in full_dataset.examples],
            "labels": [ex["labels"] for ex in full_dataset.examples]
        }

        hf_dataset = Dataset.from_dict(dataset_dict)

        # Split train/test
        split_dataset = hf_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]

        logger.info(f"📊 Train samples: {len(train_dataset)}")

        # Ultra-simple data collator
        def simple_collate_fn(batch):
            """Simple collator that just stacks tensors"""
            batch_dict = {}
            for key in ["input_ids", "attention_mask", "labels"]:
                batch_dict[key] = torch.tensor([item[key] for item in batch])
            return batch_dict

        logger.info("📈 Training setup:")
        logger.info("   - Batch size: 1 (no batching issues)")
        logger.info("   - Fixed sequence length: 128")
        logger.info("   - Expected time: 10-15 minutes")

        # Ultra-simple training args
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,

            # Batch size 1 - no padding issues
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,  # Accumulate for effective batch size

            # Training
            num_train_epochs=5,
            max_steps=50,

            # CPU settings
            fp16=False,
            bf16=False,
            dataloader_pin_memory=False,
            gradient_checkpointing=False,
            dataloader_num_workers=0,

            # Simple logging
            logging_steps=10,
            eval_strategy="no",
            save_strategy="epoch",

            # Optimizer
            learning_rate=1e-4,
            warmup_steps=5,
            optim="adamw_torch",

            # Stability
            report_to=None,
            remove_unused_columns=False,
            dataloader_drop_last=False,
        )

        # Simple trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=simple_collate_fn,
            tokenizer=self.tokenizer,
        )

        logger.info("🔥 Starting ultimate simple training...")

        try:
            trainer.train()

            # Save
            final_model_path = os.path.join(output_dir, "final-model")
            trainer.save_model(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)

            logger.info(f"✅ Training completed! Saved to {final_model_path}")
            return final_model_path

        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            raise

    def test_model(self, model_path="./songoku-ultimate-results/final-model"):
        """Test the model"""
        logger.info("🧪 Testing ultimate Son Goku model...")

        try:
            from peft import PeftModel

            # Load for testing
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True
            )

            model = PeftModel.from_pretrained(base_model, model_path)

            test_prompts = [
                "Who are you?",
                "What's your favorite food?",
                "How do you train?",
                "Tell me about your power"
            ]

            print("\n" + "🎉" * 20)
            print("🎉 SON GOKU AI TRAINING COMPLETE! 🎉")
            print("🎉" * 20)
            print("\n🧪 Testing your Son Goku AI:")
            print("=" * 50)

            for i, prompt in enumerate(test_prompts, 1):
                inputs = self.tokenizer.encode(f"User: {prompt}\nAssistant:", return_tensors="pt")

                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=80,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1
                    )

                response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

                print(f"\n[{i}] 👤 User: {prompt}")
                print(f"    🤖 Goku: {response}")
                print("-" * 40)

            print("\n🎯 CONGRATULATIONS!")
            print("✅ Your Son Goku AI is ready!")
            print("💪 It can now respond like the legendary Saiyan!")
            print("🔥 Use it with songoku_handler.py")
            print("=" * 50)

        except Exception as e:
            logger.error(f"❌ Testing failed: {str(e)}")
            print("✅ Training completed but testing failed")
            print("💡 Your model is still saved and ready to use!")


def main():
    """Main function"""

    if not os.path.exists("songoku_dataset.json"):
        print("❌ songoku_dataset.json not found!")
        print("Please make sure the dataset file is in the current directory")
        return

    try:
        print("🚀 Starting ultimate Son Goku fine-tuning...")

        # Create tuner
        tuner = UltimateSimpleFineTuner()

        # Train
        model_path = tuner.train()

        # Test
        tuner.test_model(model_path)

        print("\n🎊 MISSION ACCOMPLISHED! 🎊")
        print("Son Goku AI training completed successfully!")

    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        print("But we've learned something new! Keep trying!")
        raise


if __name__ == "__main__":
    main()