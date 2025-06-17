import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_path, base_model_name="microsoft/Phi-3-mini-4k-instruct"):
    """Tải mô hình và tokenizer."""
    logger.info("📥 Đang tải tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Đặt pad_token nếu chưa có
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("🔧 Đã đặt pad_token bằng eos_token")

    logger.info("📥 Đang tải mô hình gốc...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,  # Sử dụng float32 cho CPU
        device_map="cpu",  # Ép chạy trên CPU
        attn_implementation="eager"
    )

    logger.info("📥 Đang tải mô hình LoRA đã tinh chỉnh...")
    try:
        model = PeftModel.from_pretrained(base_model, model_path)
    except Exception as e:
        logger.error(f"❌ Lỗi khi tải mô hình LoRA: {e}")
        raise

    logger.info("✅ Mô hình và tokenizer đã được tải thành công")
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_length=200):
    """Tạo phản hồi từ mô hình."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def main():
    """Chạy giao diện chat."""
    model_path = "./songoku-ultimate-results/final-model"
    try:
        model, tokenizer = load_model_and_tokenizer(model_path)
    except Exception as e:
        logger.error(f"❌ Không thể tải mô hình: {e}")
        print("❌ Lỗi khi tải mô hình. Kiểm tra đường dẫn hoặc cấu hình.")
        return

    print("\n🎉 Chào mừng bạn đến với Son Goku AI! 🎉")
    print("Hãy nói chuyện với Goku! Nhập 'thoát' để kết thúc.\n")

    while True:
        user_input = input("👤 Bạn: ")
        if user_input.lower() == "thoát":
            print("👋 Tạm biệt! Hẹn gặp lại, chiến binh!")
            break

        # Tạo prompt theo phong cách Son Goku
        prompt = f"[INST] Bạn là Son Goku, một chiến binh Saiyan mạnh mẽ và nhiệt huyết. Hãy trả lời câu hỏi này với phong cách của Goku, sử dụng giọng điệu hào sảng, thân thiện và sẵn sàng chiến đấu! Câu hỏi: {user_input} [/INST]\n"
        logger.info(f"📩 Nhận được câu hỏi: {user_input}")

        try:
            response = generate_response(model, tokenizer, prompt)
            print(f"🤖 Goku: {response.replace(prompt, '').strip()}\n")
        except Exception as e:
            logger.error(f"❌ Lỗi khi tạo phản hồi: {e}")
            print("❌ Có lỗi xảy ra, vui lòng thử lại!\n")


if __name__ == "__main__":
    main()