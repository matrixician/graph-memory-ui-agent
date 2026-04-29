import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json

class VisionEngine:
    def __init__(self):
        print("[System] Allocating Qwen2.5-VL-7B to GPU (4-bit)...")
        # We load in 4-bit to fit safely inside Colab's 16GB T4 GPU
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True 
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        print("[System] Vision Matrix Online.")

    def analyze_screen(self, image_path, goal, context=""):
        print("[Vision] Analyzing local screen state...")
        
        # We enforce strict JSON output in the prompt since we aren't using an API flag
        system_instruction = (
            "You are an autonomous UI agent. Look at the provided screenshot with numbered bounding boxes. "
            f"Your Goal: {goal}. Context: {context}. "
            "Identify the correct UI element to interact with. "
            "You MUST respond ONLY with a raw JSON object, no markdown, no other text: "
            '{"action_type": "click", "target_box_id": 42, "reasoning": "brief explanation"}'
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": system_instruction},
                ],
            }
        ]

        # Prepare inputs for Qwen
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # Generate output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Clean the string to ensure it's valid JSON
        cleaned_output = output_text.strip().replace('```json', '').replace('```', '')
        
        try:
            result = json.loads(cleaned_output)
            return result
        except json.JSONDecodeError:
            print(f"[Error] Failed to parse VLM output as JSON: {output_text}")
            # Fallback safe response
            return {"action_type": "error", "target_box_id": 0, "reasoning": "Parse failure"}

    def get_screen_description(self, image_path):
        """A quick pass to generate a description of the screen for the Graph Node"""
        # For prototype speed, we still mock this, but you could run another prompt through Qwen here.
        return "Software Interface Screen"