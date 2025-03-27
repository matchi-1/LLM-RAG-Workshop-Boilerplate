from huggingface_hub import InferenceClient

hf_token = "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"
model_name = "mistralai/Mistral-Nemo-Instruct-2407"  # Ensure this is public or you have permission

client = InferenceClient(api_key=hf_token)
response = client.text_generation("Test input", max_new_tokens=50)

print(response)
