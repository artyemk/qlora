from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path="/qlora/output",
    repo_id="Kernel/qlora_guanaco",
    repo_type="model",
)
