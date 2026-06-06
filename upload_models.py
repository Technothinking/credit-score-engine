from huggingface_hub import HfApi
import os

USERNAME = "Tasty-kadai-paneer"
SPACE_NAME = "credit-score-engine"

api = HfApi()

for f in os.listdir("models"):
    file_path = os.path.join("models", f)

    if os.path.isfile(file_path):
        print(f"Uploading {f}...")

        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"models/{f}",
            repo_id=f"{USERNAME}/{SPACE_NAME}",
            repo_type="space"
        )

print("All models uploaded successfully!")