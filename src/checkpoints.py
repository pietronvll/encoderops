from pathlib import Path

from huggingface_hub import hf_hub_download


def resolve_checkpoint_path(
    *,
    source: str = "hf",
    repo_id: str | None = None,
    filename: str | None = None,
    repo_type: str = "model",
    local_path: str | Path | None = None,
) -> str:
    if source == "hf":
        if repo_id is None or filename is None:
            raise ValueError("repo_id and filename are required when source='hf'.")
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
        )

    if source == "local":
        if local_path is None:
            raise ValueError("local_path is required when source='local'.")
        checkpoint_path = Path(local_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        return str(checkpoint_path)

    raise ValueError(f"Unsupported checkpoint source: {source}")
