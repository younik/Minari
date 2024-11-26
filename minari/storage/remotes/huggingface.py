import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple


try:
    from huggingface_hub.hf_api import HfApi, RepoFile
    from huggingface_hub.utils import EntryNotFoundError
except ImportError:
    raise ImportError(
        'huggingface_hub is not installed. Please install it using `pip install "minari[hf]"`'
    )

from minari.dataset.minari_storage import METADATA_FILE_NAME
from minari.storage.datasets_root_dir import get_dataset_path
from minari.storage.remotes.cloud_storage import CloudStorage


_NAMESPACE_METADATA_FILENAME = "namespace_metadata.json"


class HuggingFaceStorage(CloudStorage):

    def __init__(self, name: str, token: Optional[str] = None) -> None:
        self.name = name
        self._api = HfApi(token=token)

    def _decompose_path(self, path: str) -> Tuple[str, str]:
        root, *rem = path.split("/")
        return root, "/".join(rem)

    def upload_dataset(self, dataset_id: str) -> None:
        path = get_dataset_path(dataset_id)
        repo_name, path_in_repo = self._decompose_path(dataset_id)
        repo_id = f"{self.name}/{repo_name}"

        self._api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        self._api.upload_folder(
            repo_id=repo_id,
            folder_path=path,
            path_in_repo=path_in_repo,
            repo_type="dataset",
        )

    def upload_namespace(self, namespace: str) -> None:
        local_filepath = get_dataset_path(namespace) / _NAMESPACE_METADATA_FILENAME
        repo_name, path_in_repo = self._decompose_path(namespace)
        repo_id = f"{self.name}/{repo_name}"

        self._api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        self._api.upload_file(
            path_or_fileobj=local_filepath,
            path_in_repo=os.path.join(path_in_repo, _NAMESPACE_METADATA_FILENAME),
            repo_id=repo_id,
            repo_type="dataset",
        )

    def list_datasets(self, prefix: Optional[str] = None) -> Iterable[str]:
        group_name, in_repo_prefix = None, None
        if prefix is not None:
            group_name, in_repo_prefix = self._decompose_path(prefix)

        metadata_end = f"/data/{METADATA_FILE_NAME}"
        hf_datasets = self._api.list_datasets(author=self.name, dataset_name=group_name)
        for group_info in hf_datasets:
            repo_name = group_info.id.split("/", 1)[1]
            tree = self._api.list_repo_tree(
                group_info.id,
                path_in_repo=in_repo_prefix,
                repo_type="dataset",
                recursive=True,
            )
            try:
                for entry in tree:
                    if isinstance(entry, RepoFile) and entry.path.endswith(
                        metadata_end
                    ):
                        yield f"{repo_name}/{entry.path[:-len(metadata_end)]}"
            except EntryNotFoundError:
                yield from []

    def download_dataset(self, dataset_id: Any, path: Path) -> None:
        repo_id, path_in_repo = self._decompose_path(dataset_id)
        self._api.snapshot_download(
            repo_id=f"{self.name}/{repo_id}",
            allow_patterns=os.path.join(path_in_repo, "*"),
            repo_type="dataset",
            local_dir=path.joinpath(repo_id),
        )

    def get_dataset_metadata(self, dataset_id: str) -> dict:
        repo_id, path_in_repo = self._decompose_path(dataset_id)
        dataset_metadata = self._api.hf_hub_download(
            repo_id=f"{self.name}/{repo_id}",
            filename=os.path.join(path_in_repo, "data", METADATA_FILE_NAME),
            repo_type="dataset",
        )
        with open(dataset_metadata) as f:
            metadata = json.load(f)
        return metadata

    def list_namespaces(self) -> Iterable[str]:
        metadata_end = f"/{_NAMESPACE_METADATA_FILENAME}"
        for hf_dataset in self._api.list_datasets(author=self.name):
            repo_name = hf_dataset.id.split("/", 1)[1]
            tree = self._api.list_repo_tree(
                hf_dataset.id,
                repo_type="dataset",
                recursive=True,
            )
            try:
                for entry in tree:
                    if isinstance(entry, RepoFile) and entry.path.endswith(
                        metadata_end
                    ):
                        yield f"{repo_name}/{entry.path[:-len(metadata_end)]}"
            except EntryNotFoundError:
                yield from []

    def download_namespace_metadata(self, namespace: str, path: Path) -> None:
        repo_id, path_in_repo = self._decompose_path(namespace)
        self._api.hf_hub_download(
            repo_id=f"{self.name}/{repo_id}",
            filename=os.path.join(path_in_repo, _NAMESPACE_METADATA_FILENAME),
            repo_type="dataset",
            local_dir=path.joinpath(repo_id),
            force_download=True,
        )
