"""Download LAFAN1 g1 clips from ModelScope (unitreerobotics official mirror)."""
import os

for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)

from modelscope.hub.snapshot_download import dataset_snapshot_download

dest = "data/raw/lafan1_ms"
path = dataset_snapshot_download(
    dataset_id="unitreerobotics/LAFAN1_Retargeting_Dataset",
    local_dir=dest,
    allow_patterns=["g1/*", "*.md"],
)
print("DONE", path)
