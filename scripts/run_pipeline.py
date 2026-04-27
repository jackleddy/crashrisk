from pathlib import Path
import subprocess
import sys


def _run_step(name: str, script_rel_path: str, repo_root: Path) -> None:
    print(f"\n=== {name} ===")
    script_path = repo_root / script_rel_path
    subprocess.run([sys.executable, str(script_path)], check=True, cwd=repo_root)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _run_step("Build network", "scripts/build_network.py", repo_root)
    _run_step("Download data", "scripts/download_data.py", repo_root)
    _run_step("Align data", "scripts/align_data.py", repo_root)
    _run_step("Build training tables", "scripts/training_tables.py", repo_root)
    _run_step("Train GNN", "scripts/train_gnn.py", repo_root)
    _run_step("Render risk map", "scripts/render_map.py", repo_root)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
