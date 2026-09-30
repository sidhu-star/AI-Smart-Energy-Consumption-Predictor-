from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_requirements_file_exists():
    assert (REPO_ROOT / "requirements.txt").exists()


def test_core_project_files_exist():
    assert (REPO_ROOT / "App").exists()
    assert (REPO_ROOT / "Config").exists()
    assert (REPO_ROOT / "Predictor").exists()
    assert (REPO_ROOT / "Utils").exists()
