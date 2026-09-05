from __future__ import annotations

from opencollab.adapters._workspace_baseline import (
    baseline_from_paths,
    changed_entries,
    entry_for_path,
    is_control_plane,
)


def test_workspace_baseline_keeps_hidden_relative_names(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (root / ".opencollab").mkdir()
    (root / ".opencollab" / "config").write_text("hidden\n", encoding="utf-8")

    entry = entry_for_path(str(root), "./.env")

    assert entry.path == ".env"
    assert is_control_plane("./.opencollab")
    assert is_control_plane(".opencollab/config")
    assert not is_control_plane("./.env")


def test_workspace_baseline_ignores_control_plane_paths_without_stripping_dots(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (root / ".opencollab").mkdir()
    (root / ".opencollab" / "config").write_text("hidden\n", encoding="utf-8")

    baseline = baseline_from_paths(str(root), ["./.env", "./.opencollab/config"])
    entries = changed_entries(str(root), ["./.env", "./.opencollab/config"], baseline)

    assert [entry.path for entry in baseline.entries] == [".env"]
    assert entries == ()
