#!/usr/bin/env python3
"""Brainlife-oriented FitLins runner with a human-readable model interface."""

from __future__ import annotations

import csv
import json
import re
import shlex
import shutil
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Any

ENTITY_ALIASES = {
    "subject": "sub",
    "session": "ses",
    "task": "task",
    "run": "run",
    "space": "space",
}
MOTION_BASE = ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z")
MOTION_24_SUFFIXES = ("", "_derivative1", "_power2", "_derivative1_power2")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _safe_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def _to_path(value: str | None, field_name: str, required: bool = True) -> Path | None:
    if not value:
        if required:
            raise ValueError(f"Missing required config field: {field_name}")
        return None
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    return path


def _get_nested(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _set_nested_default(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    if _get_nested(config, path) is not None:
        return
    current = config
    for key in path[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[path[-1]] = value


def apply_brainlife_flat_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Map flat Brainlife UI keys into nested app config blocks.

    Brainlife app config schema is typically flat, but this app uses nested
    blocks (`design`, `confounds`, `fitlins`, `advanced`).
    """
    mappings: dict[str, tuple[str, ...]] = {
        "design_name": ("design", "name"),
        "design_description": ("design", "description"),
        "condition_column": ("design", "condition_column"),
        "hrf_model": ("design", "hrf_model"),
        "auto_dummy_contrasts": ("design", "auto_dummy_contrasts"),
        "auto_task_vs_baseline": ("design", "auto_task_vs_baseline"),
        "confounds_strategy": ("confounds", "strategy"),
        "include_framewise_displacement": ("confounds", "include_framewise_displacement"),
        "include_dvars": ("confounds", "include_dvars"),
        "include_cosine_drift_terms": ("confounds", "include_cosine_drift_terms"),
        "confounds_strict": ("confounds", "strict"),
        "fitlins_space": ("fitlins", "space"),
        "fitlins_desc_label": ("fitlins", "desc_label"),
        "fitlins_estimator": ("fitlins", "estimator"),
        "fitlins_drift_model": ("fitlins", "drift_model"),
        "fitlins_n_cpus": ("fitlins", "n_cpus"),
        "fitlins_mem_gb": ("fitlins", "mem_gb"),
        "fitlins_drop_missing": ("fitlins", "drop_missing"),
        "dry_run": ("advanced", "dry_run"),
        "model_json_path": ("advanced", "model_json_path"),
    }
    for key, nested_path in mappings.items():
        if key in config and config[key] is not None:
            _set_nested_default(config, nested_path, config[key])
    return config


def infer_entities_from_name(path: Path) -> dict[str, str]:
    entities: dict[str, str] = {}
    name = path.name
    for entity, key in ENTITY_ALIASES.items():
        match = re.search(rf"{key}-([A-Za-z0-9]+)", name)
        if match:
            entities[entity] = match.group(1)
    return entities


def _ordered_unique(values: list[str]) -> list[str]:
    return list(OrderedDict((value, None) for value in values).keys())


def _read_tsv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        try:
            return next(reader)
        except StopIteration:
            return []


def _component_columns(header: list[str], prefix: str) -> list[str]:
    columns = [column for column in header if column.startswith(prefix)]

    def _sort_key(column: str) -> tuple[int, str]:
        match = re.search(r"(\d+)$", column)
        if match:
            return (int(match.group(1)), column)
        return (10_000, column)

    return sorted(columns, key=_sort_key)


def _pick_confounds_from_header(header: list[str], confounds_cfg: dict[str, Any]) -> list[str]:
    strategy = confounds_cfg.get("strategy", "acompcor6")
    include_fd = confounds_cfg.get("include_framewise_displacement", True)
    include_dvars = confounds_cfg.get("include_dvars", True)
    include_cosine = confounds_cfg.get("include_cosine_drift_terms", False)
    columns: list[str] = []

    if strategy == "none":
        columns = []
    elif strategy == "basic":
        columns = list(MOTION_BASE)
    elif strategy == "motion_24":
        for base in MOTION_BASE:
            for suffix in MOTION_24_SUFFIXES:
                columns.append(f"{base}{suffix}")
    elif strategy == "acompcor6":
        columns = list(MOTION_BASE)
        columns.extend(_component_columns(header, "a_comp_cor_")[:6])
    elif strategy == "custom":
        columns = list(confounds_cfg.get("custom_columns", []))
    else:
        raise ValueError(
            f"Unsupported confounds strategy '{strategy}'. "
            "Use one of: none, basic, motion_24, acompcor6, custom."
        )

    if include_fd:
        columns.append("framewise_displacement")
    if include_dvars:
        columns.append("dvars")
    if include_cosine:
        columns.extend(_component_columns(header, "cosine"))

    return _ordered_unique([column for column in columns if column in header])


def select_confound_columns(
    confound_paths: list[Path],
    confounds_cfg: dict[str, Any],
) -> list[str]:
    strict = confounds_cfg.get("strict", False)
    all_headers = [_read_tsv_header(path) for path in confound_paths]
    per_run_columns = [_pick_confounds_from_header(header, confounds_cfg) for header in all_headers]
    if not per_run_columns:
        return []

    common = list(per_run_columns[0])
    for run_columns in per_run_columns[1:]:
        common = [column for column in common if column in run_columns]

    if strict:
        expected = _ordered_unique(
            [column for run_columns in per_run_columns for column in run_columns]
        )
        missing = [column for column in expected if column not in common]
        if missing:
            raise ValueError(
                "Confounds columns are inconsistent across runs while strict=true. "
                f"Missing in at least one run: {', '.join(missing)}"
            )
    return common


def _collect_conditions(events_paths: list[Path], condition_column: str) -> list[str]:
    conditions: list[str] = []
    for events_path in events_paths:
        with events_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if reader.fieldnames is None or condition_column not in reader.fieldnames:
                raise ValueError(
                    f"Events file {events_path} is missing '{condition_column}' column"
                )
            for row in reader:
                value = (row.get(condition_column) or "").strip()
                if not value or value.lower() == "n/a":
                    continue
                conditions.append(value)
    unique_conditions = _ordered_unique(conditions)
    if not unique_conditions:
        raise ValueError(
            "No non-empty condition values were found in events.tsv files "
            f"for column '{condition_column}'"
        )
    return unique_conditions


def _contrast_from_weights(
    name: str,
    weights_map: dict[str, float],
    condition_to_regressor: dict[str, str],
    test: str = "t",
) -> dict[str, Any]:
    condition_list: list[str] = []
    weights: list[float] = []
    for condition, weight in weights_map.items():
        regressor = condition_to_regressor.get(condition, condition)
        if regressor not in condition_to_regressor.values():
            raise ValueError(
                f"Contrast '{name}' references unknown condition/regressor '{condition}'"
            )
        condition_list.append(regressor)
        weights.append(float(weight))
    return {
        "Name": name,
        "ConditionList": condition_list,
        "Weights": weights,
        "Test": test,
    }


def build_model(
    config: dict[str, Any],
    events_paths: list[Path],
    confound_columns: list[str],
    run_tasks: list[str],
) -> dict[str, Any]:
    design = config.get("design", {})
    condition_column = design.get("condition_column", "trial_type")
    conditions = list(design.get("conditions") or _collect_conditions(events_paths, condition_column))
    regressors = [f"{condition_column}.{condition}" for condition in conditions]
    condition_to_regressor = dict(zip(conditions, regressors))

    hrf_model = design.get("hrf_model", "spm")
    group_by = design.get("group_by", ["subject", "session", "task", "run"])

    node: dict[str, Any] = {
        "Level": "run",
        "Name": "run_level",
        "GroupBy": group_by,
        "Transformations": {
            "Transformer": "pybids-transforms-v1",
            "Instructions": [
                {"Name": "Factor", "Input": [condition_column]},
                {"Name": "Convolve", "Input": regressors, "Model": hrf_model},
            ],
        },
        "Model": {"X": [*regressors, *confound_columns, 1], "Type": "glm"},
    }

    auto_dummy = design.get("auto_dummy_contrasts", True)
    if auto_dummy:
        node["DummyContrasts"] = {"Conditions": regressors, "Test": "t"}

    contrasts_cfg = design.get("contrasts", [])
    contrasts: list[dict[str, Any]] = []
    if not contrasts_cfg and design.get("auto_task_vs_baseline", True):
        if len(regressors) == 1:
            contrasts.append(
                {
                    "Name": f"{conditions[0]}_vs_baseline",
                    "ConditionList": [regressors[0]],
                    "Weights": [1.0],
                    "Test": "t",
                }
            )
        else:
            contrasts.append(
                {
                    "Name": "task_vs_baseline",
                    "ConditionList": regressors,
                    "Weights": [round(1.0 / len(regressors), 8)] * len(regressors),
                    "Test": "t",
                }
            )

    for contrast_cfg in contrasts_cfg:
        name = contrast_cfg["name"]
        test = contrast_cfg.get("test", "t")
        if "weights" in contrast_cfg and isinstance(contrast_cfg["weights"], dict):
            contrasts.append(
                _contrast_from_weights(name, contrast_cfg["weights"], condition_to_regressor, test)
            )
            continue
        condition_list = contrast_cfg.get("condition_list")
        weights = contrast_cfg.get("weight_list")
        if not condition_list or not weights:
            raise ValueError(
                f"Contrast '{name}' must provide either a weights map or condition_list + weight_list"
            )
        mapped = [condition_to_regressor.get(condition, condition) for condition in condition_list]
        contrasts.append(
            {
                "Name": name,
                "ConditionList": mapped,
                "Weights": weights,
                "Test": test,
            }
        )

    if contrasts:
        node["Contrasts"] = contrasts

    model: dict[str, Any] = {
        "Name": design.get("name", "brainlife_fitlins_model"),
        "BIDSModelVersion": "1.0.0",
        "Description": design.get(
            "description",
            "Human-readable first-level model generated by the Brainlife FitLins app.",
        ),
        "Nodes": [node],
    }

    unique_tasks = _ordered_unique([task for task in run_tasks if task])
    if len(unique_tasks) == 1:
        model["Input"] = {"task": unique_tasks[0]}

    return model


def _entity_stem(entities: dict[str, str]) -> str:
    if not entities.get("subject"):
        raise ValueError("subject entity is required")
    task = entities.get("task")
    if not task:
        raise ValueError("task entity is required (either infer from filename or set in config)")

    parts = [f"sub-{entities['subject']}"]
    if entities.get("session"):
        parts.append(f"ses-{entities['session']}")
    parts.append(f"task-{task}")
    if entities.get("run"):
        parts.append(f"run-{entities['run']}")
    return "_".join(parts)


def _infer_extension(path: Path) -> str:
    name = path.name
    for ext in (".nii.gz", ".dtseries.nii", ".func.gii"):
        if name.endswith(ext):
            return ext
    if path.suffix:
        return path.suffix
    raise ValueError(f"Cannot infer file extension for {path}")


def _default_entities(config: dict[str, Any]) -> dict[str, str]:
    defaults = dict(config.get("default_entities", {}))
    defaults.setdefault("subject", "01")
    defaults.setdefault("task", "task")
    defaults.setdefault("space", config.get("fitlins", {}).get("space", "MNI152NLin2009cAsym"))
    return defaults


def _normalize_run_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    runs = config.get("runs")
    if runs:
        return list(runs)
    return [
        {
            "bold": config.get("bold"),
            "bold_json": config.get("bold_json"),
            "events_tsv": config.get("events_tsv"),
            "confounds_tsv": config.get("confounds_tsv"),
            "confounds_json": config.get("confounds_json"),
            "brain_mask": config.get("brain_mask"),
            "entities": config.get("entities", {}),
        }
    ]


def stage_bids_inputs(
    config: dict[str, Any],
    work_root: Path,
) -> tuple[Path, Path, list[Path], list[Path], list[str]]:
    bids_root = work_root / "bids"
    deriv_root = bids_root / "derivatives" / "fmriprep"
    bids_root.mkdir(parents=True, exist_ok=True)
    deriv_root.mkdir(parents=True, exist_ok=True)

    _write_json(
        bids_root / "dataset_description.json",
        {
            "Name": "Brainlife FitLins App Input",
            "BIDSVersion": "1.8.0",
            "DatasetType": "raw",
        },
    )
    _write_json(
        deriv_root / "dataset_description.json",
        {
            "Name": "Brainlife FitLins fMRIPrep Inputs",
            "BIDSVersion": "1.8.0",
            "DatasetType": "derivative",
            "GeneratedBy": [{"Name": "fMRIPrep"}],
        },
    )

    defaults = _default_entities(config)
    run_configs = _normalize_run_configs(config)
    if not run_configs or not run_configs[0].get("bold"):
        raise ValueError("No run inputs found. Provide either `runs` or top-level bold/events/confounds fields.")

    events_paths: list[Path] = []
    confounds_paths: list[Path] = []
    tasks: list[str] = []

    for index, run_cfg in enumerate(run_configs, start=1):
        bold = _to_path(run_cfg.get("bold"), f"runs[{index}].bold")
        events = _to_path(run_cfg.get("events_tsv"), f"runs[{index}].events_tsv")
        confounds = _to_path(run_cfg.get("confounds_tsv"), f"runs[{index}].confounds_tsv")
        bold_json = _to_path(run_cfg.get("bold_json"), f"runs[{index}].bold_json", required=False)
        confounds_json = _to_path(
            run_cfg.get("confounds_json"), f"runs[{index}].confounds_json", required=False
        )
        brain_mask = _to_path(run_cfg.get("brain_mask"), f"runs[{index}].brain_mask", required=False)

        inferred = infer_entities_from_name(bold)
        entities = {**inferred, **defaults, **run_cfg.get("entities", {})}
        stem = _entity_stem(entities)
        tasks.append(entities["task"])

        raw_base = bids_root / f"sub-{entities['subject']}"
        deriv_base = deriv_root / f"sub-{entities['subject']}"
        if entities.get("session"):
            raw_base = raw_base / f"ses-{entities['session']}"
            deriv_base = deriv_base / f"ses-{entities['session']}"
        raw_func = raw_base / "func"
        deriv_func = deriv_base / "func"

        bold_ext = _infer_extension(bold)
        raw_bold = raw_func / f"{stem}_bold{bold_ext}"
        raw_events = raw_func / f"{stem}_events.tsv"
        deriv_desc = config.get("fitlins", {}).get("desc_label", "preproc")
        space = entities.get("space", defaults["space"])
        deriv_space = f"_space-{space}" if space else ""
        deriv_bold = deriv_func / f"{stem}{deriv_space}_desc-{deriv_desc}_bold{bold_ext}"
        deriv_confounds = deriv_func / f"{stem}_desc-confounds_timeseries.tsv"

        _safe_link(bold, raw_bold)
        _safe_link(events, raw_events)
        _safe_link(bold, deriv_bold)
        _safe_link(confounds, deriv_confounds)

        if bold_json:
            raw_json = raw_func / f"{stem}_bold.json"
            deriv_json = deriv_func / f"{stem}{deriv_space}_desc-{deriv_desc}_bold.json"
            _safe_link(bold_json, raw_json)
            _safe_link(bold_json, deriv_json)
        else:
            tr_value = run_cfg.get("repetition_time", config.get("repetition_time"))
            if tr_value is None:
                raise ValueError(
                    f"Run {index} is missing bold_json and repetition_time. "
                    "Provide one of them so FitLins can read TR metadata."
                )
            metadata = {"RepetitionTime": float(tr_value), "TaskName": entities["task"]}
            _write_json(raw_func / f"{stem}_bold.json", metadata)
            _write_json(deriv_func / f"{stem}{deriv_space}_desc-{deriv_desc}_bold.json", metadata)

        if confounds_json:
            deriv_confounds_json = deriv_func / f"{stem}_desc-confounds_timeseries.json"
            _safe_link(confounds_json, deriv_confounds_json)

        if brain_mask:
            mask_ext = _infer_extension(brain_mask)
            deriv_mask = deriv_func / f"{stem}{deriv_space}_desc-brain_mask{mask_ext}"
            _safe_link(brain_mask, deriv_mask)

        events_paths.append(events)
        confounds_paths.append(confounds)

    return bids_root, deriv_root, events_paths, confounds_paths, tasks


def _fitlins_command_from_config(config: dict[str, Any]) -> list[str]:
    runtime = config.get("runtime", {})
    configured = runtime.get("fitlins_command")
    if isinstance(configured, str):
        return shlex.split(configured)
    if isinstance(configured, list):
        return [str(token) for token in configured]
    if shutil.which("fitlins"):
        return ["fitlins"]
    return ["python3", "-m", "fitlins"]


def _write_model_summary(
    path: Path,
    confounds: list[str],
    conditions: list[str],
    model_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FitLins Model Summary",
        "",
        "- Analysis level: `run` (first-level only)",
        f"- Model file: `{model_path}`",
        f"- Conditions ({len(conditions)}): {', '.join(conditions)}",
        f"- Confounds ({len(confounds)}): {', '.join(confounds) if confounds else 'None'}",
        "",
        "This model was generated from the app's human-readable `design` and `confounds` blocks.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _publish_outputs(out_dir: Path) -> None:
    exports = {
        "fitlins_output": "fitlins",
        "model_output": "model",
    }
    for dst_name, src_name in exports.items():
        src = (out_dir / src_name).resolve()
        if not src.exists():
            continue
        dst = Path(dst_name)
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        dst.symlink_to(src)


def run(config_path: str = "config.json") -> None:
    config = apply_brainlife_flat_overrides(_read_json(Path(config_path)))
    out_dir = Path("out_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    work_root = out_dir / ".work_fitlins"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    bids_root, deriv_root, events_paths, confound_paths, tasks = stage_bids_inputs(config, work_root)

    advanced = config.get("advanced", {})
    model_override = advanced.get("model_json_path")

    if model_override:
        model_path = _to_path(model_override, "advanced.model_json_path")
        model = _read_json(model_path)
        confound_columns = []
        conditions = list(config.get("design", {}).get("conditions", []))
        if not conditions:
            condition_column = config.get("design", {}).get("condition_column", "trial_type")
            try:
                conditions = _collect_conditions(events_paths, condition_column)
            except ValueError:
                conditions = []
    else:
        confound_columns = select_confound_columns(confound_paths, config.get("confounds", {}))
        model = build_model(config, events_paths, confound_columns, tasks)
        model_path = work_root / "model.json"
        _write_json(model_path, model)
        condition_column = config.get("design", {}).get("condition_column", "trial_type")
        conditions = list(config.get("design", {}).get("conditions") or _collect_conditions(events_paths, condition_column))

    saved_model = out_dir / "model" / "model-generated_smdl.json"
    _write_json(saved_model, model)

    analysis_level = "run"
    requested_level = config.get("analysis_level")
    if requested_level and requested_level != "run":
        raise ValueError(
            "This app currently supports first-level only. "
            "Set analysis_level to 'run' or omit it."
        )

    fitlins_cfg = config.get("fitlins", {})
    cmd = _fitlins_command_from_config(config)
    bids_root_abs = bids_root.resolve()
    out_fitlins_abs = (out_dir / "fitlins").resolve()
    deriv_root_abs = deriv_root.resolve()
    model_path_abs = Path(model_path).resolve()
    cmd.extend(
        [
            str(bids_root_abs),
            str(out_fitlins_abs),
            analysis_level,
            "-d",
            str(deriv_root_abs),
            "-m",
            str(model_path_abs),
            "--space",
            str(fitlins_cfg.get("space", "MNI152NLin2009cAsym")),
            "--desc-label",
            str(fitlins_cfg.get("desc_label", "preproc")),
            "--estimator",
            str(fitlins_cfg.get("estimator", "nilearn")),
        ]
    )

    if fitlins_cfg.get("drift_model"):
        cmd.extend(["--drift-model", str(fitlins_cfg["drift_model"])])
    if fitlins_cfg.get("n_cpus"):
        cmd.extend(["--n-cpus", str(fitlins_cfg["n_cpus"])])
    if fitlins_cfg.get("mem_gb"):
        cmd.extend(["--mem-gb", str(fitlins_cfg["mem_gb"])])
    if fitlins_cfg.get("work_dir"):
        cmd.extend(["--work-dir", str(fitlins_cfg["work_dir"])])
    if fitlins_cfg.get("drop_missing", True):
        cmd.append("--drop-missing")

    participant_labels = [str(participant) for participant in fitlins_cfg.get("participant_label", []) or []]
    if participant_labels:
        cmd.extend(["--participant-label", *participant_labels])

    force_index = [str(pattern) for pattern in fitlins_cfg.get("force_index", []) or []]
    if force_index:
        cmd.extend(["--force-index", *force_index])

    ignore = [str(pattern) for pattern in fitlins_cfg.get("ignore", []) or []]
    if ignore:
        cmd.extend(["--ignore", *ignore])
    for token in fitlins_cfg.get("extra_args", []) or []:
        cmd.append(str(token))

    _write_model_summary(
        out_dir / "model" / "model-summary.md",
        confounds=confound_columns,
        conditions=conditions,
        model_path=saved_model,
    )

    if advanced.get("dry_run", False):
        _publish_outputs(out_dir)
        print("Dry run enabled; skipping FitLins execution.")
        print("Generated model:", saved_model)
        return

    print("Running FitLins command:")
    print(" ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True)
    _publish_outputs(out_dir)


if __name__ == "__main__":
    run()
