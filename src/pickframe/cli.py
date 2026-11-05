"""Command-line interface: ``pickframe cull``, ``reset``, ``models``, ``gui`` ..."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import Counter
from pathlib import Path

from . import __version__, models
from .cache import AnalysisCache, default_path
from .config import RejectStyle, Settings, SidecarStyle
from .pipeline import Analyzer, CullResult
from .results import FLAG_TEXT, PICK, REJECT

EPILOG = """examples:
  pickframe cull ~/Pictures/wedding                 analyse and write XMP ratings
  pickframe cull ~/Pictures/shoot -r --dry-run --report report.html
  pickframe cull . --sidecar lightroom --reject-as one-star
  pickframe reset ~/Pictures/wedding                undo everything Pickframe wrote
  pickframe models download clip                    better grouping (88 MB, once)
  pickframe gui ~/Pictures/wedding                  open the desktop app
"""


def _console(quiet: bool = False):
    from rich.console import Console

    return Console(stderr=True, quiet=quiet, highlight=False)


def _probability(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _add_analysis_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("analysis")
    group.add_argument("-r", "--recursive", action="store_true", help="include sub-folders")
    group.add_argument(
        "--embedder", choices=["auto", "clip", "classic"], default="auto",
        help="similarity model for grouping (auto = CLIP if downloaded, else classic)",
    )  # fmt: skip
    group.add_argument("--workers", type=_positive_int, default=0, help="worker threads (default: CPU count, max 8)")
    group.add_argument("--no-cache", action="store_true", help="re-analyse every file")

    tuning = parser.add_argument_group("grouping and decisions")
    tuning.add_argument("--similarity", type=_probability, help="similarity to join a group (0-1)")
    tuning.add_argument("--max-gap", type=float, default=60.0, metavar="SECONDS",
                        help="max time between shots of one group (default 60)")  # fmt: skip
    tuning.add_argument("--blur", type=_probability, default=0.30, metavar="0-1",
                        help="sharpness below which a shot is blurry (default 0.30)")  # fmt: skip
    tuning.add_argument("--eyes", type=_probability, default=0.55, metavar="0-1",
                        help="eyes-closed probability that rejects a shot (default 0.55)")  # fmt: skip
    tuning.add_argument("--picks", type=_positive_int, default=1, metavar="N",
                        help="picks per group (default 1)")  # fmt: skip
    tuning.add_argument("--smile-weight", type=_probability, default=0.15, metavar="0-1",
                        help="how much smiles count in face scores (default 0.15)")  # fmt: skip


def _settings_from(args: argparse.Namespace) -> Settings:
    return Settings(
        embedder=args.embedder,
        workers=args.workers,
        similarity=args.similarity,
        max_gap=args.max_gap,
        blur_threshold=args.blur,
        eyes_threshold=args.eyes,
        picks_per_group=args.picks,
        smile_weight=args.smile_weight,
        reject_style=RejectStyle(getattr(args, "reject_as", "reject")),
        sidecar_style=SidecarStyle(getattr(args, "sidecar", "auto")),
        write_flags=not getattr(args, "no_flags", False),
        pick_label=getattr(args, "label", "") or "",
        overwrite_manual=getattr(args, "overwrite", False),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pickframe",
        description="Free, local AI photo culling. Groups near-duplicates, flags blur and "
        "closed eyes, scores faces and writes XMP star ratings.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-V", "--version", action="store_true", help="show version and build information")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    cull = sub.add_parser("cull", help="analyse photos and write ratings", epilog=EPILOG,
                          formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    cull.add_argument("paths", nargs="+", type=Path, help="photo files or folders")
    _add_analysis_options(cull)
    out = cull.add_argument_group("output")
    out.add_argument("-n", "--dry-run", action="store_true", help="analyse only; do not write any metadata")
    out.add_argument("--sidecar", choices=[s.value for s in SidecarStyle], default="auto",
                     help="sidecar naming: auto (both), darktable (IMG.CR3.xmp), lightroom (IMG.xmp)")  # fmt: skip
    out.add_argument("--embed-jpeg", action="store_true",
                     help="also embed XMP inside JPEG files (needed for Lightroom with JPEGs)")  # fmt: skip
    out.add_argument("--reject-as", choices=[s.value for s in RejectStyle], default="reject",
                     help="write rejects as rating -1 + reject flag, 1 star, or leave unrated")  # fmt: skip
    out.add_argument("--no-flags", action="store_true", help="do not write pick/reject flags")
    out.add_argument("--label", metavar="COLOUR", help="colour label for picks, e.g. Green")
    out.add_argument("--overwrite", action="store_true", help="overwrite ratings you set by hand")
    out.add_argument("--report", type=Path, action="append", default=[], metavar="FILE",
                     help="write a report (.html, .csv or .json); may be repeated")  # fmt: skip
    out.add_argument("--save-session", type=Path, metavar="FILE", help="save results to open later in the GUI")
    out.add_argument("-l", "--list", action="store_true", help="print a line for every photo")
    out.add_argument("-q", "--quiet", action="store_true", help="only print errors")

    reset = sub.add_parser("reset", help="remove ratings and flags written by Pickframe")
    reset.add_argument("paths", nargs="+", type=Path)
    reset.add_argument("-r", "--recursive", action="store_true")
    reset.add_argument("-n", "--dry-run", action="store_true")
    reset.add_argument("--keep-jpeg", action="store_true", help="do not touch XMP embedded in JPEGs")

    info = sub.add_parser("info", help="show the measurements for individual photos")
    info.add_argument("files", nargs="+", type=Path)
    info.add_argument("--json", action="store_true", help="print JSON")

    models_cmd = sub.add_parser("models", help="list, download or remove optional models")
    models_sub = models_cmd.add_subparsers(dest="action", metavar="ACTION")
    models_sub.add_parser("list", help="show bundled and optional models")
    dl = models_sub.add_parser("download", help="download an optional model")
    dl.add_argument("name", nargs="?", default="clip", choices=sorted(models.REMOTE_MODELS))
    dl.add_argument("--force", action="store_true", help="download again")
    rm = models_sub.add_parser("remove", help="delete a downloaded model")
    rm.add_argument("name", choices=sorted(models.REMOTE_MODELS))
    models_sub.add_parser("path", help="print the model folder")

    cache = sub.add_parser("cache", help="manage the analysis cache")
    cache.add_argument("action", choices=["clear", "path"])

    gui = sub.add_parser("gui", help="open the desktop app")
    gui.add_argument("path", nargs="?", type=Path, help="folder to open")
    return parser


# -- commands ---------------------------------------------------------------------
def _run_analysis(args: argparse.Namespace, settings: Settings, console) -> CullResult:
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
    )

    analyzer = Analyzer(settings, use_cache=not args.no_cache)
    cancel = threading.Event()
    with Progress(
        TextColumn("[bold]Analysing[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn("[dim]{task.fields[name]}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("analyse", total=None, name="")

        def update(done: int, total: int, name: str) -> None:
            progress.update(task, completed=done, total=total, name=name[:40])

        try:
            return analyzer.run(args.paths, recursive=args.recursive, progress=update, cancel=cancel)
        except KeyboardInterrupt:
            cancel.set()
            raise
        finally:
            analyzer.close()


def _print_summary(result: CullResult, console) -> None:
    from rich.table import Table

    counts = result.counts()
    reasons = Counter(flag for d in result.decisions.values() if d.status == REJECT for flag in d.flags)
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Photos", f"[bold]{counts['shots']}[/]")
    table.add_row("Groups", f"[bold]{counts['groups']}[/]")
    table.add_row("Picks", f"[bold green]{counts['pick']}[/]")
    table.add_row("Alternates", f"[bold]{counts['alternate']}[/]")
    reject_detail = ", ".join(f"{n} {FLAG_TEXT.get(flag, flag)}" for flag, n in reasons.most_common())
    table.add_row("Rejects", f"[bold red]{counts['reject']}[/]" + (f"  [dim]({reject_detail})[/]" if reject_detail else ""))
    rate = counts["shots"] / result.elapsed if result.elapsed > 0 else 0.0
    cached = f", {result.cached} from cache" if result.cached else ""
    table.add_row("Time", f"{result.elapsed:.1f} s ({rate:.1f} photos/s{cached}), grouping: {result.embedder}")
    console.print(table)


def _print_list(result: CullResult, console) -> None:
    from rich.table import Table

    table = Table(header_style="bold", box=None, padding=(0, 1))
    for col in ("Group", "Rating", "Status", "Score", "Photo", "Notes"):
        table.add_column(col)
    style = {PICK: "green", REJECT: "red"}
    for group in result.groups:
        for shot_id in sorted(group, key=lambda i: result.decisions[i].rank):
            d = result.decisions[shot_id]
            rating = result.rating(shot_id)
            stars = "rejected" if rating == -1 else ("★" * rating if rating else "-")
            table.add_row(
                str(d.group + 1), stars, f"[{style.get(d.status, 'white')}]{d.status}[/]",
                f"{d.score * 100:.0f}", result.shots[shot_id].label(), d.summary,
            )  # fmt: skip
    console.print(table)


def cmd_cull(args: argparse.Namespace) -> int:
    from .report import write_report
    from .writer import write_result

    console = _console(args.quiet)
    settings = _settings_from(args)
    if settings.embedder == "clip" and not models.is_installed("clip"):
        console.print("[red]The CLIP model is not installed.[/] Run [bold]pickframe models download clip[/].")
        return 2
    for path in args.paths:
        if not path.exists():
            console.print(f"[red]Not found:[/] {path}")
            return 2
    try:
        result = _run_analysis(args, settings, console)
    except KeyboardInterrupt:
        console.print("[yellow]Cancelled.[/]")
        return 130
    if not result.shots:
        console.print("[yellow]No photos found.[/] Use -r to include sub-folders.")
        return 1

    _print_summary(result, console)
    if args.list:
        _print_list(result, console)
    unreadable = [result.analyses[s.id] for s in result.shots if result.analyses[s.id].error]
    for analysis in unreadable[:10]:
        console.print(f"[yellow]Skipped:[/] {analysis.error}")
    if len(unreadable) > 10:
        console.print(f"[yellow]... and {len(unreadable) - 10} more unreadable files[/]")

    report = write_result(result, dry_run=args.dry_run, embed_jpeg=args.embed_jpeg)
    verb = "Would write" if args.dry_run else "Wrote"
    console.print(f"{verb} XMP for [bold]{len(report.written)}[/] file(s); {len(report.unchanged)} already up to date.")
    if report.skipped_manual:
        console.print(
            f"[yellow]Kept {len(report.skipped_manual)} rating(s) you set by hand[/] (use --overwrite to replace them)."
        )
    for target, message in report.errors[:10]:
        console.print(f"[red]Error[/] {target}: {message}")
    if settings.sidecar_style == SidecarStyle.LIGHTROOM and not args.embed_jpeg:
        jpegs = sum(1 for s in result.shots for f in s.files if f.suffix.lower() in (".jpg", ".jpeg"))
        if jpegs:
            console.print("[dim]Lightroom ignores sidecars for JPEG files; add --embed-jpeg to rate them too.[/]")

    for path in args.report:
        try:
            write_report(result, path)
        except (OSError, ValueError) as exc:
            console.print(f"[red]Could not write report {path}:[/] {exc}")
            return 1
        console.print(f"Report: {path}")
    if args.save_session:
        result.save(args.save_session)
        console.print(f"Session: {args.save_session}")
    return 1 if report.errors else 0


def cmd_reset(args: argparse.Namespace) -> int:
    from .writer import reset

    console = _console()
    report = reset(args.paths, recursive=args.recursive, include_jpeg=not args.keep_jpeg, dry_run=args.dry_run)
    verb = "Would reset" if args.dry_run else "Reset"
    console.print(f"{verb} {len(report.written)} file(s) and removed {len(report.removed)} Pickframe sidecar(s).")
    for target, message in report.errors:
        console.print(f"[red]Error[/] {target}: {message}")
    return 1 if report.errors else 0


def cmd_info(args: argparse.Namespace) -> int:
    from .scoring import score_shot

    console = _console()
    analyzer = Analyzer(Settings(), use_cache=False)
    rows = []
    for path in args.files:
        analysis = analyzer.analyze_image(path)
        decision = score_shot(analysis, analyzer.settings)
        rows.append({"file": str(path), "analysis": analysis.to_dict(with_embedding=False),
                     "score": round(decision.score, 4), "flags": decision.flags})  # fmt: skip
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    for row in rows:
        a = row["analysis"]
        console.rule(Path(row["file"]).name)
        if a["error"]:
            console.print(f"[red]{a['error']}[/]")
            continue
        console.print(f"Size {a['width']}x{a['height']}  camera {a['camera'] or '-'}  source {a['source']}")
        console.print(f"Sharpness {a['sharpness']:.2f}  exposure {a['exposure']:.2f} (brightness {a['brightness']:.2f}, "
                      f"highlights {a['highlights']:.1%}, shadows {a['shadows']:.1%})")  # fmt: skip
        for i, face in enumerate(analysis_faces(a), start=1):
            console.print(f"Face {i}: eyes closed {face.eyes_closed_probability:.2f}, eye focus {face.sharpness:.2f}, "
                          f"smile {face.smile:.2f}, facing camera {face.frontal:.2f}")  # fmt: skip
        flags = ", ".join(row["flags"]) or "none"
        console.print(f"Score [bold]{row['score'] * 100:.0f}[/]  flags: {flags}")
    return 0


def analysis_faces(data: dict):
    from .analysis.faces import Face

    return [Face.from_dict(f) for f in data["faces"]]


def cmd_models(args: argparse.Namespace) -> int:
    from rich.progress import BarColumn, DownloadColumn, Progress, TransferSpeedColumn

    console = _console()
    action = args.action or "list"
    if action == "path":
        print(models.models_dir())
        return 0
    if action == "remove":
        removed = models.remove(args.name)
        console.print(f"Removed {args.name}." if removed else f"{args.name} is not installed.")
        return 0
    if action == "download":
        model = models.REMOTE_MODELS[args.name]
        console.print(f"Downloading {model.description} (~{model.approx_mb} MB, {model.license} license)")
        with Progress(BarColumn(), DownloadColumn(), TransferSpeedColumn(), console=console, transient=True) as bar:
            task = bar.add_task("download", total=None)
            try:
                path = models.download(args.name, lambda done, total: bar.update(task, completed=done, total=total),
                                       force=args.force)  # fmt: skip
            except models.ModelError as exc:
                console.print(f"[red]{exc}[/]")
                return 1
        console.print(f"Saved to {path}")
        return 0
    console.print("[bold]Bundled[/] (always available)")
    for name in (models.FACE_DETECTOR, models.FACE_LANDMARKS, models.FACE_BLENDSHAPES):
        console.print(f"  {name}")
    console.print("[bold]Optional[/]")
    for model in models.REMOTE_MODELS.values():
        state = "[green]installed[/]" if models.is_installed(model.name) else "not installed"
        console.print(f"  {model.name:<8} {state:<22} {model.description} (~{model.approx_mb} MB)")
    console.print(f"Folder: {models.models_dir()}")
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    if args.action == "path":
        print(default_path())
        return 0
    with AnalysisCache() as cache:
        count = cache.clear()
    _console().print(f"Removed {count} cached analysis result(s).")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    try:
        from .gui import main as gui_main
    except ImportError:
        _console().print("[red]The desktop app needs PySide6:[/] pip install 'pickframe[gui]'")
        return 1
    return gui_main([str(args.path)] if args.path else [])


def _safe_streams() -> None:
    """Never crash on consoles that cannot print ★ (e.g. legacy Windows code pages)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _safe_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from .version import version_string

        print(version_string(verbose=True))
        return 0
    handlers = {
        "cull": cmd_cull,
        "reset": cmd_reset,
        "info": cmd_info,
        "models": cmd_models,
        "cache": cmd_cache,
        "gui": cmd_gui,
    }
    if args.command is None:
        parser.print_help()
        return 0
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        sys.stderr.close()
        return 0


__all__ = ["__version__", "build_parser", "main"]
