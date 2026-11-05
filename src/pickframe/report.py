"""Export a culling result as a self-contained HTML page, CSV or JSON."""

from __future__ import annotations

import base64
import csv
import html
import json
from collections.abc import Callable
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image

from . import __version__
from .imaging import ImageLoadError, load_thumbnail
from .pipeline import CullResult
from .results import FLAG_TEXT

ProgressFn = Callable[[int, int, str], None]


def write_json(result: CullResult, path: Path) -> None:
    path.write_text(json.dumps(result.to_dict(with_embeddings=False), indent=2), encoding="utf-8")


CSV_FIELDS = [
    "group", "rank", "status", "rating", "score", "flags", "file", "other_files",
    "sharpness", "subject_sharpness", "exposure", "faces", "eyes_closed", "capture_time", "camera",
]  # fmt: skip


def write_csv(result: CullResult, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for group in result.groups:
            for shot_id in group:
                shot = result.shots[shot_id]
                d = result.decisions[shot_id]
                a = result.analyses[shot_id]
                when = (
                    datetime.fromtimestamp(a.capture_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                    if a.capture_time is not None
                    else ""
                )
                writer.writerow(
                    {
                        "group": d.group + 1,
                        "rank": d.rank,
                        "status": d.status,
                        "rating": "" if result.rating(shot_id) is None else result.rating(shot_id),
                        "score": f"{d.score:.3f}",
                        "flags": ";".join(d.flags),
                        "file": str(shot.primary),
                        "other_files": ";".join(str(p) for p in shot.files if p != shot.primary),
                        "sharpness": f"{a.sharpness:.3f}",
                        "subject_sharpness": f"{d.subject_sharpness:.3f}",
                        "exposure": f"{a.exposure:.3f}",
                        "faces": d.faces,
                        "eyes_closed": d.eyes_closed,
                        "capture_time": when,
                        "camera": a.camera,
                    }
                )


def _thumb_data_uri(path: Path, size: int) -> str:
    try:
        rgb = load_thumbnail(path, size)
    except (ImageLoadError, OSError):
        return ""
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


_CSS = """
:root{--bg:#0f1115;--panel:#171a21;--card:#1d212a;--text:#e8eaf0;--muted:#9aa3b2;--pick:#3ecf8e;
--alt:#7c8696;--reject:#ff5c69;--accent:#ffb547;--line:#2a2f3a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{position:sticky;top:0;z-index:5;background:rgba(15,17,21,.92);backdrop-filter:blur(8px);
border-bottom:1px solid var(--line);padding:14px 28px;display:flex;gap:24px;align-items:center;flex-wrap:wrap}
h1{font-size:18px;margin:0;letter-spacing:.2px}h1 span{color:var(--accent)}
.stats{display:flex;gap:10px;flex-wrap:wrap}.stat{background:var(--panel);border:1px solid var(--line);
border-radius:999px;padding:4px 12px;color:var(--muted)}.stat b{color:var(--text)}
.filters{margin-left:auto;display:flex;gap:6px}.filters button{background:var(--panel);color:var(--text);
border:1px solid var(--line);border-radius:8px;padding:6px 12px;cursor:pointer;font:inherit}
.filters button.on{border-color:var(--accent);color:var(--accent)}
main{padding:20px 28px 60px}.group{margin:0 0 22px;background:var(--panel);border:1px solid var(--line);
border-radius:14px;padding:14px 16px}.group h2{font-size:13px;font-weight:600;color:var(--muted);margin:0 0 10px;
text-transform:uppercase;letter-spacing:.6px}.cards{display:flex;gap:14px;flex-wrap:wrap}
.card{width:236px;background:var(--card);border-radius:10px;overflow:hidden;border:2px solid transparent}
.card.pick{border-color:var(--pick)}.card.reject{border-color:var(--reject)}.card.reject .img img{opacity:.55}
.img{position:relative;height:170px;background:#0a0b0e;display:flex;align-items:center;justify-content:center;overflow:hidden}
.img img{max-width:224px;max-height:170px;display:block}.frame{position:relative;display:inline-block;line-height:0}
.face{position:absolute;border:1.5px solid rgba(62,207,142,.95);border-radius:4px}
.face.closed{border-color:var(--reject)}
.badge{position:absolute;top:8px;left:8px;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700;
text-transform:uppercase;letter-spacing:.5px;background:var(--alt);color:#0f1115}
.pick .badge{background:var(--pick)}.reject .badge{background:var(--reject)}
.stars{position:absolute;top:8px;right:8px;background:rgba(0,0,0,.6);border-radius:999px;padding:1px 8px;
color:var(--accent);font-size:12px;letter-spacing:1px}
.meta{padding:9px 11px 11px}.name{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bar{height:5px;background:#2b303b;border-radius:3px;margin:7px 0 6px;overflow:hidden}
.bar i{display:block;height:100%;background:linear-gradient(90deg,#ff5c69,#ffb547,#3ecf8e)}
.why{color:var(--muted);font-size:12px;min-height:17px}.why .f{color:var(--reject)}
footer{color:var(--muted);text-align:center;padding:20px;font-size:12px}
a{color:var(--accent)}
"""

_JS = """
document.querySelectorAll('.filters button').forEach(b=>b.onclick=()=>{
document.querySelectorAll('.filters button').forEach(x=>x.classList.toggle('on',x===b));
const f=b.dataset.f;document.querySelectorAll('.card').forEach(c=>{
c.style.display=(f==='all'||c.classList.contains(f))?'':'none'});
document.querySelectorAll('.group').forEach(g=>{g.style.display=[...g.querySelectorAll('.card')]
.some(c=>c.style.display!=='none')?'':'none'})});
"""


def _stars(rating: int | None) -> str:
    if rating is None or rating == 0:
        return ""
    if rating < 0:
        return "&#10005;"
    return "&#9733;" * rating


def render_html(result: CullResult, thumb_size: int = 360, progress: ProgressFn | None = None) -> str:
    from .scoring import face_closed, prominent_faces

    counts = result.counts()
    parts: list[str] = []
    done = 0
    for group in result.groups:
        cards = []
        for shot_id in sorted(group, key=lambda i: result.decisions[i].rank):
            shot = result.shots[shot_id]
            d = result.decisions[shot_id]
            a = result.analyses[shot_id]
            uri = _thumb_data_uri(shot.primary, thumb_size)
            faces_html = "".join(
                '<span class="face{cls}" style="left:{x:.2f}%;top:{y:.2f}%;width:{w:.2f}%;height:{h:.2f}%"></span>'.format(
                    cls=" closed" if face_closed(f, result.settings) else "",
                    x=f.box[0] * 100, y=f.box[1] * 100, w=f.box[2] * 100, h=f.box[3] * 100,
                )
                for f in prominent_faces(a.faces)
            )
            why = ", ".join(
                f'<span class="f">{html.escape(FLAG_TEXT.get(flag, flag))}</span>'
                if flag in ("blurry", "eyes-closed", "unreadable")
                else html.escape(FLAG_TEXT.get(flag, flag))
                for flag in d.flags
            ) or html.escape(d.summary)
            img = f'<span class="frame"><img loading="lazy" src="{uri}" alt="">{faces_html}</span>' if uri else ""
            cards.append(
                f'<div class="card {d.status}" title="{html.escape(str(shot.primary))}">'
                f'<div class="img">{img}<span class="badge">{d.status}</span>'
                f'<span class="stars">{_stars(result.rating(shot_id))}</span></div>'
                f'<div class="meta"><div class="name">{html.escape(shot.label())}</div>'
                f'<div class="bar"><i style="width:{d.score * 100:.0f}%"></i></div>'
                f'<div class="why">Score {d.score * 100:.0f} &middot; {why}</div></div></div>'
            )
            done += 1
            if progress:
                progress(done, len(result.shots), shot.name)
        first = result.shots[group[0]]
        title = f"Group {result.decisions[group[0]].group + 1} &middot; {len(group)} shot{'s' if len(group) != 1 else ''}"
        title += f" &middot; {html.escape(first.primary.parent.name)}"
        parts.append(f'<section class="group"><h2>{title}</h2><div class="cards">{"".join(cards)}</div></section>')

    stats = "".join(
        f'<span class="stat"><b>{counts[key]}</b> {label}</span>'
        for key, label in (("shots", "shots"), ("groups", "groups"), ("pick", "picks"),
                           ("alternate", "alternates"), ("reject", "rejects"))
    )  # fmt: skip
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        "<!DOCTYPE html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>Pickframe report</title><style>{_CSS}</style></head><body>"
        f"<header><h1>Pick<span>frame</span> report</h1><div class=stats>{stats}</div>"
        '<div class=filters><button class=on data-f=all>All</button><button data-f=pick>Picks</button>'
        "<button data-f=alternate>Alternates</button><button data-f=reject>Rejects</button></div></header>"
        f"<main>{''.join(parts)}</main>"
        f'<footer>Generated {generated} by <a href="https://github.com/lobojoel2003/Pickframe">Pickframe</a> '
        f"{__version__} &middot; grouping: {html.escape(result.embedder or 'n/a')}</footer>"
        f"<script>{_JS}</script></body></html>"
    )


def write_html(result: CullResult, path: Path, thumb_size: int = 360, progress: ProgressFn | None = None) -> None:
    path.write_text(render_html(result, thumb_size, progress), encoding="utf-8")


def write_report(result: CullResult, path: Path, progress: ProgressFn | None = None) -> None:
    """Pick the format from the file extension (.html, .csv or .json)."""
    suffix = path.suffix.lower()
    if suffix in (".html", ".htm"):
        write_html(result, path, progress=progress)
    elif suffix == ".csv":
        write_csv(result, path)
    elif suffix == ".json":
        write_json(result, path)
    else:
        raise ValueError(f"unsupported report format {suffix!r} (use .html, .csv or .json)")
