"""
HTML report generation and result summary for C2CAD-Bench v7.0.
Click-to-expand detail panels per level: position errors, orientation errors,
connectivity breakdown, gravity issues, score decomposition, JSON comparison,
plus engineering sections: DFM (interference/clearance/wall),
assembly mates, Chamfer Distance, symmetry/pattern, and reasoning chain.
"""

import html as _h
import json
from datetime import datetime
from typing import List
from dataclasses import asdict

from .config import CFG, LevelResult
from .generators import GENERATOR_NAMES


# ═══════════════════════════════════════════════════════════════
# DETAIL PANEL BUILDER
# ═══════════════════════════════════════════════════════════════

def _detail_panel(r: LevelResult, uid: str) -> str:
    """Build the click-to-expand detail panel for a single level result."""

    # ── Score Breakdown ──────────────────────────────────────────────────────
    if r.expected_parts == 0:
        count_acc = 1.0
    else:
        count_acc = max(0.0, 1.0 - abs(r.part_count - r.expected_parts) / r.expected_parts)

    type_score   = 1.0 if r.types_correct else 0.0
    orient_score = r.orientation_accuracy if r.orientations_checked > 0 else 1.0

    # Detect engineering mode: any engineering score was populated
    is_eng = (r.mate_score > 0 or r.clearance_score > 0 or r.chamfer_distance >= 0
              or r.symmetry_score > 0 or r.pattern_score > 0 or r.wall_details)

    if is_eng:
        # DFM combined score for display
        dfm_scores = []
        if r.clearance_score > 0 or r.clearance_details:
            dfm_scores.append(r.clearance_score)
        if r.wall_details:
            dfm_scores.append(1.0 if r.wall_ok else 0.0)
        dfm_score = (sum(dfm_scores) / len(dfm_scores)) if dfm_scores else 1.0

        # v7.0 engineering mode: 8 scoring dimensions
        geo_acc = 0.8 * r.position_accuracy + 0.2 * type_score
        spatial = 0.5 * r.connectivity_score + 0.5 * r.gravity_score
        reasoning = 0.6 * r.reasoning_depth + 0.4 * r.reasoning_chain_score if r.reasoning_chain_score > 0 else r.reasoning_depth
        eng_parts = []
        if r.mate_score > 0 or r.mate_details:
            eng_parts.append(r.mate_score)
        if dfm_scores:
            eng_parts.append(sum(dfm_scores) / len(dfm_scores))
        eng_score = (sum(eng_parts) / len(eng_parts)) if eng_parts else 1.0
        pat_parts = []
        if r.symmetry_score > 0 or r.symmetry_details:
            pat_parts.append(r.symmetry_score)
        if r.pattern_score > 0 or r.pattern_details:
            pat_parts.append(r.pattern_score)
        struct_score = (sum(pat_parts) / len(pat_parts)) if pat_parts else 1.0

        components = [
            ("Geometric",    "30%", 0.30, geo_acc,              "Position accuracy (80%) + type correctness (20%)"),
            ("Part Count",   "10%", 0.10, count_acc,            "How close the number of generated shapes is to expected"),
            ("Spatial",      "15%", 0.15, spatial,              "Connectivity (50%) + gravity support (50%)"),
            ("Surface",      "10%", 0.10, r.chamfer_score,      "Surface-sampled Chamfer Distance: captures size and shape errors"),
            ("Orientation",  " 5%", 0.05, orient_score,         "Cylinder axis vectors match ground truth within 15°"),
            ("Reasoning",    "10%", 0.10, reasoning,            "Depth of step-by-step derivation + intermediate value chain"),
            ("Engineering",  "15%", 0.15, eng_score,            "Assembly mates + DFM (clearance fit, wall thickness)"),
            ("Patterns",     " 5%", 0.05, struct_score,         "Symmetry and circular/linear pattern validation"),
        ]
    else:
        # v7.0 core mode: 7 scoring dimensions (engineering weight redistributed)
        geo_acc = 0.8 * r.position_accuracy + 0.2 * type_score
        spatial = 0.5 * r.connectivity_score + 0.5 * r.gravity_score

        components = [
            ("Geometric",    "35%", 0.35, geo_acc,               "Position accuracy (80%) + type correctness (20%)"),
            ("Part Count",   "12%", 0.12, count_acc,             "How close the number of generated shapes is to expected"),
            ("Spatial",      "23%", 0.23, spatial,               "Connectivity (50%) + gravity support (50%)"),
            ("Surface",      "10%", 0.10, r.chamfer_score if r.chamfer_distance >= 0 else r.position_accuracy, "Surface-sampled Chamfer Distance"),
            ("Orientation",  " 5%", 0.05, orient_score,          "Cylinder axis vectors match ground truth within 15°"),
            ("Reasoning",    "10%", 0.10, r.reasoning_depth,     "Presence of step-by-step formulas, loops, and derivations"),
            ("No Interference", " 5%", 0.05, r.interference_score, "Parts occupy their own space without unintended overlaps"),
        ]

    score_items = ""
    for name, wt, w, val, tip in components:
        col = "#3be8b0" if val >= 0.8 else ("#fbbf24" if val >= 0.4 else "#f472b6")
        contrib = w * val
        score_items += (
            f'<div class="sci" title="{_h.escape(tip)}">'
            f'<div class="scv" style="color:{col}">{val:.0%}</div>'
            f'<div class="scn">{name}</div>'
            f'<div class="scw">×{wt} = <b>{contrib:.3f}</b></div>'
            f'</div>'
        )

    score_section = (
        f'<div class="dp-sec">'
        f'<div class="ds">Score Breakdown → <b style="color:#c8cfe0">{r.score:.0%}</b>'
        f'  <span class="ds-hint">(hover each component for description)</span></div>'
        f'<div class="sc-grid">{score_items}</div>'
        f'</div>'
    )

    # ── Position Details ─────────────────────────────────────────────────────
    pos_section = ""
    if r.position_details:
        rows = ""
        for d in r.position_details:
            ok = "OK" in d and "FAIL" not in d
            cls = "ok" if ok else "fail"
            rows += f'<tr class="{cls}"><td class="mono">{_h.escape(d)}</td></tr>'

        n_fail = r.positions_checked - r.positions_correct
        if n_fail > 0:
            expl = (
                f'<div class="dp-note">'
                f'⚠ <b>{n_fail}</b> of {r.positions_checked} checked shape(s) are in the wrong position. '
                f'In a physical assembly, misplaced shapes cause <b>collisions</b> (if too close) or '
                f'<b>structural gaps</b> (if too far). A gear whose centre is off by even a few millimetres '
                f'will not mesh correctly with its neighbour.'
                f'</div>'
            )
        else:
            expl = (
                '<div class="dp-note ok-note">'
                '✓ All checked positions match ground truth within tolerance. '
                'Every shape centroid is where the specification requires it to be.'
                '</div>'
            )

        pos_section = (
            f'<div class="dp-sec">'
            f'<div class="ds">Position Check — {r.positions_correct}/{r.positions_checked} correct</div>'
            f'{expl}'
            f'<table class="dpt"><tbody>{rows}</tbody></table>'
            f'</div>'
        )
    elif r.json_ok:
        pos_section = (
            '<div class="dp-sec">'
            '<div class="ds">Position Check</div>'
            '<div class="dp-note">No positions were checked for this level (stress test sampling may have yielded zero overlap).</div>'
            '</div>'
        )

    # ── Orientation Details ──────────────────────────────────────────────────
    orient_section = ""
    if r.orientation_details:
        rows = ""
        for d in r.orientation_details:
            ok = "OK" in d and "FAIL" not in d
            cls = "ok" if ok else "fail"
            rows += f'<tr class="{cls}"><td class="mono">{_h.escape(d)}</td></tr>'

        n_fail = r.orientations_checked - r.orientations_correct
        if n_fail > 0:
            expl = (
                f'<div class="dp-note">'
                f'⚠ <b>{n_fail}</b> cylinder axis/axes are wrong (angle &gt; 15°). '
                f'A cylinder is defined by its central axis — if it points in the wrong direction, '
                f'a <b>vertical shaft becomes a diagonal rod</b>, a horizontal bore becomes vertical, '
                f'and any bolts, gears, or joints depending on that axis will be misaligned. '
                f'Antiparallel (180° flipped) is treated as correct.'
                f'</div>'
            )
        else:
            expl = (
                '<div class="dp-note ok-note">'
                '✓ All cylinder axes are correctly oriented (within 15°). '
                'Every shaft, bore, and pipe points in the direction the specification requires.'
                '</div>'
            )

        orient_section = (
            f'<div class="dp-sec">'
            f'<div class="ds">Orientation Check — {r.orientations_correct}/{r.orientations_checked} cylinders correct</div>'
            f'{expl}'
            f'<table class="dpt"><tbody>{rows}</tbody></table>'
            f'</div>'
        )

    # ── Connectivity ─────────────────────────────────────────────────────────
    if not r.connectivity_ok:
        flt_ids = r.connectivity_floating_ids
        id_str = ", ".join(f"#{i}" for i in flt_ids[:25])
        if len(flt_ids) > 25:
            id_str += f" … +{len(flt_ids)-25} more"
        penalty = 1.0 - r.connectivity_score
        conn_section = (
            f'<div class="dp-sec">'
            f'<div class="ds">Connectivity — {r.connectivity_islands} Disconnected Islands (FAIL)</div>'
            f'<div class="dp-note">'
            f'⚠ The assembly splits into <b>{r.connectivity_islands}</b> separate groups of shapes that do not '
            f'touch each other. Isolated shapes: {id_str or "(none identified)"}. '
            f'In a real structure, disconnected parts would <b>drift freely</b> — there is no structural '
            f'load path binding them to the rest. This typically happens when the LLM places parts far from '
            f'where they belong, leaving isolated clusters. '
            f'Score penalty: <b>{penalty:.0%}</b> (−0.2 per extra island).'
            f'</div>'
            f'</div>'
        )
    else:
        conn_section = (
            '<div class="dp-sec">'
            '<div class="ds">Connectivity — Fully Connected ✓</div>'
            '<div class="dp-note ok-note">'
            '✓ All shapes form a single connected assembly. Every part touches at least one neighbour '
            'via overlapping bounding boxes, ensuring a valid structural load path exists throughout the assembly.'
            '</div>'
            '</div>'
        )

    # ── Gravity ───────────────────────────────────────────────────────────────
    flt = r.gravity_floating_ids
    if not r.gravity_ok and flt:
        flt_str = ", ".join(f"#{i}" for i in flt[:25])
        if len(flt) > 25:
            flt_str += f" … +{len(flt)-25} more"
        penalty = 1.0 - r.gravity_score
        grav_section = (
            f'<div class="dp-sec">'
            f'<div class="ds">Gravity — {len(flt)} Floating Shape(s) (FAIL)</div>'
            f'<div class="dp-note">'
            f'⚠ Shape(s) {flt_str} are suspended in mid-air: they neither touch the ground plane '
            f'(Z ≈ 0) nor rest on any lower shape. Under real gravity these parts would <b>fall</b>. '
            f'This usually means the LLM stacked parts using wrong Z offsets — e.g. placing a lid '
            f'50 mm above a box that is only 40 mm tall leaves a 10 mm air gap. '
            f'Score penalty: <b>{penalty:.0%}</b>.'
            f'</div>'
            f'</div>'
        )
    else:
        grav_section = (
            '<div class="dp-sec">'
            '<div class="ds">Gravity — All Shapes Supported ✓</div>'
            '<div class="dp-note ok-note">'
            '✓ Every shape either rests on the ground (Z ≈ 0) or sits on a shape below it. '
            'The assembly would stand stably under gravity with no floating parts.'
            '</div>'
            '</div>'
        )

    # ── Reasoning Quality ────────────────────────────────────────────────────
    loop_txt = ("Loop / trigonometric formula logic detected — the LLM derived positions "
                "programmatically rather than hardcoding each coordinate."
                if r.has_loop_logic else
                "No loop or formula logic detected. Positions appear hardcoded. "
                "A model that cannot generalise with formulas will fail on larger assemblies.")
    reason_section = (
        f'<div class="dp-sec">'
        f'<div class="ds">Reasoning Quality</div>'
        f'<div class="dp-note{" ok-note" if r.has_loop_logic and r.reasoning_depth >= 0.5 else ""}">'
        f'Depth score: <b>{r.reasoning_depth:.0%}</b> | '
        f'Formula / loop logic: <b>{"YES ✓" if r.has_loop_logic else "NO"}</b> | '
        f'Response length: <b>{r.response_len:,}</b> chars.<br/>'
        f'{loop_txt}'
        f'</div>'
        f'</div>'
    )

    # ── Part Count ───────────────────────────────────────────────────────────
    if r.json_ok:
        delta = r.part_count - r.expected_parts
        direction = "extra" if delta > 0 else "missing"
        count_note_cls = "" if delta == 0 else " ok-note" if abs(delta) == 0 else ""
        count_section = (
            f'<div class="dp-sec">'
            f'<div class="ds">Part Count</div>'
            f'<div class="dp-note{count_note_cls}">'
            f'Generated: <b>{r.part_count}</b> shapes | Expected: <b>{r.expected_parts}</b> | '
            f'{"<b>Exact match ✓</b>" if delta == 0 else f"<b>{abs(delta)} {direction}</b> — extra shapes add clutter; missing shapes mean parts of the assembly are absent."}'
            f'</div>'
            f'</div>'
        )
    else:
        count_section = (
            '<div class="dp-sec">'
            '<div class="ds">Part Count</div>'
            '<div class="dp-note">No valid JSON was extracted from the LLM response — '
            f'0 shapes generated, {r.expected_parts} expected. '
            f'{r.failure or ""}'
            '</div>'
            '</div>'
        )

    # ── Interference Detection ─────────────────────────────────────────
    inter_section = ""
    if r.interference_details or r.interference_score < 1.0:
        if r.interference_score >= 1.0:
            inter_section = (
                '<div class="dp-sec">'
                '<div class="ds">🔩 Interference — No Unintended Overlaps ✓</div>'
                '<div class="dp-note ok-note">'
                '✓ No unintended volumetric interference detected between parts. '
                'Each part occupies its own space (intentional union/subtract pairs excluded).'
                '</div>'
                '</div>'
            )
        else:
            pairs = r.interference_details[:6]
            rows = "".join(f'<tr class="fail"><td class="mono">{_h.escape(str(d))}</td></tr>' for d in pairs)
            inter_section = (
                f'<div class="dp-sec">'
                f'<div class="ds">🔩 Interference — Overlapping Parts Detected (FAIL)</div>'
                f'<div class="dp-note">'
                f'⚠ Unintended volumetric overlaps found. In a real assembly, colliding parts would '
                f'be physically impossible to assemble or would cause structural failure at the overlap zone. '
                f'Score: <b>{r.interference_score:.0%}</b>'
                f'</div>'
                f'<table class="dpt"><tbody>{rows}</tbody></table>'
                f'</div>'
            )

    # ── Chamfer Distance ────────────────────────────────────────────────
    cd_section = ""
    if r.chamfer_distance >= 0:
        cd_col = "#3be8b0" if r.chamfer_score >= 0.8 else ("#fbbf24" if r.chamfer_score >= 0.4 else "#f472b6")
        cd_section = (
            f'<div class="dp-sec">'
            f'<div class="ds">📐 Chamfer Distance — Surface Accuracy</div>'
            f'<div class="dp-note{"  ok-note" if r.chamfer_score >= 0.8 else ""}">'
            f'Bidirectional surface-sampled Chamfer Distance: '
            f'<b style="color:{cd_col}">{r.chamfer_distance:.2f}mm</b> '
            f'→ score <b style="color:{cd_col}">{r.chamfer_score:.0%}</b>.<br/>'
            f'CD measures the mean nearest-neighbor distance between LLM and ground-truth surface point clouds. '
            f'Unlike centroid-point checks, CD detects size errors, wrong radii, and shape distortions '
            f'that position-accuracy alone misses. Score = exp(−CD/25).'
            f'</div>'
            f'</div>'
        )

    # ── Assembly Mates ─────────────────────────────────────────────────
    mate_section = ""
    if r.mate_details:
        rows = ""
        for d in r.mate_details:
            ok = isinstance(d, str) and "OK" in d
            cls = "ok" if ok else "fail"
            rows += f'<tr class="{cls}"><td class="mono">{_h.escape(str(d))}</td></tr>'
        mate_ok = r.mate_score >= 0.8
        mate_section = (
            f'<div class="dp-sec">'
            f'<div class="ds">🔗 Assembly Mates — {r.mate_score:.0%}</div>'
            f'<div class="dp-note{"  ok-note" if mate_ok else ""}">'
            f'{"✓ All mate constraints satisfied." if mate_ok else "⚠ Some mate constraints violated."} '
            f'Mates verify geometric relationships: concentric (shared axis), coincident (coplanar faces), '
            f'tangent (touching surfaces), distance (specific gap). Unsatisfied mates produce '
            f'misaligned joints, loose shafts, or gaps in sealing surfaces.'
            f'</div>'
            f'<table class="dpt"><tbody>{rows}</tbody></table>'
            f'</div>'
        )

    # ── DFM — Clearance Fit ────────────────────────────────────────────
    clearance_section = ""
    if r.clearance_details:
        rows = ""
        for d in r.clearance_details:
            ok = isinstance(d, str) and "OK" in d
            cls = "ok" if ok else "fail"
            rows += f'<tr class="{cls}"><td class="mono">{_h.escape(str(d))}</td></tr>'
        clr_ok = r.clearance_score >= 0.9
        clearance_section = (
            f'<div class="dp-sec">'
            f'<div class="ds">⚙️ DFM: Clearance Fit — {r.clearance_score:.0%}</div>'
            f'<div class="dp-note{"  ok-note" if clr_ok else ""}">'
            f'{"✓ Clearance fits within tolerance." if clr_ok else "⚠ Clearance fit error."} '
            f'Clearance = bore_radius − shaft_radius. Too little → interference (shaft cannot enter); '
            f'too much → excessive play and vibration. H7/g6 running fit target: 0.1–0.3mm per side.'
            f'</div>'
            f'<table class="dpt"><tbody>{rows}</tbody></table>'
            f'</div>'
        )

    # ── DFM — Wall Thickness ───────────────────────────────────────────
    wall_section = ""
    if r.wall_details:
        rows = ""
        for d in r.wall_details:
            ok = isinstance(d, str) and "OK" in d
            cls = "ok" if ok else "fail"
            rows += f'<tr class="{cls}"><td class="mono">{_h.escape(str(d))}</td></tr>'
        wall_section = (
            f'<div class="dp-sec">'
            f'<div class="ds">🧱 DFM: Wall Thickness — {"OK ✓" if r.wall_ok else "FAIL"}</div>'
            f'<div class="dp-note{"  ok-note" if r.wall_ok else ""}">'
            f'{"✓ Minimum wall thickness satisfied." if r.wall_ok else "⚠ Wall too thin — part may crack during manufacture or use."} '
            f'Thin walls (< min_wall_mm) are a critical DFM (Design for Manufacture) failure: '
            f'injection-moulded or 3D-printed parts with walls under ~1.5mm warp or break.'
            f'</div>'
            f'<table class="dpt"><tbody>{rows}</tbody></table>'
            f'</div>'
        )

    # ── Symmetry ───────────────────────────────────────────────────────
    sym_section = ""
    if r.symmetry_details:
        rows = ""
        for d in r.symmetry_details:
            ok = isinstance(d, str) and "OK" in d
            cls = "ok" if ok else "fail"
            rows += f'<tr class="{cls}"><td class="mono">{_h.escape(str(d))}</td></tr>'
        sym_ok = r.symmetry_score >= 0.9
        sym_section = (
            f'<div class="dp-sec">'
            f'<div class="ds">🪞 Symmetry Validation — {r.symmetry_score:.0%}</div>'
            f'<div class="dp-note{"  ok-note" if sym_ok else ""}">'
            f'{"✓ Assembly is correctly symmetric." if sym_ok else "⚠ Symmetry violated — mirrored shapes missing or misplaced."} '
            f'Mirror-plane symmetry is a fundamental design constraint for brackets, enclosures, and '
            f'balanced structures. A symmetric design distributes loads evenly and is easier to manufacture.'
            f'</div>'
            f'<table class="dpt"><tbody>{rows}</tbody></table>'
            f'</div>'
        )

    # ── Pattern ────────────────────────────────────────────────────────
    pat_section = ""
    if r.pattern_details:
        rows = ""
        for d in r.pattern_details:
            ok = isinstance(d, str) and "OK" in d
            cls = "ok" if ok else "fail"
            rows += f'<tr class="{cls}"><td class="mono">{_h.escape(str(d))}</td></tr>'
        pat_ok = r.pattern_score >= 0.9
        pat_section = (
            f'<div class="dp-sec">'
            f'<div class="ds">🔄 Pattern Validation — {r.pattern_score:.0%}</div>'
            f'<div class="dp-note{"  ok-note" if pat_ok else ""}">'
            f'{"✓ Pattern (circular/linear) correctly realised." if pat_ok else "⚠ Pattern incorrect — shapes not evenly distributed."} '
            f'Circular and linear feature patterns are ubiquitous in mechanical design (bolt circles, '
            f'cooling fins, gear teeth spacing). Even angular spacing requires accurate sin/cos derivation.'
            f'</div>'
            f'<table class="dpt"><tbody>{rows}</tbody></table>'
            f'</div>'
        )

    # ── Reasoning Chain ────────────────────────────────────────────────
    chain_section = ""
    if r.reasoning_chain_details:
        rows = ""
        for d in r.reasoning_chain_details:
            ok = isinstance(d, str) and "FOUND" in d
            cls = "ok" if ok else "fail"
            rows += f'<tr class="{cls}"><td class="mono">{_h.escape(str(d))}</td></tr>'
        chain_ok = r.reasoning_chain_score >= 0.8
        chain_section = (
            f'<div class="dp-sec">'
            f'<div class="ds">🧮 Reasoning Chain — {r.reasoning_chain_score:.0%} intermediate values found</div>'
            f'<div class="dp-note{"  ok-note" if chain_ok else ""}">'
            f'{"✓ Key intermediate values computed correctly." if chain_ok else "⚠ Missing intermediate calculations — LLM may have guessed final values without derivation."} '
            f'A sound derivation chain shows each formula step (pin_x, clearance, wall_thickness, etc.) '
            f'explicitly. Skipping steps and outputting final coordinates directly is a hallmark of '
            f'pattern-matching rather than genuine geometric reasoning.'
            f'</div>'
            f'<table class="dpt"><tbody>{rows}</tbody></table>'
            f'</div>'
        )

    # ── JSON Comparison ──────────────────────────────────────────────────────
    gt_raw  = r.ground_truth_json or ""
    llm_raw = r.llm_json or ""
    gt_lines  = gt_raw.split("\n")
    llm_lines = llm_raw.split("\n")

    MAXLINES = 70
    gt_display  = "\n".join(gt_lines[:MAXLINES])
    if len(gt_lines)  > MAXLINES: gt_display  += f"\n… (+{len(gt_lines)-MAXLINES} more lines)"
    llm_display = "\n".join(llm_lines[:MAXLINES]) if llm_raw else "(no JSON extracted)"
    if len(llm_lines) > MAXLINES: llm_display += f"\n… (+{len(llm_lines)-MAXLINES} more lines)"

    json_section = (
        f'<div class="dp-sec">'
        f'<div class="ds">JSON Comparison — Ground Truth vs LLM Output</div>'
        f'<div class="json-cmp">'
        f'<div class="jc-half">'
        f'<div class="jc-label">🟢 Ground Truth ({len(gt_lines)} lines, {r.expected_parts} shapes)</div>'
        f'<pre class="jc-pre">{_h.escape(gt_display)}</pre>'
        f'</div>'
        f'<div class="jc-half">'
        f'<div class="jc-label">{"🔵" if r.json_ok else "🔴"} LLM Output ({len(llm_lines)} lines, {r.part_count} shapes)</div>'
        f'<pre class="jc-pre">{_h.escape(llm_display)}</pre>'
        f'</div>'
        f'</div>'
        f'</div>'
    )

    return (
        f'<div class="dp" id="{uid}">'
        f'{score_section}'
        f'{count_section}'
        f'{pos_section}'
        f'{orient_section}'
        f'{conn_section}'
        f'{grav_section}'
        f'{inter_section}'
        f'{cd_section}'
        f'{mate_section}'
        f'{clearance_section}'
        f'{wall_section}'
        f'{sym_section}'
        f'{pat_section}'
        f'{chain_section}'
        f'{reason_section}'
        f'{json_section}'
        f'</div>'
    )


# ═══════════════════════════════════════════════════════════════
# MAIN REPORT BUILDER
# ═══════════════════════════════════════════════════════════════

def make_report(results: List[LevelResult], model: str, elapsed: float,
                mode: str = "fixed", master_seed: int = 0) -> str:

    n = len(results)
    avg = sum(r.score for r in results) / n if n else 0
    high_acc = sum(1 for r in results if r.position_accuracy >= 0.8)
    sound_count = sum(1 for r in results if r.connectivity_ok and r.gravity_ok)
    total_parts_generated = sum(r.part_count for r in results)
    total_parts_expected = sum(r.expected_parts for r in results)
    is_stress = (mode == "stress")

    max_clean = 0
    for r in results:
        if r.score >= 0.6: max_clean = r.wave if is_stress else r.level_id
        else: break

    det = next((r for r in results if r.score < 0.4), None)

    grades = [(0.85,"A+","#3be8b0"),(0.75,"A","#3be8b0"),(0.65,"B","#6c9eff"),
              (0.55,"C","#fbbf24"),(0.40,"D","#fb923c"),(0.0,"F","#f472b6")]
    grade, gcol = "F", "#f472b6"
    for thr, g, c in grades:
        if avg >= thr: grade, gcol = g, c; break

    # Waterfall SVG
    bw = max(24, min(60, 700 // max(n, 1)))
    bh = 200; gap = 6
    total_w = n * (bw + gap) + gap
    bar_svg = ""
    for i, r in enumerate(results):
        x = i * (bw + gap) + gap
        h_b = int(r.score * bh)
        y = bh - h_b + 30
        col = "#3be8b0" if r.score >= 0.7 else ("#fbbf24" if r.score >= 0.4 else "#f472b6")
        det_mark = ""
        if det and r.level_id == det.level_id and r.wave == det.wave:
            det_mark = (f'<line x1="{x}" y1="25" x2="{x+bw}" y2="25" '
                        f'stroke="#f472b6" stroke-width="2" stroke-dasharray="4"/>'
                        f'<text x="{x+bw//2}" y="18" text-anchor="middle" '
                        f'fill="#f472b6" font-size="8" font-weight="700">DETACH</text>')
        label = f"S{r.wave}" if is_stress else f"L{r.level_id}"
        n_lbl = str(r.n_parts) if is_stress and r.n_parts else ""
        acc_lbl = f"{r.position_accuracy:.0%}" if r.positions_checked > 0 else "—"
        bar_svg += f'''
        <rect x="{x}" y="{y}" width="{bw}" height="{h_b}" rx="3" fill="{col}" opacity="0.85"/>
        <text x="{x+bw//2}" y="{y-5}" text-anchor="middle" fill="{col}" font-size="10" font-weight="700">{r.score:.0%}</text>
        <text x="{x+bw//2}" y="{bh+44}" text-anchor="middle" fill="#6b7794" font-size="9">{label}</text>
        <text x="{x+bw//2}" y="{bh+54}" text-anchor="middle" fill="#4a5068" font-size="8">{n_lbl}</text>
        <text x="{x+bw//2}" y="{bh+64}" text-anchor="middle" fill="#6b7794" font-size="8">{acc_lbl}</text>
        {det_mark}'''

    chart_svg = (f'<svg viewBox="0 0 {total_w} {bh+80}" '
                 f'style="width:100%;max-width:{total_w}px;height:auto">'
                 f'{bar_svg}'
                 f'<line x1="{gap}" y1="{bh+30}" x2="{total_w-gap}" y2="{bh+30}" '
                 f'stroke="#252d44" stroke-width="1"/></svg>')

    # Level rows + detail panels
    rows = ""
    for r in results:
        bcol = "#3be8b0" if r.score >= 0.7 else ("#fbbf24" if r.score >= 0.4 else "#f472b6")
        st = "PASS" if r.score >= 0.7 else ("PARTIAL" if r.score >= 0.4 else "FAIL")
        stc = "sp" if r.score >= 0.7 else ("sw" if r.score >= 0.4 else "sf")
        pct = int(r.score * 100)
        det_tag = ""
        if det and r.level_id == det.level_id and r.wave == det.wave:
            det_tag = '<span class="dt">DETACHMENT</span>'
        label = f"S{r.wave}" if is_stress else f"L{r.level_id}"
        uid = f"dp-{'S' if is_stress else 'L'}{r.wave if is_stress else r.level_id}"

        parts_badge = (f'<span class="pb">{r.part_count}/{r.expected_parts} parts</span>'
                       if r.json_ok else '<span class="pb px">no JSON</span>')
        _pos_cls = "pg" if r.position_accuracy >= 0.8 else ("py" if r.position_accuracy >= 0.4 else "pr")
        pos_badge = (f'<span class="pb {_pos_cls}">pos {r.positions_correct}/{r.positions_checked}</span>'
                     if r.positions_checked > 0 else "")
        _ori_cls = "pg" if r.orientation_accuracy >= 0.8 else ("py" if r.orientation_accuracy >= 0.4 else "pr")
        orient_badge = (f'<span class="pb {_ori_cls}">orient {r.orientations_correct}/{r.orientations_checked}</span>'
                        if r.orientations_checked > 0 else "")
        loop_badge = ('<span class="pb pg">loop logic</span>'
                      if r.has_loop_logic else '<span class="pb px">no loop logic</span>')
        conn_badge = '<span class="pb pg">connected</span>' if r.connectivity_ok else '<span class="pb pr">not connected</span>'
        grav_badge = '<span class="pb pg">supported</span>' if r.gravity_ok else '<span class="pb pr">floating</span>'
        # Engineering badges
        inter_badge = ""
        if r.interference_score > 0 or r.interference_details:
            inter_badge = ('<span class="pb pg">no interference</span>' if r.interference_score >= 1.0
                          else '<span class="pb pr">interference!</span>')
        cd_badge = ""
        if r.chamfer_distance >= 0:
            cd_col = "pg" if r.chamfer_score >= 0.8 else ("py" if r.chamfer_score >= 0.4 else "pr")
            cd_badge = f'<span class="pb {cd_col}">CD {r.chamfer_distance:.1f}mm</span>'
        mate_badge = ""
        if r.mate_details:
            mate_col = "pg" if r.mate_score >= 0.8 else ("py" if r.mate_score >= 0.5 else "pr")
            mate_badge = f'<span class="pb {mate_col}">mates {r.mate_score:.0%}</span>'
        eng_badge = ""
        if r.clearance_details or r.wall_details:
            dfm_ok = r.clearance_score >= 0.9 and r.wall_ok
            eng_badge = (f'<span class="pb pg">DFM OK</span>' if dfm_ok
                        else f'<span class="pb pr">DFM FAIL</span>')

        stress_info = (f'<div class="si">Generator: {r.generator_name} | '
                       f'{r.n_parts} parts | +/-{r.coord_range:.0f}mm | seed={r.seed}</div>'
                       if is_stress else "")
        fail_row = f'<div class="fr">{_h.escape(r.failure)}</div>' if r.failure else ""
        is_det = " dr" if (det and r.level_id == det.level_id and r.wave == det.wave) else ""

        rows += f'''
        <div class="lr{is_det}" onclick="toggleDet('{uid}')" role="button" tabindex="0">
          <div class="ln">{label}<span class="ti" id="ti-{uid}">▶</span></div>
          <div class="li">
            <div class="lt">{_h.escape(r.level_name)} {det_tag}</div>
            <div class="ls">{_h.escape(r.skill)}</div>
            <div class="bd">{parts_badge} {pos_badge} {orient_badge} {loop_badge} {conn_badge} {grav_badge} {inter_badge} {cd_badge} {mate_badge} {eng_badge}</div>
            {stress_info}
            {fail_row}
          </div>
          <div class="lv">
            <div class="bb"><div class="bf" style="width:{pct}%;background:{bcol}"></div></div>
            <span class="sn">{r.score:.0%}</span>
            <span class="{stc}">{st}</span>
            <span class="exp-hint">▼ details</span>
          </div>
        </div>
        {_detail_panel(r, uid)}'''

    # Detachment box
    det_html = ""
    if det:
        lbl = f"Wave {det.wave} ({det.n_parts} parts)" if is_stress else f"Level {det.level_id}"
        det_html = f'''
        <div class="db">
          <h3>Reasoning Detachment at {lbl}</h3>
          <p>LLM 3D generation logic breaks down at <b>{_h.escape(det.level_name)}</b>.<br/>
          Skill: <b>{_h.escape(det.skill)}</b>.
          {(" Max parts handled correctly: " + str(max((r.n_parts for r in results if r.position_accuracy >= 0.5), default=0))) if is_stress else ""}
          <br/>{_h.escape(det.failure) if det.failure else "Positions in generated geometry were incorrect."}</p>
        </div>'''

    # Interpretation
    if is_stress:
        max_correct_parts = max((r.n_parts for r in results if r.position_accuracy >= 0.5), default=0)
        max_wave_ok = max((r.wave for r in results if r.position_accuracy >= 0.5), default=0)
        interp = (
            f"This LLM can correctly generate assemblies up to approximately "
            f"<b>{max_correct_parts} parts</b> (wave {max_wave_ok}). "
            f"Beyond that, its spatial model breaks down."
        )
    else:
        if max_clean >= 12:
            interp = "This LLM demonstrates <b>excellent 3D generation logic</b> — correct chained positioning, orientation, engineering constraints, and multi-component assemblies."
        elif max_clean >= 9:
            interp = "This LLM demonstrates <b>strong 3D generation logic</b> — handles derived positions, gear math, and DFM awareness."
        elif max_clean >= 6:
            interp = "This LLM handles <b>intermediate</b> 3D logic — positions, parametric patterns, symmetry — but struggles with advanced engineering constraints."
        elif max_clean >= 3:
            interp = "This LLM handles <b>basic</b> 3D logic but struggles with multi-step spatial derivations."
        else:
            interp = "This LLM shows <b>weak 3D generation logic</b>. Not suitable for reliable 3D assembly generation."

    # Export note
    if is_stress:
        export_note = (f"Ground truth JSON files saved to <code>C2CAD-Bench/codes/</code> "
                       f"(pattern: <code>S&lt;wave&gt;_gt.json</code>). "
                       f"LLM outputs saved as <code>S&lt;wave&gt;_llm.json</code>. "
                       f"Both can be loaded into any compatible 3D engine.")
    else:
        export_note = (f"Ground truth and LLM output JSON files saved to <code>C2CAD-Bench/codes/</code>. "
                       f"Feed <code>L&lt;n&gt;_gt.json</code> or <code>L&lt;n&gt;_llm.json</code> "
                       f"into a compatible viewer for visual validation.")

    mode_title = "Stress Test — Encrypted Procedural" if is_stress else "Fixed Levels (L1–L14)"
    mode_sub = (f"Master Seed: <code>{master_seed}</code> | {' → '.join(GENERATOR_NAMES)}"
                if is_stress else "14 cognitive-skill 3D generation benchmarks")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>C2CAD-Bench v7.0 — {model}</title>
<style>
:root{{--bg:#07090e;--s1:#0e1118;--s2:#151a26;--bd:#252d44;--tx:#c8cfe0;--tm:#6b7794;--tb:#f0f2f8}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--tx);font:14px/1.6 'Segoe UI',system-ui,sans-serif;padding:28px}}
.w{{max-width:1020px;margin:0 auto}}
h1{{font-size:23px;color:var(--tb);margin-bottom:4px}}
.su{{color:var(--tm);font-size:12px;margin-bottom:4px}}
.ms{{color:var(--tm);font-size:11px;margin-bottom:22px}}
code{{background:var(--s2);padding:1px 5px;border-radius:3px;font-size:11px;color:#6c9eff}}

/* ── Summary grid ── */
.sg{{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:10px;margin-bottom:22px}}
.st{{background:var(--s1);border:1px solid var(--bd);border-radius:10px;padding:14px;text-align:center}}
.sv{{font-size:24px;font-weight:800;color:var(--tb)}}
.sl{{font-size:9px;color:var(--tm);text-transform:uppercase;letter-spacing:1px;margin-top:3px}}
.gb{{border:2px solid {gcol}}}.gb .sv{{color:{gcol}}}
.interp{{background:var(--s1);border:1px solid var(--bd);border-radius:10px;padding:13px 17px;margin-bottom:20px;font-size:13px;line-height:1.7}}
.ch{{background:var(--s1);border:1px solid var(--bd);border-radius:10px;padding:16px;margin-bottom:22px;overflow-x:auto}}
.ch h3{{font-size:11px;color:var(--tm);margin-bottom:10px;text-transform:uppercase;letter-spacing:1px}}

/* ── Detachment box ── */
.db{{background:rgba(244,114,182,.05);border:1px solid rgba(244,114,182,.25);border-left:3px solid #f472b6;border-radius:8px;padding:13px 17px;margin-bottom:20px}}
.db h3{{color:#f472b6;font-size:13px;margin-bottom:5px}}.db p{{font-size:12px;margin:0}}

/* ── Level row ── */
.lr{{display:flex;align-items:center;gap:11px;padding:11px 15px;background:var(--s1);border:1px solid var(--bd);border-radius:8px;margin-bottom:0;cursor:pointer;user-select:none;transition:border-color .15s}}
.lr:hover{{border-color:#6c9eff}}.dr{{border-color:#f472b6!important;background:rgba(244,114,182,.03)}}
.lr.expanded{{border-radius:8px 8px 0 0;border-color:#6c9eff;margin-bottom:0}}
.ln{{font:800 15px system-ui;color:var(--tm);min-width:44px;display:flex;align-items:center;gap:4px}}
.ti{{font-size:9px;color:var(--tm);display:inline-block;transition:transform .2s;line-height:1}}
.ti.open{{transform:rotate(90deg)}}
.li{{flex:1;min-width:0}}
.lt{{font-weight:700;color:var(--tb);font-size:13px}}.ls{{font-size:11px;color:var(--tm)}}
.bd{{margin-top:4px;display:flex;flex-wrap:wrap;gap:4px}}
.si{{font-size:10px;color:var(--tm);margin-top:3px;font-family:monospace}}
.pb{{font-size:9.5px;font-weight:700;padding:1px 6px;border-radius:3px}}
.pg{{color:#3be8b0;background:rgba(59,232,176,.1)}}
.py{{color:#fbbf24;background:rgba(251,191,36,.1)}}
.pr{{color:#f472b6;background:rgba(244,114,182,.1)}}
.px{{color:#6b7794;background:rgba(107,119,148,.1)}}
.fr{{font-size:10px;color:#f472b6;margin-top:3px}}
.lv{{display:flex;align-items:center;gap:7px;min-width:180px}}
.bb{{width:55px;height:6px;background:var(--s2);border-radius:4px;overflow:hidden}}.bf{{height:100%;border-radius:4px}}
.sn{{font:700 12px system-ui;color:var(--tb);min-width:32px;text-align:right}}
.sp{{font-size:9px;font-weight:700;color:#3be8b0;background:rgba(59,232,176,.1);padding:2px 5px;border-radius:4px}}
.sw{{font-size:9px;font-weight:700;color:#fbbf24;background:rgba(251,191,36,.1);padding:2px 5px;border-radius:4px}}
.sf{{font-size:9px;font-weight:700;color:#f472b6;background:rgba(244,114,182,.1);padding:2px 5px;border-radius:4px}}
.dt{{font-size:8px;font-weight:700;color:#f472b6;background:rgba(244,114,182,.12);border:1px solid rgba(244,114,182,.35);padding:1px 5px;border-radius:10px;margin-left:4px}}
.exp-hint{{font-size:9px;color:var(--tm);white-space:nowrap}}

/* ── Detail panel ── */
.dp{{display:none;background:var(--s2);border:1px solid #6c9eff;border-top:none;border-radius:0 0 8px 8px;padding:16px 18px;margin-bottom:8px}}
.dp.open{{display:block}}
.dp-sec{{margin-bottom:16px}}
.dp-sec:last-child{{margin-bottom:0}}
.ds{{font-size:10px;font-weight:700;color:var(--tm);text-transform:uppercase;letter-spacing:1px;margin-bottom:7px;border-bottom:1px solid var(--bd);padding-bottom:4px}}
.ds-hint{{font-size:9px;font-weight:400;text-transform:none;letter-spacing:0;color:#4a5068}}
.dp-note{{font-size:11.5px;color:var(--tx);line-height:1.65;background:rgba(244,114,182,.06);border-left:3px solid #f472b6;padding:7px 10px;border-radius:0 5px 5px 0;margin-bottom:6px}}
.dp-note.ok-note{{background:rgba(59,232,176,.06);border-left-color:#3be8b0;color:#3be8b0}}

/* ── Score grid ── */
.sc-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(90px,1fr));gap:6px;margin-bottom:4px}}
.sci{{background:var(--s1);border:1px solid var(--bd);border-radius:6px;padding:7px 5px;text-align:center;cursor:help}}
.sci:hover{{border-color:#6c9eff}}
.scv{{font-size:15px;font-weight:800;margin-bottom:2px}}
.scn{{font-size:9px;color:var(--tm);margin-bottom:2px}}
.scw{{font-size:9px;color:var(--tm)}}

/* ── Detail table ── */
.dpt{{width:100%;border-collapse:collapse;font-size:11px;margin-bottom:4px}}
.dpt td{{padding:2px 5px}}
.dpt tr.ok td{{color:#3be8b0}}
.dpt tr.fail td{{color:#f472b6}}
.mono{{font-family:monospace}}

/* ── JSON comparison ── */
.json-cmp{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
.jc-half{{min-width:0}}
.jc-label{{font-size:10px;color:var(--tm);margin-bottom:5px;font-weight:600}}
.jc-pre{{background:var(--bg);border:1px solid var(--bd);border-radius:4px;padding:9px;font-family:monospace;font-size:9.5px;overflow:auto;max-height:240px;color:var(--tx);white-space:pre;tab-size:2;line-height:1.4}}

/* ── Misc ── */
.note{{background:var(--s1);border:1px solid var(--bd);border-radius:10px;padding:13px 17px;margin-top:20px;font-size:11.5px;color:var(--tm);line-height:1.6}}
.note b{{color:var(--tx)}}
footer{{text-align:center;color:var(--tm);font-size:10px;margin-top:26px;padding-top:13px;border-top:1px solid var(--bd)}}

@media(max-width:720px){{
  .sg{{grid-template-columns:repeat(2,1fr)}}
  .sc-grid{{grid-template-columns:repeat(4,1fr)}}
  .json-cmp{{grid-template-columns:1fr}}
  .exp-hint{{display:none}}
}}
</style>
</head><body>
<div class="w">
<h1>C2CAD-Bench v7.0 — {mode_title}</h1>
<p class="su">Model: <b>{model}</b> | {datetime.now().strftime('%Y-%m-%d %H:%M')} | {n} tests | {elapsed:.0f}s</p>
<p class="ms">{mode_sub}</p>
<div class="sg">
  <div class="st gb"><div class="sv">{grade}</div><div class="sl">Grade</div></div>
  <div class="st"><div class="sv">{avg:.0%}</div><div class="sl">Avg Score</div></div>
  <div class="st"><div class="sv">{high_acc}/{n}</div><div class="sl">High Accuracy</div></div>
  <div class="st"><div class="sv">{sound_count}/{n}</div><div class="sl">Structurally Sound</div></div>
  <div class="st"><div class="sv">{total_parts_generated}</div><div class="sl">Parts Output</div></div>
  <div class="st"><div class="sv">{total_parts_expected}</div><div class="sl">Parts Expected</div></div>
</div>
<div class="interp">{interp}</div>
<div class="ch">
  <h3>Score waterfall</h3>
  {chart_svg}
</div>
{det_html}
<p style="font-size:11px;color:var(--tm);margin-bottom:10px;">Click any row to expand full validation details ↓</p>
{rows}
<div class="note">
<b>How scoring works (v7.0):</b> The LLM receives a 3D assembly specification and must output a
universal JSON array of shape objects. C2CAD-Bench computes the exact ground truth internally
and compares the LLM's JSON against it. 8 scoring dimensions are evaluated.<br/>
<b>Core mode:</b> 35% geometric accuracy | 12% part count | 23% spatial reasoning |
10% surface fidelity | 5% orientation | 10% reasoning | 5% interference.<br/>
<b>Engineering mode:</b> 30% geometric | 10% count | 15% spatial | 10% surface |
5% orientation | 10% reasoning | 15% engineering constraints | 5% patterns.<br/><br/>
<b>Human validation:</b> {export_note}
</div>
<footer>C2CAD-Bench v7.0 | Anonymous submission artifact | {datetime.now().year}</footer>
</div>

<script>
function toggleDet(uid) {{
  var dp = document.getElementById(uid);
  var ti = document.getElementById('ti-' + uid);
  var row = dp ? dp.previousElementSibling : null;
  if (!dp) return;
  var isOpen = dp.classList.contains('open');
  dp.classList.toggle('open', !isOpen);
  if (ti) ti.classList.toggle('open', !isOpen);
  if (row) row.classList.toggle('expanded', !isOpen);
}}
// keyboard accessibility
document.addEventListener('keydown', function(e) {{
  if (e.key === 'Enter' || e.key === ' ') {{
    var el = document.activeElement;
    if (el && el.classList.contains('lr')) {{
      e.preventDefault();
      var uid = el.getAttribute('onclick').match(/'([^']+)'/)[1];
      toggleDet(uid);
    }}
  }}
}});
</script>
</body></html>"""


# ═══════════════════════════════════════════════════════════════
# SAVE & CONSOLE SUMMARY
# ═══════════════════════════════════════════════════════════════

def save_and_summarize(results, elapsed, model, mode, master_seed=0):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = "stress" if mode == "stress" else "fixed"

    html = make_report(results, model, elapsed, mode, master_seed)
    hp = CFG.RESULTS_DIR / f"cg3d_{model}_{tag}_{ts}.html"
    hp.write_text(html, encoding='utf-8')

    jp = CFG.RESULTS_DIR / f"cg3d_{model}_{tag}_{ts}.json"
    jp.write_text(json.dumps({
        "model": model, "version": "10.0", "mode": mode,
        "master_seed": master_seed,
        "timestamp": ts, "elapsed": elapsed,
        "tests": len(results),
        "high_accuracy": sum(1 for r in results if r.position_accuracy >= 0.8),
        "structurally_sound": sum(1 for r in results if r.connectivity_ok and r.gravity_ok),
        "average_score": round(sum(r.score for r in results)/len(results), 3) if results else 0,
        "total_parts_expected": sum(r.expected_parts for r in results),
        "total_parts_generated": sum(r.part_count for r in results),
        "results": [asdict(r) for r in results]
    }, indent=2, default=str), encoding='utf-8')

    avg = sum(r.score for r in results)/len(results) if results else 0
    det = next((r for r in results if r.score < 0.4), None)

    print("=" * 58)
    print(f"  RESULTS — {mode.upper()} MODE")
    print("=" * 58)
    print(f"  Average Score:       {avg:.0%}")
    print(f"  Structurally Sound:  {sum(1 for r in results if r.connectivity_ok and r.gravity_ok)}/{len(results)}")
    print(f"  Parts generated:     {sum(r.part_count for r in results)} "
          f"/ {sum(r.expected_parts for r in results)} expected")
    if mode == "stress":
        ok_waves = [r.wave for r in results if r.position_accuracy >= 0.5]
        print(f"  Last OK wave:        {max(ok_waves, default=0)}")
        print(f"  Max parts OK:        {max((r.n_parts for r in results if r.position_accuracy >= 0.5), default=0)}")
        print(f"  Seed:                {master_seed}")
    if det:
        lbl = f"Wave {det.wave}" if mode == "stress" else f"L{det.level_id}"
        print(f"  Detachment:          {lbl} — {det.level_name}")
    print(f"\n  HTML: {hp}")
    print(f"  JSON: {jp}")
    print(f"  Codes/GT: {CFG.CODES_DIR}")
    print("=" * 58)
    return hp
