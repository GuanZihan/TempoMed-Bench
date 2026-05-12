import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

def build_payload(summary):
    return json.dumps(
        {
            "overall": summary["overall"],
            "category_summaries": summary["category_summaries"],
            "records": [
                {
                    "category": record["category"],
                    "category_label": record["category_label"],
                    "color": record["color"],
                    "pmc_id": record["pmc_id"],
                    "pmid": record["pmid"],
                    "topic": record["topic"],
                    "title": record["title"],
                    "year": record["year"],
                    "organizations": record["organizations"][:12],
                    "prior_count": record["prior_count"],
                    "node_count": record["node_count"],
                    "path": record["path"],
                    "prior_guidelines": [
                        {
                            "year": prior["year"],
                            "pmid": prior["pmid"],
                            "organization": prior["organization"],
                            "title": prior["title"],
                        }
                        for prior in sorted(record["prior_guidelines"], key=lambda item: (item["year"] is None, item["year"], item["title"]))
                    ],
                }
                for record in summary["records"]
            ],
        },
        ensure_ascii=False,
    )


def render_html(summary):
    payload = build_payload(summary)
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Clean Guideline Trajectories</title>
  <style>
    :root {{
      --bg: #f5efe6;
      --panel: rgba(255, 252, 247, 0.93);
      --ink: #1d1b18;
      --muted: #6a635c;
      --line: rgba(61, 52, 43, 0.12);
      --shadow: 0 18px 42px rgba(71, 54, 35, 0.12);
      --radius: 22px;
      --serif: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      --sans: "Avenir Next", "Segoe UI", Helvetica, Arial, sans-serif;
      --mono: "IBM Plex Mono", Consolas, monospace;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: var(--sans);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.18), transparent 30%),
        radial-gradient(circle at top right, rgba(124,58,237,0.10), transparent 26%),
        linear-gradient(180deg, #faf5ed 0%, var(--bg) 50%, #eee6d9 100%);
    }}
    .page {{ max-width: 1580px; margin: 0 auto; padding: 26px; }}
    .hero, .metric, .card {{
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.78);
      box-shadow: var(--shadow);
      border-radius: var(--radius);
    }}
    .hero {{ padding: 32px; margin-bottom: 22px; }}
    h1 {{ margin: 0 0 10px; font-family: var(--serif); font-size: clamp(2.1rem, 4vw, 4.2rem); line-height: 0.98; letter-spacing: -0.04em; max-width: 11ch; }}
    .lede {{ max-width: 90ch; color: var(--muted); line-height: 1.65; margin: 0 0 18px; }}
    .metric-grid, .main-grid, .controls {{ display: grid; gap: 16px; }}
    .metric-grid {{ grid-template-columns: repeat(4, minmax(0, 1fr)); margin-bottom: 22px; }}
    .main-grid {{ grid-template-columns: minmax(390px, 0.9fr) minmax(520px, 1.3fr); }}
    .metric {{ padding: 20px; }}
    .metric .label {{ color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.75rem; font-weight: 700; }}
    .metric .value {{ font-family: var(--serif); font-size: clamp(2rem, 3vw, 3.1rem); line-height: 1; margin: 10px 0 6px; letter-spacing: -0.05em; }}
    .metric .sub {{ color: var(--muted); font-size: 0.95rem; }}
    .card {{ padding: 20px; }}
    .section-title {{ margin: 0 0 8px; font-size: 1.06rem; }}
    .section-copy {{ margin: 0 0 16px; color: var(--muted); line-height: 1.55; }}
    .pill-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }}
    .pill {{ display: inline-flex; gap: 8px; align-items: center; border-radius: 999px; padding: 9px 12px; border: 1px solid var(--line); background: rgba(255,255,255,0.82); cursor: pointer; user-select: none; }}
    .pill.active {{ background: rgba(29,27,24,0.93); color: white; border-color: transparent; }}
    .dot {{ width: 10px; height: 10px; border-radius: 999px; }}
    .controls {{ grid-template-columns: repeat(2, minmax(0, 1fr)); margin-bottom: 12px; }}
    .control {{ display: flex; flex-direction: column; gap: 6px; }}
    .control label {{ color: var(--muted); text-transform: uppercase; letter-spacing: 0.07em; font-size: 0.8rem; font-weight: 700; }}
    input, select {{ border: 1px solid var(--line); border-radius: 14px; padding: 12px 14px; font: inherit; background: rgba(255,255,255,0.9); }}
    .small-note {{ color: var(--muted); font-size: 0.9rem; line-height: 1.5; }}
    .list-panel {{ max-height: 1040px; overflow: auto; padding-right: 4px; }}
    .record-card {{ padding: 16px; border-radius: 18px; border: 1px solid rgba(0,0,0,0.06); background: rgba(255,255,255,0.8); margin-bottom: 12px; cursor: pointer; }}
    .record-card.active {{ border-color: rgba(15,118,110,0.35); box-shadow: 0 18px 34px rgba(15,118,110,0.10); }}
    .eyebrow {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; color: var(--muted); font-size: 0.82rem; margin-bottom: 8px; }}
    .record-title {{ margin: 0 0 8px; font-size: 1.02rem; line-height: 1.35; font-weight: 700; }}
    .record-meta {{ color: var(--muted); font-size: 0.92rem; line-height: 1.5; }}
    .detail-title {{ margin: 0 0 8px; font-family: var(--serif); font-size: clamp(1.55rem, 2vw, 2.5rem); line-height: 1.05; letter-spacing: -0.04em; }}
    .detail-topic {{ margin: 0 0 18px; color: var(--muted); line-height: 1.55; }}
    .badge {{ display: inline-block; padding: 7px 10px; border-radius: 999px; background: rgba(15,118,110,0.10); margin: 0 8px 8px 0; font-size: 0.82rem; font-weight: 700; }}
    .detail-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-bottom: 18px; }}
    .mini-card {{ border-radius: 18px; padding: 16px; background: rgba(255,255,255,0.78); border: 1px solid rgba(0,0,0,0.05); }}
    .mini-card h3 {{ margin: 0 0 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.07em; font-size: 0.82rem; }}
    .mini-card p {{ margin: 0; line-height: 1.6; }}
    .timeline {{ display: grid; gap: 12px; }}
    .timeline-item {{ padding: 14px 14px 14px 20px; border-left: 3px solid rgba(15,118,110,0.24); border-radius: 0 18px 18px 0; background: rgba(255,255,255,0.8); }}
    .timeline-item.current {{ border-left-color: rgba(15,118,110,0.8); background: linear-gradient(90deg, rgba(15,118,110,0.10), rgba(255,255,255,0.85)); }}
    .timeline-year {{ color: var(--muted); font-family: var(--mono); font-size: 0.84rem; margin-bottom: 6px; }}
    .timeline-title {{ font-weight: 700; line-height: 1.4; margin-bottom: 6px; }}
    .timeline-meta {{ color: var(--muted); font-size: 0.92rem; line-height: 1.5; }}
    code {{ font-family: var(--mono); font-size: 0.92em; background: rgba(0,0,0,0.05); padding: 0.14em 0.34em; border-radius: 0.38em; }}
    .empty-state {{ padding: 24px; text-align: center; color: var(--muted); border: 1px dashed rgba(0,0,0,0.12); border-radius: 18px; background: rgba(255,255,255,0.74); }}
    @media (max-width: 1180px) {{
      .metric-grid, .main-grid, .controls, .detail-grid {{ grid-template-columns: 1fr; }}
      .page {{ padding: 18px; }}
      .list-panel {{ max-height: none; }}
    }}
  </style>
</head>
<body>
  <div class=\"page\">
    <section class=\"hero\">
      <h1>Clean Guideline Trajectories</h1>
      <p class=\"lede\">Only trajectories that pass the strict filter are shown here: at least two nodes, no incompleteness issues, no chronology issues, and no within-trajectory redundancy. All excluded trajectories are omitted from this browser.</p>
      <p class=\"small-note\"><strong>Filter definition:</strong> {summary['overall']['definition']}</p>
    </section>
    <section class=\"metric-grid\" id=\"metricGrid\"></section>
    <section class=\"main-grid\">
      <div class=\"card\">
        <h2 class=\"section-title\">Trajectory Explorer</h2>
        <p class=\"section-copy\">Search and browse only the clean trajectories.</p>
        <div class=\"pill-row\" id=\"categoryPills\"></div>
        <div class=\"controls\">
          <div class=\"control\"><label for=\"searchInput\">Search</label><input id=\"searchInput\" type=\"text\" placeholder=\"Topic, title, PMID, PMC\"></div>
          <div class=\"control\"><label for=\"yearSelect\">Current Year</label><select id=\"yearSelect\"><option value=\"all\">All years</option></select></div>
          <div class=\"control\"><label for=\"depthSelect\">Prior Depth</label><select id=\"depthSelect\"><option value=\"all\">All clean trajectories</option><option value=\"1\">1 prior guideline</option><option value=\"2plus\">At least 2 prior guidelines</option><option value=\"5plus\">At least 5 prior guidelines</option></select></div>
          <div class=\"control\"><label for=\"sortSelect\">Sort</label><select id=\"sortSelect\"><option value=\"depth_desc\">Longest trajectory first</option><option value=\"year_desc\">Newest current guidance</option><option value=\"year_asc\">Oldest current guidance</option><option value=\"title_asc\">Title A-Z</option></select></div>
        </div>
        <div class=\"small-note\" id=\"resultCount\"></div>
        <div class=\"list-panel\" id=\"recordList\"></div>
      </div>
      <div class=\"card\"><div id=\"detailPanel\"></div></div>
    </section>
  </div>
  <script>
    const DATA = {payload};
    const byId = (id) => document.getElementById(id);
    const formatNumber = (value) => new Intl.NumberFormat('en-US').format(value);
    const percent = (value, total) => total ? `${{((value / total) * 100).toFixed(1)}}%` : '0.0%';
    const escapeHtml = (text) => String(text ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\"/g, '&quot;').replace(/'/g, '&#39;');
    const state = {{
      selectedCategories: new Set(DATA.category_summaries.map((item) => item.category)),
      search: '',
      year: 'all',
      depth: 'all',
      sort: 'depth_desc',
      selectedRecordId: null,
    }};

    function buildMetrics() {{
      const total = DATA.overall.total_clean_trajectories;
      const metrics = [
        {{ label: 'Clean Trajectories', value: formatNumber(total), sub: 'All records shown in this page pass the strict filter' }},
        {{ label: 'Commercial', value: formatNumber((DATA.category_summaries.find((item) => item.category === 'comm') || {{total_records: 0}}).total_records), sub: percent((DATA.category_summaries.find((item) => item.category === 'comm') || {{total_records: 0}}).total_records, total) }},
        {{ label: 'Non-commercial', value: formatNumber((DATA.category_summaries.find((item) => item.category === 'noncomm') || {{total_records: 0}}).total_records), sub: percent((DATA.category_summaries.find((item) => item.category === 'noncomm') || {{total_records: 0}}).total_records, total) }},
        {{ label: 'Other', value: formatNumber((DATA.category_summaries.find((item) => item.category === 'other') || {{total_records: 0}}).total_records), sub: percent((DATA.category_summaries.find((item) => item.category === 'other') || {{total_records: 0}}).total_records, total) }},
      ];
      byId('metricGrid').innerHTML = metrics.map((metric) => `<div class=\"metric\"><div class=\"label\">${{metric.label}}</div><div class=\"value\">${{metric.value}}</div><div class=\"sub\">${{metric.sub}}</div></div>`).join('');
    }}

    function buildPills() {{
      byId('categoryPills').innerHTML = DATA.category_summaries.map((item) => `<div class=\"pill active\" data-category=\"${{item.category}}\"><span class=\"dot\" style=\"background:${{item.color}}\"></span><span>${{item.label}}</span></div>`).join('');
      byId('categoryPills').querySelectorAll('.pill').forEach((pill) => {{
        pill.addEventListener('click', () => {{
          const category = pill.dataset.category;
          if (state.selectedCategories.has(category)) state.selectedCategories.delete(category); else state.selectedCategories.add(category);
          pill.classList.toggle('active', state.selectedCategories.has(category));
          renderList();
        }});
      }});
    }}

    function buildYearOptions() {{
      const years = [...new Set(DATA.records.map((record) => record.year).filter((year) => Number.isInteger(year) && year > 0))].sort((a, b) => b - a);
      byId('yearSelect').innerHTML += years.map((year) => `<option value=\"${{year}}\">${{year}}</option>`).join('');
    }}

    function filtered() {{
      let items = DATA.records.filter((record) => state.selectedCategories.has(record.category));
      if (state.year !== 'all') items = items.filter((record) => String(record.year) === state.year);
      if (state.depth === '1') items = items.filter((record) => record.prior_count === 1);
      else if (state.depth === '2plus') items = items.filter((record) => record.prior_count >= 2);
      else if (state.depth === '5plus') items = items.filter((record) => record.prior_count >= 5);
      if (state.search.trim()) {{
        const q = state.search.trim().toLowerCase();
        items = items.filter((record) => [record.topic, record.title, String(record.pmid ?? ''), record.pmc_id].join(' ').toLowerCase().includes(q));
      }}
      items.sort((a, b) => {{
        if (state.sort === 'depth_desc') return (b.prior_count - a.prior_count) || ((b.year ?? -9999) - (a.year ?? -9999)) || a.title.localeCompare(b.title);
        if (state.sort === 'year_desc') return ((b.year ?? -9999) - (a.year ?? -9999)) || (b.prior_count - a.prior_count);
        if (state.sort === 'year_asc') return ((a.year ?? 9999) - (b.year ?? 9999)) || (b.prior_count - a.prior_count);
        return a.title.localeCompare(b.title);
      }});
      return items;
    }}

    function timeline(record) {{
      const blocks = record.prior_guidelines.map((prior) => `<div class=\"timeline-item\"><div class=\"timeline-year\">${{prior.year ?? 'Unknown year'}}${{prior.pmid ? ` · PMID ${{prior.pmid}}` : ''}}</div><div class=\"timeline-title\">${{escapeHtml(prior.title)}}</div><div class=\"timeline-meta\">${{escapeHtml(prior.organization || 'Unknown organization')}}</div></div>`);
      blocks.push(`<div class=\"timeline-item current\"><div class=\"timeline-year\">${{record.year ?? 'Unknown year'}}${{record.pmid ? ` · PMID ${{record.pmid}}` : ''}}</div><div class=\"timeline-title\">${{escapeHtml(record.title)}}</div><div class=\"timeline-meta\">${{escapeHtml(record.organizations.join('; '))}}</div></div>`);
      return blocks.join('');
    }}

    function renderDetail(record) {{
      if (!record) {{
        byId('detailPanel').innerHTML = `<div class=\"empty-state\">Select a clean trajectory to inspect it.</div>`;
        return;
      }}
      byId('detailPanel').innerHTML = `
        <div class=\"eyebrow\"><span class=\"dot\" style=\"background:${{record.color}}\"></span><span>${{record.category_label}}</span><span>·</span><span>PMC ${{record.pmc_id}}</span></div>
        <h2 class=\"detail-title\">${{escapeHtml(record.title)}}</h2>
        <p class=\"detail-topic\">${{escapeHtml(record.topic)}}</p>
        <span class=\"badge\">${{record.year}}</span>
        <span class=\"badge\">${{record.node_count}} nodes</span>
        <span class=\"badge\">${{record.prior_count}} prior guidelines</span>
        <div class=\"detail-grid\">
          <div class=\"mini-card\"><h3>Current Guideline</h3><p><strong>PMID:</strong> ${{record.pmid ?? 'Unknown'}}<br><strong>Organizations:</strong> ${{escapeHtml(record.organizations.join('; '))}}<br><strong>Source:</strong> <code>${{escapeHtml(record.path)}}</code></p></div>
          <div class=\"mini-card\"><h3>Quality Status</h3><p>This trajectory passed the strict clean filter and has no recorded incompleteness, chronology, or within-trajectory redundancy issue.</p></div>
        </div>
        <h2 class=\"section-title\">Trajectory</h2>
        <p class=\"section-copy\">Prior guidelines are shown in chronological order, followed by the current guideline.</p>
        <div class=\"timeline\">${{timeline(record)}}</div>
      `;
    }}

    function renderList() {{
      const items = filtered();
      byId('resultCount').textContent = `${{formatNumber(items.length)}} clean trajectories match the current filters`;
      if (!items.length) {{
        byId('recordList').innerHTML = `<div class=\"empty-state\">No clean trajectories match the current filters.</div>`;
        byId('detailPanel').innerHTML = `<div class=\"empty-state\">Select a clean trajectory to inspect it.</div>`;
        return;
      }}
      if (!state.selectedRecordId || !items.some((record) => record.pmc_id === state.selectedRecordId)) state.selectedRecordId = items[0].pmc_id;
      byId('recordList').innerHTML = items.map((record) => `
        <article class=\"record-card ${{record.pmc_id === state.selectedRecordId ? 'active' : ''}}\" data-pmc=\"${{record.pmc_id}}\">
          <div class=\"eyebrow\"><span class=\"dot\" style=\"background:${{record.color}}\"></span><span>${{record.category_label}}</span><span>·</span><span>${{record.year}}</span><span>·</span><span>${{record.node_count}} nodes</span></div>
          <h3 class=\"record-title\">${{escapeHtml(record.title)}}</h3>
          <div class=\"record-meta\"><div><strong>Topic:</strong> ${{escapeHtml(record.topic)}}</div><div><strong>PMC:</strong> ${{record.pmc_id}}${{record.pmid ? ` · <strong>PMID:</strong> ${{record.pmid}}` : ''}}</div><div><strong>Organizations:</strong> ${{escapeHtml(record.organizations.join('; '))}}</div></div>
        </article>
      `).join('');
      byId('recordList').querySelectorAll('.record-card').forEach((card) => {{
        card.addEventListener('click', () => {{
          state.selectedRecordId = card.dataset.pmc;
          renderList();
        }});
      }});
      renderDetail(items.find((record) => record.pmc_id === state.selectedRecordId));
    }}

    function wireControls() {{
      byId('searchInput').addEventListener('input', (event) => {{ state.search = event.target.value; renderList(); }});
      byId('yearSelect').addEventListener('change', (event) => {{ state.year = event.target.value; renderList(); }});
      byId('depthSelect').addEventListener('change', (event) => {{ state.depth = event.target.value; renderList(); }});
      byId('sortSelect').addEventListener('change', (event) => {{ state.sort = event.target.value; renderList(); }});
    }}

    buildMetrics();
    buildPills();
    buildYearOptions();
    wireControls();
    renderList();
  </script>
</body>
</html>
"""


def render_report(summary):
    lines = [
        "# Clean Trajectory Report",
        "",
        f"- Clean trajectories: {summary['overall']['total_clean_trajectories']}",
        f"- Definition: {summary['overall']['definition']}",
        "",
        "## By Corpus",
        "",
    ]
    for row in summary["category_summaries"]:
        lines.append(
            f"- {row['label']}: {row['total_records']} clean trajectories, median prior depth {row['median_prior_depth']}, max prior depth {row['max_prior_depth']}"
        )
    lines.append("")
    lines.append("## Example Trajectories")
    lines.append("")
    for record in summary["records"][:20]:
        lines.append(f"- `{record['category']}` {record['pmc_id']} ({record['year']}): {record['title']} | nodes={record['node_count']}")
    lines.append("")
    return "\n".join(lines)