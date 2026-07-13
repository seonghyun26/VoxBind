#!/usr/bin/env python3
"""Inject into 260701_meeting.html §4: (a) an example card (thrombin 2jh6/2jh5) with the two
molecule SVGs + per-model predictions, and (b) the all-available-pairs table (no test filter)."""
import re
HTML = "/home/shpark/prj-denovo/VoxBind/notebook/html/260701/260701_meeting.html"

def svg(pid):
    s = open(f"/tmp/cliff_{pid}.svg").read()
    s = re.sub(r"<\?xml[^>]*\?>", "", s)            # strip xml decl, inline the <svg>
    return re.sub(r"<svg ", '<svg style="max-width:330px;height:auto;" ', s, count=1)

# ── (a) example card ───────────────────────────────────────────────────
CARD = f"""
  <section class="block">
    <p class="table-title">4.2 &nbsp;&middot;&nbsp; Example &mdash; a cliff only density resolves</p>
    <p class="table-sub">Thrombin (CHEMBL204). The two co-crystallised ligands differ by a <b>single bond</b> &mdash; a saturated
      <b>ethyl</b> linker (2jh6) vs an (E)-<b>vinyl</b> linker (2jh5, highlighted) to the chlorothiophene &mdash; yet potency differs
      <b>21&times;</b> (&Delta;pKi&nbsp;1.33). <b>Only the density model (C+D+G) ranks the pair correctly</b>; coords and both supervised baselines invert it.</p>
    <div class="table-wrap" style="display:flex;flex-wrap:wrap;gap:8px;align-items:flex-start;padding:14px;">
      <div style="text-align:center;flex:1;min-width:300px;">
        <div style="font-weight:600;font-size:13.5px;margin-bottom:2px;">2jh6 &nbsp;&middot;&nbsp; pKi <b>7.77</b> <span class="sd">(more potent)</span></div>
        {svg('2jh6')}
        <div class="sd">ethyl linker &mdash;CH<sub>2</sub>CH<sub>2</sub>&mdash;</div>
      </div>
      <div style="text-align:center;flex:1;min-width:300px;">
        <div style="font-weight:600;font-size:13.5px;margin-bottom:2px;">2jh5 &nbsp;&middot;&nbsp; pKi <b>6.44</b> <span class="sd">(weaker)</span></div>
        {svg('2jh5')}
        <div class="sd">vinyl linker &mdash;CH=CH&mdash;</div>
      </div>
    </div>
    <div class="table-wrap" style="margin-top:14px;">
    <table class="results">
      <thead><tr class="sub">
        <th class="col-method">Model</th><th>pred 2jh6</th><th>pred 2jh5</th><th>predicted order</th><th>correct?</th>
      </tr></thead>
      <tbody>
        <tr><td class="col-method"><b>ground truth (pKi)</b></td><td class="metric"><span class="val">7.77</span></td>
            <td class="metric"><span class="val">6.44</span></td><td><b>2jh6 &gt; 2jh5</b></td><td>&mdash;</td></tr>
        <tr><td class="col-method">C+D+G &middot; ChannelViT <span class="tag supervised">density</span></td>
            <td class="metric"><span class="val">6.93</span></td><td class="metric"><span class="val">6.53</span></td>
            <td>2jh6 &gt; 2jh5</td><td class="best"><b style="color:#1d5a3a;">&#10003; correct</b></td></tr>
        <tr><td class="col-method">C &middot; ViT (coords)</td><td class="metric"><span class="val">6.74</span></td>
            <td class="metric"><span class="val">6.78</span></td><td>2jh6 &lt; 2jh5</td><td><b style="color:#b0573f;">&#10007; inverted</b></td></tr>
        <tr><td class="col-method">TargetDiff / EGNN <span class="tag supervised">supervised</span></td>
            <td class="metric"><span class="val">6.95</span></td><td class="metric"><span class="val">6.97</span></td>
            <td>2jh6 &lt; 2jh5</td><td><b style="color:#b0573f;">&#10007; inverted</b></td></tr>
        <tr><td class="col-method">HBGSA <span class="tag supervised">supervised</span></td>
            <td class="metric"><span class="val">5.99</span></td><td class="metric"><span class="val">6.29</span></td>
            <td>2jh6 &lt; 2jh5</td><td><b style="color:#b0573f;">&#10007; inverted</b></td></tr>
      </tbody>
    </table>
    </div>
    <p class="notes"><b>Why density helps here:</b> the two ligands share an <i>identical atom graph</i> &mdash; the only change is one C&ndash;C
      bond order (ethyl&rarr;vinyl). A coords/atom-type encoder sees essentially the same molecule, so it cannot separate them; the
      experimental electron density reflects the electronic/conformational consequence of the saturation, which the density model reads.</p>
  </section>
"""

# ── (b) all-available-pairs table ──────────────────────────────────────
ALLTABLE = """
  <section class="block">
    <p class="table-title">4.3 &nbsp;&middot;&nbsp; All reachable pairs &mdash; without the out-of-sample restriction</p>
    <p class="table-sub">Dropping the &ldquo;both-in-test&rdquo; filter and scoring every pair each model can reach. The frozen probes predict on
      all cliff molecules with cached features (<b>11/17 pairs</b>; 6 molecules &mdash; the 4 new structures + 2 others &mdash; have no features, so 17 is
      unreachable). <b>Molecules in our train/val split are IN-SAMPLE (leaky, optimistic)</b> for the probes. The supervised baselines only ship
      test-split predictions &rarr; still 7 pairs. sign-acc &uarr;, RMSE<sub>cliff</sub> &darr;; best shaded.</p>
    <div class="table-wrap">
    <table class="results">
      <thead><tr class="sub">
        <th class="col-method">Method</th><th>sign-acc &uarr;</th><th>RMSE<sub>cliff</sub> &darr;</th><th>n pairs</th>
      </tr></thead>
      <tbody>
        <tr><td class="col-method">C &middot; ViT (coords only)</td>
            <td class="metric best"><span class="val">0.818</span> <span class="sd">9/11</span></td>
            <td class="metric best"><span class="val">0.911</span></td><td class="metric"><span class="val">11</span></td></tr>
        <tr><td class="col-method">C+D+G &middot; ChannelViT <span class="tag supervised">density</span></td>
            <td class="metric"><span class="val">0.727</span> <span class="sd">8/11</span></td>
            <td class="metric"><span class="val">1.031</span></td><td class="metric"><span class="val">11</span></td></tr>
        <tr><td class="col-method">TargetDiff / EGNN <span class="tag supervised">supervised</span></td>
            <td class="metric"><span class="val">0.286</span> <span class="sd">2/7</span></td>
            <td class="metric"><span class="val">1.224</span></td><td class="metric"><span class="val">7</span></td></tr>
        <tr><td class="col-method">HBGSA <span class="tag supervised">supervised</span></td>
            <td class="metric"><span class="val">0.571</span> <span class="sd">4/7</span></td>
            <td class="metric"><span class="val">0.979</span></td><td class="metric"><span class="val">7</span></td></tr>
      </tbody>
    </table>
    </div>
    <p class="notes"><b>The density win does not survive.</b> On the clean out-of-sample 7 (&sect;4.1) C+D+G ranks 7/7 vs coords 5/7; adding the 4
      in-sample pairs <b>flips the ranking &mdash; coords 9/11 &gt; density 8/11</b>. So the &sect;4.1 density advantage is <b>not robust to which
      pairs are included</b>, reinforcing that n is far too small to conclude. (The 4 extra pairs are also leaky &mdash; in-sample for the probe &mdash; so
      even this table is optimistic.) <b>Net: on this tiny density-backed cliff set no ranking is stable; a real conclusion needs a larger cliff set.</b></p>
  </section>
"""

html = open(HTML).read()
anchor = "small, clean cross-check, not a benchmark. <span class=\"sd\">Larger n needs looser"
# insert CARD + ALLTABLE right after the 4.1 </section> (which follows the 4.1 notes)
mark = "</ul>\n    </p>\n  </section>\n  </div>"
idx = html.find(mark)
assert idx != -1, "section-4 close marker not found"
# only the FIRST such close after §4.1 — verify it's within section 4 by checking the anchor precedes it
assert 0 < html.find(anchor) < idx, "anchor/marker ordering wrong"
insert_at = idx + len("</ul>\n    </p>\n  </section>")   # right after 4.1 </section>, before </div>
html = html[:insert_at] + "\n" + CARD + ALLTABLE + html[insert_at:]
open(HTML, "w").write(html)
print("injected 4.2 (example card) + 4.3 (all-pairs table). new size:", len(html))
