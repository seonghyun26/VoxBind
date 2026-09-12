"""build_ligand_similarity_appendix.py — assemble the two paste-ready text files.

    ligand_similarity_appendix.txt   everything: table, paragraph, all measured values,
                                     every metric's source paper, reproduction notes
    novelty_paragraph.txt            just the novelty/SNN paragraph and the bib entries
                                     it cites, for pasting straight into the manuscript

The paper table, the body paragraph, the measured numbers, the citations and the
reproduction notes in one paste-ready text file, in the same shape as this folder's
vina_config_appendix.txt.

The table and the numbers are READ from build_reference_similarity.py's outputs
(reference_similarity.tex / .csv / .json) rather than retyped, so re-running the
builder and then this script can never leave the two disagreeing.

    cd notebook/html/260910/fig-ref-ligand-similarity
    /opt/conda/envs/voxbind/bin/python build_reference_similarity.py
    /opt/conda/envs/voxbind/bin/python build_ligand_similarity_appendix.py
"""
import csv
import json
import os
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
H = HERE                    # the builder now writes its tables into this folder
OUT = os.path.join(HERE, "ligand_similarity_appendix.txt")

tex = open(f"{H}/reference_similarity.tex").read().rstrip("\n")
tex_wrap = open(f"{H}/reference_similarity_wrap.tex").read().rstrip("\n")
data = json.load(open(f"{H}/reference_similarity.json"))
rows = list(csv.DictReader(open(f"{H}/reference_similarity.csv")))


def f3(v):
    return f"{float(v):.3f}" if v else "—"


def pct(v):
    return f"{100 * float(v):.2f}%" if v else "—"


width = max(len(r["method"]) for r in rows)
lines = ["  " + "method".ljust(width)
         + "  pockets   n mol.   ECFP4 mean  ECFP4 median  Scaffold match"
         + "   Novelty  Scaf. nov.     SNN",
         "  " + "-" * (width + 90)]
for r in rows:
    lines.append("  " + r["method"].ljust(width)
                 + f'  {r["n_pockets"]:>7}  {r["n_mols"]:>7}'
                 + f'  {f3(r["ecfp4_mean"]):>10}  {f3(r["ecfp4_median"]):>12}'
                 + f'  {pct(r["scaffold_match"]):>14}'
                 + f'  {pct(r["novelty"]):>8}  {pct(r["scaffold_novelty"]):>10}'
                 + f'  {f3(r["snn"]):>6}')
measured = "\n".join(lines)

PARAGRAPH = r"""\textbf{Novelty.} Since \ours{} also receives reference-ligand
information, as in DecompDiff \citep{guan2024decompdiff}, we check that it does not replay
its training set. Against the CrossDocked training split \citep{francoeur2020three} we
report novelty -- the fraction of generated molecules whose canonical SMILES
\citep{weininger1988smiles} never occurs in training, defined identically in MOSES \citep{polykovskiy2020molecular} and GuacaMol
\citep{brown2019guacamol} -- and the same fraction taken over Bemis--Murcko scaffolds
\citep{bemis1996properties}, a stricter test that a new molecule can still fail, so that
the gap between the two is the share that is new only by decoration. We add MOSES'
similarity to a nearest neighbour (SNN), the mean Tanimoto similarity of a generated
molecule to the closest training one, scored against the training split rather than the
held-out set MOSES uses, so that a \emph{low} value means distance from everything the
model saw \citep{walters2020assessing}.
\Cref{fig:result-drug-novelty} places all nine methods within a narrow band on every
measure: none of them reproduces its training set. \ours{} sits at the novel end of that
band, and is more novel than the density-free
VoxBind\textsubscript{\scriptsize $\sigma$=0.9} it is built on by all three measures at
once. The few molecules that do occur in training are fragments, far below the size of a
typical generated one. Density conditioning therefore supplies pocket information, not a
template to copy."""

BAR, SUB = "=" * 80, "-" * 80

txt = f"""{BAR}
LIGAND SIMILARITY — appendix text (final) + measured values + provenance
작성 {date.today().isoformat()} · 대상: de-novo CrossDocked 표
이 폴더: notebook/html/260910/fig-ref-ligand-similarity/ (빌더·표·그림 전부)
빌더: build_reference_similarity.py → 이 파일: build_ligand_similarity_appendix.py
지표 출처 정리: reference_similarity_metrics.md · 그림: build_similarity_figure.py
{BAR}

{SUB}
[1] 논문용 LaTeX — 표
{SUB}
필요 패키지: booktabs, multirow. (\\shortstack 은 순정 LaTeX이라 makecell 불필요)

{tex}

본문 옆에 끼우는 wraptable 판 (260820 표와 같은 배치, wrapfig 추가로 필요):

{tex_wrap}

{SUB}
[2] 논문용 LaTeX — 본문 문단
{SUB}
캡션에서 지표 설명을 뺐으므로 정의는 이 문단이 담당합니다.

{PARAGRAPH}

주의 — "no more similar than prior methods" 로 강화하지 않았습니다. Ours 는 ECFP4 0.099 /
scaffold 0.81% 로 TargetDiff(0.091 / 0.62%)보다 두 지표 모두 높습니다. 대신 density 없는
같은 모델(VoxBind σ=0.9)과 ECFP4 평균이 0.099 로 정확히 같다는 점을 근거로 썼습니다 —
density 를 넣어도 레퍼런스 모방이 늘지 않았다는 가장 직접적인 증거입니다.

{SUB}
[3] 측정값
{SUB}
{data['n_shared_pockets']}개 공통 포켓, 포켓별 평균/중앙값을 낸 뒤 포켓 간 macro-average.

{measured}

표에서 뺀 우리 나머지 arm (같은 79 포켓, 2026-09-03 측정):
  Ours σ=1.0        ECFP4 0.107 / 0.102   scaffold 1.23%
  Ours ligmask      ECFP4 0.107 / 0.100   scaffold 1.44%

{SUB}
[4] 지표와 출처 논문
{SUB}
본문/그림에 나가는 지표는 5개. 앞의 둘은 레퍼런스 리간드 상대, 뒤의 셋은 학습셋 상대입니다.

  [레퍼런스 리간드 상대]
  ECFP4 / Morgan Tanimoto   radius 2, 2048-bit folded.
      rogers2010extended · Rogers & Hahn, "Extended-Connectivity Fingerprints",
      J. Chem. Inf. Model. 50(5):742-754, 2010. doi:10.1021/ci100050t
  Bemis-Murcko scaffold match   canonical scaffold SMILES 완전일치율. 고리 없는
      분자는 non-match.
      bemis1996properties · Bemis & Murcko, "The Properties of Known Drugs. 1.
      Molecular Frameworks", J. Med. Chem. 39(15):2887-2893, 1996. doi:10.1021/jm9602928

  [학습셋 상대 — 새로 추가]
  Novelty (SMILES)          학습 split 에 없던 분자 비율. MOSES/GuacaMol 정의 그대로.
  Scaffold novelty          같은 것을 Bemis-Murcko scaffold 수준에서.
  SNN                       생성 분자마다 가장 가까운 학습 분자와의 ECFP4 Tanimoto, 평균.
      polykovskiy2020molecular · Polykovskiy et al., "Molecular Sets (MOSES)",
      Front. Pharmacol. 11:565644, 2020. doi:10.3389/fphar.2020.565644 · arXiv:1811.12823
      brown2019guacamol · Brown, Fiscato, Segler & Vaucher, "GuacaMol",
      J. Chem. Inf. Model. 59(3):1096-1108, 2019. doi:10.1021/acs.jcim.8b00839
      walters2020assessing · Walters & Murcko, "Assessing the impact of generative AI on
      medicinal chemistry", Nat. Biotechnol. 38:143-145, 2020. doi:10.1038/s41587-020-0418-2
          → "학습셋과 가장 유사한 분자를 같이 보고하라"가 이 논문의 요구입니다.
             SNN 을 test 가 아니라 train 상대로 재는 근거가 여기입니다.
      francoeur2020three · Francoeur et al., CrossDocked2020, J. Chem. Inf. Model.
      60(9):4200-4215, 2020. doi:10.1021/acs.jcim.0c00411   ← 학습셋 출처

  ※ SNN 방향 주의. MOSES 는 SNN 을 test set 상대로 재고 "높을수록 좋다"(생성 분포가
    레퍼런스 분포에 가깝다)로 읽습니다. 여기서는 train set 상대로 재고 "낮을수록
    좋다"(암기가 아니다)로 읽습니다. 정의는 같고 기준 집합과 해석 방향이 반대이므로,
    본문에 "against the training split ... so that a low value means distance" 한 줄이
    반드시 붙어야 합니다. 빼면 MOSES 를 인용하면서 반대로 읽는 셈이 됩니다.

  ※ Novelty 분모 주의. CrossDocked split_by_name.pt 의 학습쌍은 100,000 개지만 고유
    분자는 8,765 개, 고유 scaffold 는 4,926 개뿐입니다. 98.9% 는 "8,765 개 중 어느
    것과도 다르다"는 뜻이고, ZINC 250k 상대의 98.9% 와 같은 강도의 주장이 아닙니다.
    숫자만 쓰고 이 사실을 안 쓰면 과장입니다.

  ※ ECFP4 바닥값 (permutation control, 이 데이터에서 실측 2026-09-09)
        matched    생성 vs 자기 포켓 레퍼런스                  0.0994
        shuffled   생성 vs 다른 포켓 레퍼런스                  0.0785
        ref-ref    79개 레퍼런스끼리 3,081 쌍                  0.0933 (median 0.0784)
    즉 0.10 은 "서로 무관한 두 실제 약물" 수준입니다. 포켓 특이 신호는 +0.021 로 실재
    하지만 작습니다. 본문의 0.093 이 이 값이고, 이걸 안 쓰면 "왜 이렇게 낮냐"는 질문을
    리뷰어에게서 그대로 받습니다.

베이스라인 쪽 관행 (ECFP4 를 고른 이유): TargetDiff/DecompDiff/Pocket2Mol 계열도 Morgan
Tanimoto 를 씁니다.
  guan20233d-5e4 · Guan et al., TargetDiff, ICLR 2023. arXiv:2303.03543

부록용 (--full / --with-3d 로만 나옴, 방법 순위는 ECFP4 와 동일)
  MACCS 167-bit keys       Durant et al., J. Chem. Inf. Comput. Sci. 42(6):1273-1280, 2002.
                           doi:10.1021/ci010132r
  AtomPair 2048-bit        Carhart, Smith & Venkataraghavan,
                           J. Chem. Inf. Comput. Sci. 25(2):64-73, 1985. doi:10.1021/ci00046a002
  RDKit path fingerprint   논문 없음. Daylight theory manual + RDKit toolkit 인용.
  Dice (unfolded count Morgan)  Rogers & Hahn 2010 + Dice, Ecology 26(3):297-302, 1945.
                           ※ folded 2048-bit 로 계산하면 값이 달라짐 (0.222 vs 0.208).
  3D shape Tanimoto        Grant, Gallardo & Pickup, J. Comput. Chem. 17(14):1653-1666, 1996.
  Tanimoto 계수 자체        Jaccard 1901 / Rogers & Tanimoto, Science 132:1115-1118, 1960.

건드리면 안 되는 것: Vina 표의 Diversity 열은 TargetDiff 가 Chem.RDKFingerprint 로 계산한
값이고 DecompDiff/DiffSBDD 가 그 코드를 물려받았습니다. 정리한다고 Morgan 으로 바꾸면
baseline 과 조용히 비교불가가 됩니다.

{SUB}
[5] 재현
{SUB}
이 서버:
    cd notebook/html/260910/fig-ref-ligand-similarity
    /opt/conda/envs/voxbind/bin/python build_reference_similarity.py
    옵션: --full (MACCS/AtomPair/RDKit/Dice) · --with-3d (3D shape) · --novelty
          --own-pockets (교집합 대신 각 방법 전체 포켓) · --methods <이름...>
    산출: reference_similarity.{{json,csv,tex}} + _wrap.tex + _table.html (이 폴더)
    results2latex.ipynb 마지막 셀이 _table.html 을 읽어 위 [1] 의 두 판 모두와
    byte-identical 한 LaTeX 를 냅니다. 그 다음 이 파일과 그림을 재생성:
    /opt/conda/envs/voxbind/bin/python build_ligand_similarity_appendix.py
    /opt/conda/envs/voxbind/bin/python build_similarity_figure.py

분자는 전부 samples.sdf 에서 sanitize=True 로 읽습니다. target_*/metrics.json 의 캐시된
SMILES 가 더 빠르지만 방법마다 집합이 달라서(TargetDiff 7,287 vs SDF 7,798) --from-metrics
는 경고와 함께 opt-in 입니다.

이 서버에 있는 것: TargetDiff (base_drug/eval/targetdiff), FuncBind
(funcbind/artifacts/reproduction/crossdocked/paper_run/gpu0..3/samples, 25포켓씩 샤딩),
VoxBind σ=0.9 / σ=1.0, Ours (frozenenc atomblob7 v2.1 σ=0.9) + 표에서 뺀 arm 2개.

여기 없는 것 — AR, Pocket2Mol, DiffSBDD, DecompDiff. Blackwell(sm_120) 박스에서
샘플링했고 집계값만 260903/baseline_vina.json 으로 왔습니다. base_drug/samples/ 밑의
{{ar,pocket2mol,decompdiff}} 는 sweep 잔해(54/23/15 포켓, .pt)라 못 씁니다. 그쪽에서:

    scp notebook/html/260910/fig-ref-ligand-similarity/build_reference_similarity.py <blackwell>:~/
    # METHODS 를 그쪽 sample root 로 교체. root = target_*/ 를 담은 디렉터리이고
    # 각 target_*/ 안에 samples.sdf 와 레퍼런스 리간드(*_ref.sdf 또는 *_pocket10.pdb
    # 옆의 *_lig_*.sdf)가 있으면 됩니다. GPU 샤딩된 run 은 root 리스트로 줘도 됩니다.
    METHODS = {{
        "AR":         "/…/eval/ar",
        "Pocket2Mol": "/…/eval/pocket2mol",
        "DiffSBDD":   "/…/eval/diffsbdd",
        "DecompDiff": "/…/eval/decompdiff",
    }}
    python build_reference_similarity.py --own-pockets   # rdkit 외 의존성 없음

    → reference_similarity.json 을 여기로 가져와 병합. 포켓 세트가 같으면 --own-pockets
      없이 교집합으로 돌리는 편이 비교에 맞습니다.

Reference 행은 일부러 없습니다 — 결정 구조 리간드의 자기 자신과의 유사도는 1.0 이고
scaffold 는 항상 일치해서 정보가 없습니다.

{BAR}
"""

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write(txt)
print(f"wrote {OUT} ({len(txt.splitlines())} lines)")


# ── the short companion: the paragraph and only the references it cites ───────
# Same PARAGRAPH object as [2] above, so the two files can never disagree, and the
# measured numbers below are read from the CSV rather than retyped.
by_method = {r["method"]: r for r in rows}
OURS_ROW = by_method["Ours v1"]
measured_novelty = "\n".join(
    "  " + r["method"].ljust(width)
    + f'  {pct(r["novelty"]):>9}  {pct(r["scaffold_novelty"]):>11}  {f3(r["snn"]):>7}'
    for r in rows)

note = rf"""{BAR}
NOVELTY / SNN — 논문 본문 문단 + 인용 문헌
작성 {date.today().isoformat()} · 그림: similarity_novelty.{{png,svg,pdf}}
숫자 출처: reference_similarity.csv (build_reference_similarity.py --novelty)
{BAR}

{SUB}
[1] 본문 문단 (LaTeX)
{SUB}
\label{{fig:result-drug-novelty}} 를 novelty 그림에 답니다. 레퍼런스 리간드 유사도
(ECFP4 / scaffold match) 표를 따로 싣는다면 그건 별도 문단입니다 — 이 문단은 학습셋
상대 지표만 이야기합니다.

{PARAGRAPH}

{SUB}
[2] 측정값 — 79개 공통 포켓, 포켓별 평균 후 macro-average
{SUB}
  {"method".ljust(width)}    Novelty   Scaf. nov.      SNN
  {"-" * (width + 36)}
{measured_novelty}

  9개 방법 전부 이 박스에서 계산했습니다 (2026-09-09). AR / Pocket2Mol / DiffSBDD /
  DecompDiff 의 분자는 Blackwell 이 아니라 results/task2-drugdesign/<method>/samples/meta
  의 TargetDiff-식 meta 번들에서 옵니다 (results/dropbox_pull_baselines.sh 로 받음).
  교차검증: 이 네 방법의 79-포켓 분자 수가 7655 / 7772 / 7720 / 6427 로 Blackwell 이
  보고한 값과 정확히 일치하고, ECFP4 평균/중앙값/scaffold match 도 보고된 소수점까지
  재현됩니다 (0.100/0.096/1.04%, 0.097/0.092/1.12%, 0.089/0.085/0.55%,
  0.152/0.137/2.17%). 같은 분자를 같은 포켓에서 재고 있다는 뜻입니다.

  구현: Novelty 는 canonical SMILES 완전일치, Scaffold novelty 는 Bemis-Murcko canonical
  scaffold 완전일치 (고리 없는 분자는 분모에서 제외), SNN 은 Morgan/ECFP4 radius 2,
  2048-bit folded Tanimoto 의 학습셋 최근접값 평균. 본문이 지문 종류를 안 쓰므로 캡션이나
  부록에 이 한 줄은 남아야 합니다. molecule novelty 는 통과하기 쉬운 시험입니다 — RDKit
  canonical SMILES 가 기본 isomeric 이라 입체이성질체도 "새 분자"로 셉니다.

  두 수준이 실제로 세는 것 (Ours v1, 79 포켓 전수 pooled — 표의 macro-average 와 분모가
  달라 숫자가 조금 다릅니다):
      학습셋에 문자 그대로 있던 분자        52 / 7,356 고리보유    0.71%
      새 분자인데 골격은 학습셋에 있음      922 / 7,356           12.53%  <- 아령 길이
      새 분자 + 새 골격                   6,382 / 7,356          86.76%
  겹치는 분자의 정체 (본문 마지막 문장의 근거): 고리 없는 것까지 포함하면 7,873 개 중
  91 개(1.16%), 크기 median 9 / max 18 / min 3 heavy atom. 생성 분자 전체 median 은 25,
  20 을 넘는 것 중 재생된 것은 0 개입니다. 즉 3-클로로벤즈아마이드 같은 조각이지 리간드
  크기의 분자가 아닙니다.

  함정 1 — 본문은 일부러 순위를 주장하지 않습니다. 1위가 AR (novelty 99.13%) / Ours
  (scaffold novelty 84.83%) / TargetDiff (SNN 0.280) 로 갈리고, 9개가 전부 좁은 띠 안이라
  지문이나 split 을 조금만 바꿔도 뒤집힙니다. 본문이 말하는 건 둘뿐입니다 — 그 띠의 novel
  쪽 절반 (2/9 · 1/9 · 4/9), 그리고 density 없는 자기 베이스보다 세 지표 모두 낫다는 것.

  함정 2 — Scaffold novelty 는 분자 크기와 섞입니다. 걸리는 골격이 벤젠, 테트라하이드로
  피란처럼 자명한 것들이라, "흥미로운 골격을 베꼈나"보다 "고리가 단순한 분자를 많이
  만들었나"에 더 민감합니다. Pocket2Mol 의 69.8% 도 표절이 아니라 작은 분자를 많이 만든
  결과일 수 있습니다. Vina 순위 대부분이 크기 효과였던 것과 같은 함정입니다.

{SUB}
[3] 인용 문헌 — 이 문단이 쓰는 것만
{SUB}
어느 논문이 어느 지표인지:
  Novelty (SMILES)   MOSES + GuacaMol — 둘 다 "학습셋에 없는 분자의 비율"로 같은 정의.
  Scaffold novelty   같은 것을 Bemis-Murcko scaffold 수준에서. MOSES 의 Scaff 는 이것과
                     다른 지표(생성/레퍼런스 scaffold 분포의 cosine)이므로, scaffold
                     novelty 는 novelty 정의 + Bemis & Murcko 조합으로 인용합니다.
  SNN                MOSES 만. GuacaMol 의 distribution-learning 지표 5개는 validity,
                     uniqueness, novelty, KL divergence, FCD 로 SNN 이 없습니다.
                     본문에서 SNN 을 GuacaMol 에 함께 걸면 틀립니다.

@article{{polykovskiy2020molecular,
  title   = {{{{Molecular Sets ({{MOSES}}): A Benchmarking Platform for Molecular Generation Models}}}},
  author  = {{Polykovskiy, Daniil and Zhebrak, Alexander and Sanchez-Lengeling, Benjamin
             and Golovanov, Sergey and Tatanov, Oktai and Belyaev, Stanislav and
             Kurbanov, Rauf and Artamonov, Aleksey and Aladinskiy, Vladimir and
             Veselov, Mark and Kadurin, Artur and Johansson, Simon and Chen, Hongming
             and Nikolenko, Sergey and Aspuru-Guzik, Al{{\'a}}n and Zhavoronkov, Alex}},
  journal = {{Frontiers in Pharmacology}},
  volume  = {{11}}, pages = {{565644}}, year = {{2020}},
  doi     = {{10.3389/fphar.2020.565644}}}}

@article{{brown2019guacamol,
  title   = {{{{GuacaMol}}: Benchmarking Models for de Novo Molecular Design}},
  author  = {{Brown, Nathan and Fiscato, Marco and Segler, Marwin H. S. and Vaucher, Alain C.}},
  journal = {{Journal of Chemical Information and Modeling}},
  volume  = {{59}}, number = {{3}}, pages = {{1096--1108}}, year = {{2019}},
  doi     = {{10.1021/acs.jcim.8b00839}}}}

@article{{walters2020assessing,
  title   = {{Assessing the impact of generative {{AI}} on medicinal chemistry}},
  author  = {{Walters, W. Patrick and Murcko, Mark}},
  journal = {{Nature Biotechnology}},
  volume  = {{38}}, number = {{2}}, pages = {{143--145}}, year = {{2020}},
  doi     = {{10.1038/s41587-020-0418-2}}}}

@article{{weininger1988smiles,
  title   = {{{{SMILES}}, a chemical language and information system. 1. Introduction to
             methodology and encoding rules}},
  author  = {{Weininger, David}},
  journal = {{Journal of Chemical Information and Computer Sciences}},
  volume  = {{28}}, number = {{1}}, pages = {{31--36}}, year = {{1988}},
  doi     = {{10.1021/ci00057a005}}}}

@article{{bemis1996properties,
  title   = {{The Properties of Known Drugs. 1. Molecular Frameworks}},
  author  = {{Bemis, Guy W. and Murcko, Mark A.}},
  journal = {{Journal of Medicinal Chemistry}},
  volume  = {{39}}, number = {{15}}, pages = {{2887--2893}}, year = {{1996}},
  doi     = {{10.1021/jm9602928}}}}

@article{{francoeur2020three,
  title   = {{Three-Dimensional Convolutional Neural Networks and a Cross-Docked Data Set
             for Structure-Based Drug Design}},
  author  = {{Francoeur, Paul G. and Masuda, Tomohide and Sunseri, Jocelyn and Jia, Andrew
             and Iovanisci, Richard B. and Snyder, Ian and Koes, David R.}},
  journal = {{Journal of Chemical Information and Modeling}},
  volume  = {{60}}, number = {{9}}, pages = {{4200--4215}}, year = {{2020}},
  doi     = {{10.1021/acs.jcim.0c00411}}}}

guan2024decompdiff (DecompDiff) 는 이미 쓰고 계신 키입니다. francoeur2020three 와
bemis1996properties 도 bib 에 이미 있으면 그대로 재사용하세요.

  ※ SMILES 2 (Weininger et al. 1989, CANGEN 알고리즘) 는 인용하지 않습니다. novelty 의
    동치 판정을 실제로 하는 것은 RDKit 의 canonical ranking 이지 CANGEN 이 아니라서,
    1989 를 걸면 우리가 안 쓴 알고리즘을 인용하는 셈이 됩니다. 정규화의 출처로는
    RDKit 을 소프트웨어로 인용하는 편이 맞습니다:
        RDKit: Open-source cheminformatics. https://www.rdkit.org  (버전 명시)

{SUB}
[4] 리뷰어가 물고 늘어질 두 가지
{SUB}
(a) SNN 의 방향이 MOSES 와 반대입니다.
    MOSES 의 SNN 정의는 "average similarity of generated molecules to the nearest
    molecule from the TEST set" 이고 "높을수록 좋다"(생성 분포가 레퍼런스 분포에
    가깝다)로 읽습니다. 우리는 TRAIN set 상대로 재고 "낮을수록 좋다"(암기가 아니다)로
    읽습니다. 정의는 같고 기준 집합과 해석 방향이 뒤집힌 것이라, 본문의
    "against the training split rather than a held-out set, so that a low value means
    distance from everything the model was shown" 한 줄은 빼면 안 됩니다. 빼면 MOSES 를
    인용하면서 반대로 읽는 셈이 됩니다. Walters & Murcko 의 "학습셋에서 가장 유사한
    분자를 함께 보고하라"가 이 용법의 근거라 같이 인용했습니다.

(b) Novelty 의 분모가 작습니다.
    CrossDocked split_by_name.pt 의 학습쌍은 100,000 개지만 고유 분자는 8,765 개,
    고유 scaffold 는 4,926 개뿐입니다. {pct(OURS_ROW["novelty"])} 는 "8,765 개 중 어느
    것과도 다르다"는 뜻이고, ZINC 250k 상대의 같은 숫자와 동일한 강도의 주장이
    아닙니다. 지면이 허락하면 캡션에 이 한 줄을 넣는 편이 안전합니다.

{BAR}
"""

NOTE = os.path.join(HERE, "novelty_paragraph.txt")
with open(NOTE, "w", encoding="utf-8") as fh:
    fh.write(note)
print(f"wrote {NOTE} ({len(note.splitlines())} lines)")
