# 요청 — reference-ligand similarity, baseline 4종 (AR / Pocket2Mol / DiffSBDD / DecompDiff)

**한 줄:** 스크립트 하나 실행해서 `reference_similarity.json` 파일 하나만 돌려주시면 됩니다.
GPU 불필요, rdkit만 있으면 되고 몇 분 걸립니다.

**왜 필요한지:** CrossDocked de-novo 표에 "생성 분자가 각 pocket의 reference ligand를 그대로
베끼지 않는다"는 근거로 ECFP4 Tanimoto와 Bemis–Murcko scaffold match 두 열을 넣습니다. 우리
쪽 5개(TargetDiff / FuncBind / VoxBind σ=0.9 / σ=1.0 / Ours)는 측정을 끝냈는데, 그쪽에서
샘플링한 4개는 **per-molecule SMILES가 이 서버에 없어서** 측정이 안 됩니다. Vina 표는 집계값만
받아도 됐지만 이건 분자 자체가 있어야 합니다.

---

## 0. 실행 전 30초 점검 — 이게 제일 중요합니다

스크립트는 조건에 안 맞는 target 디렉터리를 **에러 없이 조용히 건너뜁니다.** 그래서 돌리기 전에
이것부터 찍어주세요:

```bash
python - <<'PY'
import glob, os
ROOTS = {
    "AR":         "/…/ar",
    "Pocket2Mol": "/…/pocket2mol",
    "DiffSBDD":   "/…/diffsbdd",
    "DecompDiff": "/…/decompdiff",
}
for name, root in ROOTS.items():
    ts = sorted(glob.glob(os.path.join(root, "target_*")))
    print(f"{name:12s} {len(ts):3d} target dirs   {root}")
    if ts:
        print("              ", sorted(os.listdir(ts[0]))[:12])
PY
```

각 `target_*/` 안에 아래 셋이 있어야 합니다:

| 파일 | 쓰임 | 없으면 |
|---|---|---|
| `*_pocket10.pdb` | 파일명만 사용 (포켓 ID 정규화) | 그 포켓 통째로 스킵 |
| `*_ref.sdf` 또는 파일명에 `_lig_` 포함된 sdf | reference ligand | 그 포켓 통째로 스킵 |
| `samples.sdf` | 생성 분자 (sanitize=True로 읽음) | 그 포켓 통째로 스킵 |

포켓 ID는 `*_pocket10.pdb` 파일명에서 **마지막 `__` 뒤쪽**을 씁니다. 우리 쪽은 UniProt 접두사가
붙고 baseline은 안 붙는데, 이 정규화 덕분에 두 서버 결과가 그대로 맞물립니다.

**출력이 위 표와 다르면 스크립트를 고치지 마시고 그 `ls` 결과만 보내주세요.** 제가 loader를
맞추는 쪽이 빠르고, 정의가 갈라질 위험도 없습니다.

---

## 1. 실행

```bash
scp <thisbox>:/home1/irteam/VoxBind/notebook/html/260910/fig-ref-ligand-similarity/build_reference_similarity.py ~/
```

파일 안의 `METHODS`를 그쪽 경로로 교체합니다 (주석 처리된 4줄이 이미 들어 있습니다):

```python
METHODS = {
    "AR":         "/…/ar",
    "Pocket2Mol": "/…/pocket2mol",
    "DiffSBDD":   "/…/diffsbdd",
    "DecompDiff": "/…/decompdiff",
}
```

GPU로 샤딩된 run은 root를 **리스트**로 줘도 됩니다: `"AR": ["/…/gpu0/samples", "/…/gpu1/samples"]`.

```bash
python build_reference_similarity.py --methods AR Pocket2Mol DiffSBDD DecompDiff
# 결과는 스크립트가 있는 디렉터리에 떨어집니다 (--out-dir 로 변경 가능)
```

rdkit 외 의존성 없습니다. 4개 방법 × 약 8천 분자 = 수 분.

---

## 2. 돌려주실 것

**`reference_similarity.json` 하나만.** 같이 생기는 `.csv` / `.tex` / `_table.html`은 이쪽에서
다시 만드니 안 보내셔도 됩니다.

⚠️ **콘솔에 찍힌 표만 보내면 안 됩니다.** JSON 안의 `per_pocket` 블록이 필요합니다 — 이쪽에서
우리 79개 density pocket과 교집합을 다시 잡아 재집계할 거라서요. `per_pocket`에는 `--own-pockets`
여부와 무관하게 각 방법이 커버한 **전체** 포켓이 들어가므로, 옵션은 아무거나 상관없습니다.

---

## 3. 보내기 전 자가 점검

콘솔이 이렇게 나오면 정상입니다:

- 방법마다 `NN pockets, NNNN molecules` — 포켓 79~100개, 포켓당 분자 약 100개
- `ECFP4 mean`이 대략 **0.08 ~ 0.13**. 0.0 근처면 reference를 못 읽은 것이고, 1.0 근처면
  reference ligand 자리에 생성 분자를 집은 것입니다.
- 어떤 방법이 `0 pockets` 또는 `skip …: none of […] exist`로 나오면 0번으로 돌아가기.

---

## 4. 건드리면 안 되는 것

- **fingerprint 정의를 바꾸지 마세요.** ECFP4 = Morgan radius 2 / 2048-bit folded Tanimoto,
  scaffold = Bemis–Murcko canonical SMILES 완전일치 (고리 없는 분자는 non-match). TargetDiff /
  DecompDiff / Pocket2Mol 계열과 같은 정의라야 한 표에 나란히 놓입니다.
- **`--from-metrics` 쓰지 마세요.** 캐시된 SMILES가 SDF와 분자 집합이 다릅니다 (TargetDiff에서
  7,287 vs 7,798). 방법마다 다른 분자 집합 위에서 비교하게 됩니다.
- Vina 표의 **Diversity 열은 이 스크립트와 무관**합니다. TargetDiff가 `Chem.RDKFingerprint`로
  계산한 값이고 DecompDiff/DiffSBDD가 그 코드를 물려받았습니다. Morgan으로 "정리"하면 baseline과
  조용히 비교 불가가 됩니다.
- **`baseline_vina.json`에 들어간 그 샘플과 같은 디렉터리**여야 합니다. 새로 샘플링한 것을 쓰면
  같은 표 안에서 Vina 행과 similarity 행이 다른 분자를 가리키게 됩니다.

---

## 5. (선택) 분포 그림까지 필요해지면

표 두 열만 필요하면 위로 충분합니다. 나중에 similarity ECDF 같은 그림을 그리게 되면 per-molecule
값이 필요한데, 그때는 제가 export 플래그를 붙인 버전을 다시 보내겠습니다 — 지금 미리 하실 필요
없습니다.

---

### 참고 — 이 요청의 근거 파일 (이 서버)

모두 `notebook/html/260910/fig-ref-ligand-similarity/` 안에 있습니다.

- `build_reference_similarity.py` — 스크립트 본체, 모듈 docstring에 같은 내용
- `ligand_similarity_appendix.txt` §5 — 재현 절차와 측정값
- `reference_similarity_metrics.md` — 지표별 출처 논문
