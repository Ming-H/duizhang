#!/usr/bin/env bash
# 对账 duizhang — 全量测试一条龙
#
# 用法：
#   ./run_tests.sh           全量（pytest + ruff + 七数据集评估 + 冒烟断言，约 5 分钟）
#   QUICK=1 ./run_tests.sh   快速档（pytest + ruff + default 集，约 1 分钟）
#
# 全部走 --mode heuristic（确定性、零外呼），与 CI 同口径。
# 任何一步失败，结尾汇总标红并以非零码退出。
set -u
cd "$(dirname "$0")"

PY=venv/bin/python
[ -x "$PY" ] || PY=python3
MODE="--mode heuristic"

# 结果落盘：test_reports/ 时间戳报告 + latest 软链（核对步骤会读回此文件）
REPORT_DIR=test_reports
mkdir -p "$REPORT_DIR"
REPORT="$REPORT_DIR/report-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee "$REPORT") 2>&1

PASS=0; FAIL=0; FAILED_ITEMS=()

ok()   { PASS=$((PASS+1)); printf "  ✓ %s\n" "$1"; }
bad()  { FAIL=$((FAIL+1)); FAILED_ITEMS+=("$1"); printf "  ✗ %s\n" "$1"; }
head() { printf "\n\033[1m%s\033[0m\n" "$1"; }

printf "对账全量测试报告  %s\n" "$(date '+%Y-%m-%d %H:%M:%S')"
printf "git: %s ｜ 语料库: %s\n" \
    "$(git rev-parse --short HEAD 2>/dev/null || echo '-')" \
    "$($PY -m duizhang.cli docs 2>/dev/null | tail -1)"

# ── 1. 静态检查 + 单元测试 ─────────────────────────────
head "① ruff 静态检查"
if $PY -m ruff check . >/dev/null 2>&1; then ok "ruff"; else bad "ruff"; fi

head "② 单元测试（conftest 封网，永不外呼）"
if $PY -m pytest tests/ -q >/tmp/dz_pytest.log 2>&1; then
    ok "pytest（$(grep -o '[0-9]* passed' /tmp/dz_pytest.log | tail -1)）"
else
    bad "pytest（详见 /tmp/dz_pytest.log）"; tail -5 /tmp/dz_pytest.log
fi

# ── 2. 真值评估集 ─────────────────────────────────────
eval_dataset() {  # eval_dataset <名> <是否校验refusal>
    local ds="$1" chk_ref="$2" out
    out=$($PY -m duizhang.cli eval $MODE --dataset "$ds" --json 2>/dev/null \
        | $PY -c "
import json,sys
s=json.load(sys.stdin)
r=s['refusal']
ok = s['retrieval_top3']==1.0 and s['answer_hit']==1.0 and (r in (1.0,None) if '$chk_ref'=='y' else True)
print(('PASS' if ok else 'FAIL'),
      f\"hit={s['answer_hit']} ret={s['retrieval_top3']} refusal={r if r is not None else 'N/A'} cov={s['mean_coverage']}\")
" 2>/dev/null)
    if [[ "$out" == PASS* ]]; then ok "eval ${ds}（${out#PASS }）"; else bad "eval ${ds}（$out）"; fi
}

head "③ 真值评估集（heuristic 确定性）"
eval_dataset default y
if [ "${QUICK:-0}" != "1" ]; then
    eval_dataset agent y
    eval_dataset innolight y
    eval_dataset maotai y
    eval_dataset quarters y
    eval_dataset multihop y
    eval_dataset multiturn n   # 无陷阱题，refusal=N/A 属正常
fi

# ── 3. 冒烟问答断言（真值对库核对过） ──────────────────
ask_check() {  # ask_check <描述> <问题> <类型:kw|refused|trend> <期望关键词>
    local desc="$1" q="$2" kind="$3" want="$4" out verdict
    out=$($PY -m duizhang.cli ask "$q" $MODE --json 2>/dev/null)
    verdict=$(printf '%s' "$out" | $PY -c "import json,sys;print(json.load(sys.stdin).get('verdict',''))" 2>/dev/null)
    case "$kind" in
      kw)     if printf '%s' "$out" | grep -q "$want"; then ok "$desc → $want"
              else bad "$desc（答案未见 $want，verdict=$verdict）"; fi ;;
      refused) if [ "$verdict" = "refused" ]; then ok "$desc → 正确拒答"
               else bad "$desc（应拒答，实为 $verdict）"; fi ;;
      trend)  if printf '%s' "$out" | grep -q "归母净利" && printf '%s' "$out" | grep -q "亿"; then
                  ok "$desc → 多期骨架（归母独立取值）"
              else bad "$desc（骨架缺归母行或亿值）"; fi ;;
    esac
}

head "④ 冒烟问答断言"
if [ "${QUICK:-0}" != "1" ]; then
    ask_check "口径·扣非"      "中际旭创 2025 一季度扣非净利润多少"     kw     "1,568,385,287.19"
    ask_check "口径·总收入"    "贵州茅台 2026 半年度营业总收入多少"     kw     "92,278,072,083.21"
    ask_check "简称·扣非"      "茅台 2026 年半年度扣非净利润多少"       kw     "44,464,207,646.01"
    ask_check "归母"           "贵州茅台 2025 年归母净利润多少"         kw     "82,320,067,101.68"
    ask_check "总资产"         "中际旭创 2025 年一季度末总资产多少"      kw     "31,582,872,453.01"
    ask_check "陷阱·期间外"    "贵州茅台 2023 年年报营业收入多少"       refused ""
    ask_check "陷阱·未来期间"  "中际旭创 2030 年净利润多少"             refused ""
    ask_check "趋势"           "中际旭创最近几个季度的净利润趋势"       trend  ""
fi

# ── 汇总 ──────────────────────────────────────────────
printf "\n\033[1m═══ 汇总：%d 通过 / %d 失败 ═══\033[0m\n" "$PASS" "$FAIL"
if [ "$FAIL" -gt 0 ]; then
    printf "失败项：\n"; for f in "${FAILED_ITEMS[@]}"; do printf "  - %s\n" "$f"; done
fi

# ── 核对：读回报告文件，验证落盘结果与运行计数一致 ─────
sleep 0.5   # 等 tee 落盘
FILE_PASS=$(grep -c '^  ✓' "$REPORT" || true)
FILE_FAIL=$(grep -c '^  ✗' "$REPORT" || true)
if [ "$FILE_PASS" -eq "$PASS" ] && [ "$FILE_FAIL" -eq "$FAIL" ]; then
    printf "\n核对：报告文件（✓%s / ✗%s）与运行计数（%s/%s）一致。\n" \
        "$FILE_PASS" "$FILE_FAIL" "$PASS" "$FAIL"
else
    printf "\n⚠ 核对不一致：报告 ✓%s/✗%s vs 运行 %s/%s\n" \
        "$FILE_PASS" "$FILE_FAIL" "$PASS" "$FAIL"
    FAIL=$((FAIL+1))
fi
ln -sf "$(basename "$REPORT")" "$REPORT_DIR/latest.log"
printf "报告已存：%s（latest → test_reports/latest.log）\n" "$REPORT"

[ "$FAIL" -gt 0 ] && exit 1
[ "$PASS" -gt 0 ] && printf "全绿。🧾 每个答案都能对账。\n"
exit 0
