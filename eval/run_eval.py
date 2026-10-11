# 评测跑批脚本（交付物 B，Claude 编写）
# 遍历 eval/checks.py 的 18 问，复用 notebook 里的切块/TF-IDF/ask——测的就是 notebook 里的系统
#
# 判定与诊断：
#   must 全命中 + any 至少一命中  -> PASS
#   检索 top-3 是否覆盖 expect_docs -> 检索未覆盖诊断（不影响判定，只报告）
#   未命中的 must 三分类：块里有模型没写 / 库里有检索没捞到 / 库中无此数
#   幻觉数字扫描：回答中每个数字必须出现在检索块里，否则标记
#     （问题自带数字、检查点数字除外；[资料N] 标记和日期不参与扫描）
# 输出：eval/report_baseline.csv（UTF-8-SIG，Excel 可直接打开）+ 控制台汇总
#
# 运行：D:/jupyter_mysql/env/python.exe D:/coffee-rag/eval/run_eval.py

import json, re, csv, sys, io
from contextlib import redirect_stdout

sys.path.insert(0, 'D:/coffee-rag/eval')
from checks import CHECKS

TOP_K = 5  # v2 检索参数：喂给模型的块数（baseline 为 3；top_k 对照实验结论见评测报告）

# ---------- 1. 从 notebook 取用户写的系统 ----------
NB = 'D:/jupyter_mysql/notebooks/RAG文档切块代码.ipynb'
nb = json.load(open(NB, encoding='utf-8'))
ns = {'__name__': '__main__'}

# 两个 cell 尾部挂着检查点/三连测调用，exec 时会真的跑起来（白烧 API），按 cell id 精确截掉
CUTS = {
    '19af4eeb-ff0b-48c6-93c5-350e01315702': '# 检查点 1',
    '9d3d8c26-3fac-4926-8346-6d3497b15258': '# ================= 6. 执行检查点',
}
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    marker = CUTS.get(cell.get('id'))
    if marker and marker in src:
        src = src.split(marker)[0]
    exec(src, ns)

chunks = ns['chunks']
search, ask = ns['search'], ns['ask']
kb_all_text = '\n'.join(c['title'] + '\n' + c['body'] for c in chunks)


def doc_prefix(name):
    return name.split('_')[0]


def scan_numbers(answer, question, must, any_):
    """提取回答中需要扫描的数字：去掉 [资料N] 标记和日期，跳过问题自带与检查点数字"""
    text = re.sub(r'[\[【]资料\s*\d+[\]】]', '', answer)
    text = re.sub(r'\d{4}-\d{2}-\d{2}', '', text)
    nums = []
    for n in re.findall(r'\d+(?:\.\d+)?', text):
        if n in question:
            continue
        if any(n in s for s in must + any_):
            continue
        nums.append(n)
    return nums


def classify_number(n, retrieved_text):
    if n in retrieved_text:
        return None
    if n in kb_all_text:
        return f'{n}=库有此数但检索块未含'
    return f'{n}=库中无此数(计算值/疑似幻觉)'


# ---------- 2. 跑批 ----------
rows = []
for i, item in enumerate(CHECKS, 1):
    q, must, any_, expect = item['q'], item['must'], item['any'], item['expect_docs']
    print(f"[{i}/{len(CHECKS)}] {q[:38]}...", end=' ', flush=True)

    # 检索留痕：与 ask 内部同一条 search（TF-IDF 确定性一致，不重复调 API）
    top = search(q, TOP_K)
    hit_docs = [doc_prefix(chunks[idx]['doc_name']) for _, idx in top]
    retrieved_text = '\n'.join(chunks[idx]['title'] + '\n' + chunks[idx]['body'] for _, idx in top)
    trace = ' | '.join(f"{doc_prefix(chunks[idx]['doc_name'])}-{chunks[idx]['title'][:18]}({sim:.3f})"
                       for sim, idx in top)

    # 模型回答（吞掉 ask 的打印，结果进 CSV）
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            answer = ask(q, TOP_K)
    except Exception as e:
        answer = None
        buf.write(f'\nAPI 调用失败: {e}')

    if answer is None:
        verdict = 'ERROR' if 'API 调用失败' in buf.getvalue() else 'FAIL'
        note = 'API 调用失败' if verdict == 'ERROR' else '检索拒答(查询词全不在词表)'
        rows.append({
            '题号': i, '问题': q, '判定': verdict, 'must全命中': '-', '未命中must及分类': note,
            'any命中': '-', '检索覆盖': hit_docs, '检索缺失': expect,
            '疑似幻觉数字': '', '回答全文': '', '检索Top3': trace,
        })
        print(verdict)
        continue

    missing = [m for m in must if m not in answer]
    any_hit = (not any_) or any(s in answer for s in any_)
    retrieval_missing = [d for d in expect if d not in hit_docs]
    verdict = 'PASS' if (not missing and any_hit) else 'FAIL'

    classified = []
    for m in missing:
        if m in retrieved_text:
            classified.append(f'{m}:块里有,模型没写')
        elif m in kb_all_text:
            classified.append(f'{m}:库里有,检索没捞到')
        else:
            classified.append(f'{m}:库中无此数(计算值或检查点笔误)')
    note = '; '.join(classified)

    suspicious = [c for n in scan_numbers(answer, q, must, any_)
                  if (c := classify_number(n, retrieved_text))]

    rows.append({
        '题号': i, '问题': q, '判定': verdict,
        'must全命中': '是' if not missing else f'否(缺{len(missing)}项)',
        '未命中must及分类': note, 'any命中': '是' if any_hit else '否(可能改写)',
        '检索覆盖': ','.join(hit_docs), '检索缺失': ','.join(retrieval_missing) or '-',
        '疑似幻觉数字': '; '.join(suspicious), '回答全文': answer, '检索Top3': trace,
    })
    print(verdict)

# ---------- 3. 汇总 ----------
out = 'D:/coffee-rag/eval/report_baseline.csv'
with open(out, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

passes = sum(1 for r in rows if r['判定'] == 'PASS')
print('\n' + '=' * 60)
print(f'通过: {passes}/{len(rows)}')
print(f'报告: {out}')
print('-' * 60)
for r in rows:
    if r['判定'] != 'PASS':
        print(f"题{r['题号']} [{r['判定']}] {r['问题'][:26]}...")
        if r['未命中must及分类']:
            print(f"    {r['未命中must及分类']}")
        if r['any命中'].startswith('否'):
            print(f"    any未命中(回答改写,人工复核)")
        if r['检索缺失'] and r['检索缺失'] != '-':
            print(f"    检索缺失: {r['检索缺失']}")
        if r['疑似幻觉数字']:
            print(f"    幻觉扫描: {r['疑似幻觉数字']}")
