#!/usr/bin/env python3
"""生成《模型架构最新版》Word 文档（工作区最外层）。只写基础模型，不含测试脚本。"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


def _font(run, *, bold=False, size=11, mono=False, color=None):
    run.bold = bold
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor(*color)
    if mono:
        run.font.name = "Courier New"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "Courier New")
    else:
        run.font.name = "宋体"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")


def p(doc, text="", *, bold=False, size=11, space_after=8, center=False, mono=False):
    para = doc.add_paragraph()
    para.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    para.paragraph_format.space_after = Pt(space_after)
    para.paragraph_format.space_before = Pt(0)
    if center:
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if text:
        run = para.add_run(text)
        _font(run, bold=bold, size=size, mono=mono)
    return para


def heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = "黑体"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)


def formula(doc, text):
    para = p(doc, text, mono=True, size=11, space_after=10, center=True)
    para.paragraph_format.space_before = Pt(6)
    return para


def table(doc, headers, rows):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    t.autofit = True
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        _font(run, bold=True, size=10)
    for r_i, row in enumerate(rows, start=1):
        for c_i, val in enumerate(row):
            cell = t.rows[r_i].cells[c_i]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(val))
            _font(run, size=10)
    doc.add_paragraph().paragraph_format.space_after = Pt(8)


def code_block(doc, text):
    para = doc.add_paragraph()
    para.paragraph_format.line_spacing = 1.15
    para.paragraph_format.space_after = Pt(10)
    run = para.add_run(text)
    _font(run, mono=True, size=9, color=(0x22, 0x22, 0x22))


def build(path: Path) -> None:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.2)
    section.left_margin = Cm(2.4)
    section.right_margin = Cm(2.4)

    p(doc, "心智世界模型", bold=True, size=22, center=True, space_after=4)
    p(doc, "模型架构最新版", bold=True, size=16, center=True, space_after=4)
    p(
        doc,
        "基础模型：一次评估、一个分析对象、多认知分支",
        size=11,
        center=True,
        space_after=4,
    )
    p(
        doc,
        "主入口 MentalWorldModel.evaluate_from_yuanwen　·　感知 v21 / 认知 v14　·　扫描代码 2026-10-04",
        size=10,
        center=True,
        space_after=16,
    )

    heading(doc, "一、这份文档写什么", 1)
    p(
        doc,
        "本文只写基础模型：给定一篇原文和一个分析对象，系统如何建构其当下内心世界、"
        "如何为各主观所缘打 40 心所、如何在末事件预测行为，以及跨事件如何用认知偏置 H 与心理惯性 M 传递。"
        "公式与权重均按 2026-10-04 代码实现核对（perception/config.py、continuity/config.py、scoring/weights.py）。",
    )
    p(
        doc,
        "不写任何测试编排、双人闭环、观察者过滤或外部喂刺激。那些是模型外的用法，不改变引擎内部结构。",
    )
    p(
        doc,
        "工程边界：以「心 / 心所 / 我执 / 法执 / 所缘 / 想 / 思」为核心。"
        "不是通用心理学、ToM 或 BDI 框架。已删除心理状态粒子（persist_boost）。"
        "种子库 EMA 默认关闭，正文不展开。关闭多分支开关时会回退到旧单路提取，那不是主路径。",
    )

    heading(doc, "1.1 本次翻新相对旧架构改了什么", 2)
    p(doc, "未改：MentalEpisode / M 公式、H 时间衰减（仍不衰减）、刺激–行动隔离、apply_predicted_action、不删旧 H frame。常规所缘最多 3，扩支后最多 4。")
    table(
        doc,
        ["点", "现在怎么做"],
        [
            [
                "H 如何进入新事件",
                "不再把 H 塞进单一 L2 提示了事。Fresh L2 只看当前刺激；Historical L2 单独从相关 H 抽二执；合并后语义去重。同视角优先留 Fresh，不让 Historical 凭高 confidence 挤掉。",
            ],
            [
                "L1",
                "客观结构增加 role / relationship / constraint / resource 变化，禁止心理解释。",
            ],
            [
                "组合激活硬门闸",
                "alignment < 0.15 或 raw_score < 0.05 的 combo 不参与归一。禁止把唯一失败候选归一成 P(C)=1。",
            ],
            [
                "combo meta",
                "枚举只做一次；meta 与筛选共用同一批编号。未打分 combo 不进竞争，不得默认 alignment=1。LLM 全失败（空 meta）才全体保留默认分。",
            ],
            [
                "Accommodation",
                "有效 combo 为空，或最高 alignment < 0.25，或全部 L3 grounding 失败：fresh-only 重抽二执 1 次。再失败才 neutral。",
            ],
            [
                "L3 grounding",
                "所缘必须通过当前事件 grounding（未伪造事实且保留关键变化）；失败可按原组合重写一次，再失败淘汰。行动扩支同样跑 grounding。",
            ],
            [
                "行为",
                "先生成可观察且可区分的动作，再估概率。多个表面形式塌成同一动作时才补一次多样性；一开始就只给 1 条，视为情境确定，不强制凑多样性。沉默只在整句属于沉默集合时才合并。",
            ],
        ],
    )
    heading(doc, "1.2 相对 2026-09-19 扫描又改了什么", 2)
    p(
        doc,
        "感知提示词由 v18 升到 v21，认知提示词由 v11 升到 v14。"
        "门闸、所缘上限、H 不衰减、M 公式、刺激–行动隔离都没动。改的是二执怎么写，以及已绑定的 H 如何进入所缘和想。",
    )
    table(
        doc,
        ["点", "现在怎么做"],
        [
            [
                "我执",
                "不限于所属和身份。人、事、关系、位置，只要被放在「我这边」、不能当成与我无关，就是我执。",
            ],
            [
                "法执",
                "把人、事、规则、关系执成固定理解：理应如此、只能如此、本来如此。",
            ],
            [
                "二执要拆开",
                "不能互相替代的黏着或理解必须分成几条。一条只写一种。不要只留最醒目的那条。刺激极空洞才可以少写。",
            ],
            [
                "confidence",
                "表示当前刺激在多大程度上点燃这一条，不是这条听起来有多强、有多惨。几条可以同时高分，不因某一条更醒目就把其余压低。",
            ],
            [
                "L3 里的历史总执",
                "组合上已绑定的 reactivated_h 可以写进所缘，但不得另开执取维度。它只作为历史证据或熟悉感，强化当前组合已有的解释。不再判断这条历史是否相关。",
            ],
            [
                "想",
                "除所缘、二执和行动骨架外，直接读本支已绑定 H 的 id、正文和 mass。mass 越大，越把该命题当作既有评价惯性来组织想。不再另判相关度，也不另选 H，更不因此另起所缘。想不吃 prior / posterior。",
            ],
        ],
    )

    heading(doc, "二、模型在算什么", 1)
    p(
        doc,
        "一次评估只建模一个人。输入是原文（可含多件「事件1：…」）和分析对象。"
        "输出是此人在各事件上的主观所缘分支、支内心所、以及末事件的行为分布。"
        "连续性状态（H 与 M）留在模型实例上，供下一篇原文继续用。",
    )
    table(
        doc,
        ["概念", "在模型里是什么"],
        [
            ["分析对象 focus_person", "本次唯一内心主人；提示词禁止改写成他人"],
            ["客观刺激", "外界发生了什么（不含此人随后造作）；含角色/关系/约束/资源变化"],
            ["我执 / 法执", "与我分不开的黏着，以及对人事物的固定理解；来源 fresh / historical / accommodation / neutral"],
            ["主观所缘", "在某一组二执下，此事对他呈现为什么；一条所缘=一条认知分支。已绑定 H 只强化，不另开维度"],
            ["想", "该所缘下的认知评价；读本支已绑定 H 的正文和 mass，当作既有评价惯性"],
            ["思", "该事件的行动证据骨架；各支共享；无行动则空"],
            ["心所", "40 个心理因素在该支上的分数"],
            ["H 认知偏置", "跨事件保留的解释倾向（所缘+二执）；下一事件走 Historical L2，不直接改心所"],
            ["M 心理惯性", "跨事件保留的心所历史，掺进当前心所数字"],
        ],
    )

    heading(doc, "三、总体架构", 1)
    heading(doc, "3.1 分层", 2)
    code_block(
        doc,
        "原文 + focus_person\n"
        "        │\n"
        "        ▼\n"
        "  多事件切分（≥2 件时）\n"
        "  前序 1…N-1 = 解释事件（有行动）\n"
        "  末事件 N   = 预测事件（尚无己方行动）\n"
        "        │\n"
        "        ▼\n"
        "  感知 perception（刺激侧）\n"
        "    L0  刺激 / 行动拆分\n"
        "    L1  客观刺激结构（含结构变化）\n"
        "    L2  Fresh 二执  ∥  Historical 二执（相关 H）\n"
        "        统一 pool → 语义去重（同视角优先 Fresh）\n"
        "    L2.5 枚举组合 → combo meta（同一批编号）\n"
        "         alignment / raw_score 硬门闸 → 去同视角 → P(C)\n"
        "         全失败或最高 alignment 过低 → Accommodation（1 次）\n"
        "         再失败 → neutral\n"
        "    L3  主观所缘（prior = P(C)，常规最多 3 支）\n"
        "        已绑定 H 只强化当前解释，不另开执取维度\n"
        "        grounding（失败重写 1 次）→ 跨支所缘去重\n"
        "        │\n"
        "        ▼\n"
        "  认知 cognition（单事件）\n"
        "    解释事件：compat → 必要时扩 1 支（含 grounding）→ 旧支后验 → 搭支 → 想\n"
        "    预测事件：后验=先验，思空，只生成想，再预测行为\n"
        "    想读本支已绑定 H（id / 正文 / mass），不再另判相关度\n"
        "    思 = L0 行动骨架（各支共享）\n"
        "    各支 40 心所（四列评分 + 对治消解）\n"
        "    行为：先 distinct 动作 → 去重 → 必要时补多样性 → 再估概率\n"
        "        │\n"
        "        ▼\n"
        "  末事件落地 apply_predicted_action（主入口默认）\n"
        "    保留预测所缘，把 Top-1 接上去做 compat/扩支/想/心所\n"
        "    这时才写 H / M；调用方拿到的仍是预测结果\n"
        "        │\n"
        "        ▼\n"
        "  连续性 continuity\n"
        "    H：最终支直接登记（所缘+二执，mass=后验）→ 下一事件 Historical L2\n"
        "    M：episode 列表（存本事件跨支期望，不存掺历史后的分）\n"
        "        │\n"
        "        ▼\n"
        "  报告 EvaluationResult.to_event_report_dict()",
    )

    heading(doc, "3.2 模块职责", 2)
    table(
        doc,
        ["模块", "职责"],
        [
            ["definitions", "40 心所、40 对治、所缘/想/思定义"],
            ["perception.builder", "L0–L3 主路径编排"],
            ["perception.grasping_paths", "Fresh / Historical / Accommodation / neutral / 二执去重"],
            ["perception.combo_selection", "组合枚举、硬门闸、去同视角、P(C) 归一"],
            ["perception.grounding", "L3 当前事件 grounding 与重写"],
            ["perception.action_expand", "行动扩支子路径 L2'→combo'→L3'（含 grounding）"],
            ["cognition", "compat / L / 后验、扩支触发、想、思、心所、行为、预测落地"],
            ["continuity", "H（认知偏置）+ M（心理惯性 episode）"],
            ["scoring", "支内四列心所、对治消解、报告结构"],
            ["llm", "厂商无关 complete(messages, schema)；配置来自 .env"],
        ],
    )

    heading(doc, "3.3 核心契约", 2)
    p(doc, "1. 一人一心智。focus_person 全链路锁定。")
    p(
        doc,
        "2. 刺激与行动隔离。常规感知只用刺激文本；行动进入认知做 compat 与思。"
        "扩支时行动作为生成条件再进感知子路径，不再二次当后验似然。",
    )
    p(
        doc,
        "3. 两套概率各管一件事。"
        "a = 归一后的所缘 prior P(C)：跨支融心所、融行为都只用 a，不再乘支置信、不再乘 L。"
        "p' = 行动矫正后的 posterior：只作为 H 的 mass，表示「这件事之后还信这条解释多少」。"
        "故意分开：行为预测看的是当下怎么理解局面，不是已经被行动打过一遍的后验。",
    )
    p(doc, "4. 无心理粒子。跨事件心所只走 M，不给组合激活加 persist_boost。")
    p(doc, "5. 落库不存融合分。M episode 存本事件跨支期望 M_event，避免历史被二次加权。")
    p(
        doc,
        "6. 预测与落地对齐所缘。末事件先预测（不写 H/M）；可用的 Top-1 接到同一套所缘上再写 H/M，不重抽 L2/L3。"
        "「没表现 / 没有反应」是合法行动；系统没给出预测（空、或「（无行为预测）」等）则不写 H/M。",
    )
    p(
        doc,
        "7. H 只加不减。最终支直接登记，近义合并加重；不衰减、不按条数淘汰。"
        "注入下一事件时，按与当前刺激的相关度取前 5 条交给 Historical L2，而不是 mass 最高的 5 条。",
    )
    p(
        doc,
        "8. 硬门闸禁止假确定性。未过 alignment / raw_score 的 combo 不参与归一；"
        "不得把 alignment=0 的唯一候选归一成 P=1。",
    )
    p(
        doc,
        "9. Fresh 与 Historical 竞争，不按名额硬切。同视角去重优先保留 Fresh。"
        "neutral 只能最后 fallback，不能提前占支。",
    )
    p(
        doc,
        "10. 所缘必须 grounding 到当前事件。幻觉事实或丢掉关键结构变化的所缘，先重写一次再淘汰。",
    )

    heading(doc, "四、一次评估怎么走", 1)
    heading(doc, "4.1 锁定分析对象", 2)
    p(
        doc,
        "优先级：调用方传入的 focus_person ＞ 原文「分析对象：某某」。"
        "多事件全程同一人，写入 ContinuityState.person_id。",
    )

    heading(doc, "4.2 多事件路由", 2)
    p(
        doc,
        "原文能切出至少两件「事件1：」「事件2：」… 时进入循环。"
        "首个事件标签前的前言进入各事件 role_context。"
        "背景只带最近 3 条前序、各截 600 字，避免上下文无限膨胀。",
    )
    table(
        doc,
        ["事件", "行动反馈", "心所与连续性", "行为"],
        [
            [
                "前序 1…N-1（解释）",
                "开：拆行动、compat/L、必要时扩支、写思",
                "跨支融成 M_event，掺历史作报告；当时就落库 H/M（存 M_event）",
                "不做",
            ],
            [
                "末事件 N（预测）",
                "关：后验=先验，思空，不扩支",
                "各支掺同一 M_hist；预测本身不写 H/M",
                "支内并行预测，再跨支融合",
            ],
            [
                "末事件落地（主入口默认）",
                "把预测 Top-1 当作已发生行动接到原所缘上",
                "compat/扩支/想/心所后写入 H/M",
                "不再预测；返回给调用方的仍是上一步的行为分布",
            ],
        ],
    )
    p(
        doc,
        "只有一件事件时，整篇按末事件处理：先预测，再尝试落地。",
    )

    heading(doc, "五、感知层（L0–L3）", 1)
    p(
        doc,
        "感知链必须串行：每一步提示词吃上一步输出。常规路径不见主体随后造作。"
        "提示词版本：PERCEPTION_PROMPT_VERSION=v21。",
    )

    heading(doc, "L0 刺激 / 行动拆分", 2)
    p(
        doc,
        "把事件正文拆成 stimulus_text 与 action_span / action_description。"
        "明确写出的无反应、未行动也算行动证据，不是「没有行动」。"
        "后续感知只用刺激窗；行动骨架留给认知当思。",
    )

    heading(doc, "L1 客观刺激", 2)
    p(
        doc,
        "从刺激文本抽出客观结构，作为二执与所缘的事实底座，不是内心解释。"
        "字段：actors / action / object / scene / outcome_changes，"
        "以及结构变化 role_changes、relationship_changes、constraint_changes、resource_changes。"
        "禁止写主体反应、禁止心理评价。这些变化是 grounding 要保留的「关键变化」。",
    )

    heading(doc, "L2 我执 + 法执（Fresh / Historical）", 2)
    p(doc, "两条路径独立抽取，再合并。来源只影响去重优先级与报告，不预留固定名额。")
    table(
        doc,
        ["路径", "输入", "输出"],
        [
            [
                "Fresh",
                "当前刺激 + L1",
                "针对此刻局面的我执/法执；source_type=fresh；id 前缀 sg_f / dg_f",
            ],
            [
                "Historical",
                "当前刺激 + L1 + 相关 H（最多 5 条，按刺激相关度，并列 mass 高者优先）",
                "历史解释倾向在此刻如何抓取；source_type=historical；记下 source_bias_frame_ids",
            ],
        ],
    )
    p(
        doc,
        "合并后语义去重：弱规则近邻再可选 LLM 确认。同视角排序为"
        " Fresh < accommodation < mixed < historical < neutral，再比 -confidence。"
        "因此 Historical 即使 confidence 更高，也不会挤掉同一视角的 Fresh。",
    )
    p(
        doc,
        "我执：分析对象把什么当成与自己分不开的。不限于所属和身份；"
        "人、事、关系、位置只要放在「我这边」、不能当成与我无关，也是我执。",
    )
    p(
        doc,
        "法执：分析对象把人、事、规则、关系执成何种固定理解，即理应如此、只能如此、本来如此。",
    )
    p(
        doc,
        "不能互相替代的黏着或理解必须拆成几条，一条只写一种，不要只留最醒目的那条。"
        "confidence 表示当前刺激点燃这一条的程度，不是措辞有多强、有多惨；几条可以同时高分。",
    )
    p(
        doc,
        "H 不再作为「以后更可能如何理解」的提示词旁注塞进单一 L2；"
        "它必须变成可竞争的 Historical 二执，过 alignment 硬门闸后才能占支。",
    )

    heading(doc, "L2.5 组合激活", 2)
    p(doc, "先选过阈执取代表（confidence ≥ 0.25，我执/法执各最多 3 条，语义近的只留最强），再枚举组合。")
    p(
        doc,
        "枚举只做一次。combo meta（activation_alignment）直接打在这批候选上，"
        "筛选不再重新枚举，避免 C1/C2 错位。meta 非空时，未打分 combo 不进入竞争。",
    )
    p(doc, "组合激活分（无粒子加成）：")
    formula(doc, "score(C) = I_c × D_c × alignment")
    p(
        doc,
        "I_c、D_c 为我执/法执条目 confidence；缺一侧用 0.5 填充。"
        "alignment 由 combo meta 给出，问的是当前刺激是否点燃这个视角，不是这条组合有没有解释后来的行动。"
        "未打分且 meta 为空（LLM 全失败）时才用默认 1。",
    )
    p(doc, "硬门闸：只对过闸 combo 归一。")
    formula(doc, "过闸 ⇔ alignment ≥ 0.15 且 score ≥ 0.05")
    p(
        doc,
        "无人过闸则返回空列表，交给 Accommodation；禁止把唯一失败候选归一成 P(C)=1。",
    )
    p(
        doc,
        "去重：id Jaccard ≤ 0.5 视为过近；弱规则文本近邻（最短边 ≥4 字且 Jaccard≥0.55）后再可选语义确认。"
        "语义确认每批最多 12 对，多批问完，不截断。宁少勿滥，最多 3 支。"
        "次视角相对主组合强度须达到 0.5，否则丢掉。",
    )
    formula(doc, "P(C) = 按过闸 score 归一化")

    heading(doc, "Accommodation 与 neutral", 2)
    p(
        doc,
        "触发条件：有效 combo 为空，或保留组合的最高 alignment < 0.25，"
        "或全部 L3 grounding 失败（且尚未做过 Accommodation）。",
    )
    p(
        doc,
        "Accommodation：忽略 Historical，只按当前事件 fresh-only 重抽二执 1 次（ACCOMMODATION_MAX_RETRIES=1），"
        "再走组合与（必要时）L3/grounding。成功则 source_type=accommodation。",
    )
    p(
        doc,
        "仍无有效所缘时，才使用中性执取（「事件与主体相关，执取未明确」/「相关事物按字面理解」）。"
        "neutral 不得提前占支。",
    )

    heading(doc, "L3 主观所缘与 grounding", 2)
    p(doc, "每个保留组合渲染一条所缘正文。")
    formula(doc, "prior(O) = P(C)")
    p(
        doc,
        "不再与旧 Softmax(E/S/W) 融合。常规感知所缘上限 3；过阈才开，否则只留最高 1 支。"
        "行动扩支最多再加 1 支，总上限 4。",
    )
    p(
        doc,
        "grounding 校验两件事：hallucinated_facts（是否添加刺激中不存在的他人言行/事实；"
        "「理解为 / 当成」类主观解释不算伪造）；preserves_critical_event（是否完全忽略角色/约束/关系/结果等关键变化）。"
        "grounded = 未伪造 且 保留关键变化。失败则按原组合+L1 重写一次；再失败淘汰该支。",
    )
    p(
        doc,
        "通过 grounding 后，跨支所缘再语义去重：同一行为预测视角只留 prior 更高的一条，最多 3 条并重新归一 prior。",
    )
    p(
        doc,
        "渲染所缘时，组合上已绑定的 reactivated_h 一并给出。"
        "它不得产生新的执取维度，只允许作为历史证据或熟悉感，强化当前我执与法执已经给出的解释。"
        "不再判断这条历史是否相关。所缘的核心解释仍必须来自当前组合。",
    )

    heading(doc, "六、认知层", 1)
    p(doc, "提示词版本：COGNITION_PROMPT_VERSION=v14。")
    heading(doc, "6.1 解释事件（有行动）", 2)
    p(doc, "思 = L0 行动骨架，每事件一次、各支共享，不另开「思」模型调用。")
    p(doc, "顺序（compat 与想不再并行，因为扩支可能改变支集合）：")
    p(doc, "（1）compat：仅对已有所缘打行动贴合 raw∈[0,1]。")
    p(
        doc,
        "（2）若 max(compat)<0.45 且支数<4：走行动扩支——"
        "客观所缘+行动 → 新我法二执 → 组合（与 meta 共用同一批候选）→ 渲染 1 条所缘 → grounding。"
        "grounding 全失败则放弃扩支，不把未 grounding 的所缘加进去。",
    )
    p(doc, "（3）旧支用 L 更新后验；新支 posterior=prior，不打 compat（行动已作为生成条件）。")
    p(doc, "（4）新旧 posterior 混合归一后搭分支，再各支生成想。")
    p(
        doc,
        "想的输入是该支所缘、相关二执、行动骨架，以及本支已绑定 H 的 id、正文和 mass。"
        "这些 H 已经绑在本支上，生成想时只读，不再判断相关、不再另选。"
        "mass 越大，越把该命题当作既有评价惯性来组织想；H 不另起一条所缘。"
        "想不读取 prior 或 posterior。",
    )
    p(doc, "屏障：想就绪后才开 40 心所。")

    heading(doc, "6.2 预测事件（无行动）", 2)
    p(
        doc,
        "posterior = prior，compat=1，L=1，思为空，只生成想。预测事件不扩支。"
        "预测事件的想同样只读本支已绑定 H，不另判相关度。",
    )

    heading(doc, "6.3 后验更新", 2)
    p(doc, "中性点 compat=0.5。不对称门控 h：")
    formula(doc, "若 compat ≥ 0.5：  h = 0.5 + 0.5 a")
    formula(doc, "若 compat <  0.5：  h = 0.25 + 0.25 a")
    formula(doc, "L = 1 + h · (compat − 1)")
    formula(doc, "p' ∝ p · L     （再跨所缘归一化为 posterior）")
    p(
        doc,
        "合成 L 时 a 取 prior 的归一代理。compat 高时充分抬后验；compat 低时少压高激活视角，避免「打穿」。"
        "字段：action_compat=raw，action_likelihood=有效 L。仅 action_informed=false 的旧支参与此更新。",
    )

    heading(doc, "6.4 行动扩支新支", 2)
    p(
        doc,
        "新支 prior 取组合激活分，夹到 [0.08, 0.35]；posterior 先等于 prior，再与旧支已矫正 posterior 一起归一化。"
        "报告字段 action_informed=true。H 登记读各支二执 + 所缘 + posterior（不再经 LLM 摘要）。",
    )

    heading(doc, "6.5 支激活 a", 2)
    formula(doc, "a_raw = prior(O) = P(C)")
    formula(doc, "a = normalize(a_raw)     （失败支 raw=0）")
    p(doc, "支置信 = 所缘/想/思 confidence 的几何平均，只作诊断，不乘进 a。behavior_contribution ≡ a。")

    heading(doc, "6.6 跨支期望与行为", 2)
    formula(doc, "M_event(f) = Σ_b  a_b · s_b(f)")
    p(doc, "s_b(f) 为分支 b 对心所 f 的消解后分数。这是写入 history 的向量，尚未掺历史。")
    p(doc, "前序：用 M_event 与 M_hist 做线性混合得到报告用 M_final；落库仍是 M_event。不做行为预测。")
    p(doc, "预测：同一 M_hist、λ 写入每一支 s_b' = (1−λ)s_b + λ M_hist，再并行预测各支行为，最后只融合行为：")
    formula(doc, "P(A) = Σ_i  a_i · P(A | i)")
    p(
        doc,
        "支内行为两阶段：① 生成可观察候选并清洗；② 语义/规则去重成 distinct 动作后再估条件概率。"
        "支内保留条件概率 Top-3 并归一；跨支再取 Top-5 归一。预测不算 L。顶层不做跨支心所 final。",
    )
    p(
        doc,
        "多样性：仅当 raw 候选条数 ≥ 2 且去重后 < 2（多个表面形式塌成同一动作）时，才再要一次真正不同的行为。"
        "一开始就只生成 1 条，视为情境高度确定，不强制凑多样性；报告 candidate_collapsed=true，"
        "此时条件概率 1.0 只表示集合坍缩，不表示高确信。",
    )
    p(
        doc,
        "沉默合并：只有噪声剥离后整句属于「保持沉默 / 沉默 / 没有回应 / 不回应 / 未回应 / 没有说话 / 不说话 / 没有开口」等集合，"
        "才并入 canonical「保持沉默」。继续保持沉默（去掉「继续」）仍合并。"
        "「打破沉默发言」「沉默地点头」「沉默地递过文件」不得并入。"
        "行为文本由 LLM 批量清洗为可观察动作；无 LLM、清洗失败或洗成空句时保留原文。不按词表切句。",
    )

    heading(doc, "七、预测落地（基础模型主入口）", 1)
    p(
        doc,
        "evaluate_from_yuanwen 对末事件的约定：先出行为分布，再把已发生的 Top-1 接到预测时那套所缘上，"
        "这时才更新连续性。调用方拿到的评估结果仍是预测（含 aggregated_behaviors）；"
        "模型实例上的 H/M 来自落地。",
    )
    code_block(
        doc,
        "预测（update_continuity=False）\n"
        "  感知 L0–L3（不见己方行动）→ 想 → 心所 → 各支掺 M_hist → 行为分布\n"
        "        │\n"
        "        ▼\n"
        "取跨支融合 Top-1\n"
        "        │\n"
        "        ├─ 不可用（空 / 「（无行为预测）」等）→ 不写 H/M，结束\n"
        "        │\n"
        "        └─ 可用（含「没表现」「没有反应」）\n"
        "              deepcopy 预测感知（所缘留下）\n"
        "              注入行动证据（置信度=真实概率；仅 p≤0 时用 0.7）\n"
        "              → compat / 必要时扩支 / 后验 / 想\n"
        "              再打心所（按解释事件）→ 写 H/M",
    )
    p(
        doc,
        "关键：不把「刺激+行动」当新原文重抽 L2/L3。"
        "用来选出 Top-1 的内心画面，与写入 H/M 的所缘是同一套（扩支最多再加 1 支）。"
        "接口：MentalWorldModel.apply_predicted_action。",
    )

    heading(doc, "八、原理、公式与权重", 1)
    heading(doc, "8.1 心所四列合成", 2)
    table(
        doc,
        ["列", "符号", "基础权重 w", "含义"],
        [
            ["所缘境", "s1 / suoyuan", "0.35", "此事作为所缘是否能生起该心所。最高：无对象则心所无根。"],
            ["生起念头（想）", "s2 / shengqi_nian", "0.30", "该支「想」是否支持该心所。接近所缘，表示当下评价。"],
            ["业（思）", "s3 / ye", "0.10", "行动/思是否体现该心所造作。权重最低：预测事件思常为空，不能让业列主导。"],
            ["现代心理学", "s4 / xiandai_xinli", "0.25", "辅助列：对照现代心理描述，不压过所缘与想。"],
        ],
    )
    p(doc, "来源调节 m_i 在分支体制下恒为 1，故归一化权重 ŵ_i = w_i。种子库关闭时不再乘 W_seed。")
    formula(doc, "final = Σ_i  ŵ_i · s_i")
    p(
        doc,
        "所缘境按 13 组合并 + 8 单独批量评分（波内并行）；后三列对过关心所并行。"
        "忿→恨→恼链式依赖仍保留。",
    )

    heading(doc, "8.2 门控", 2)
    table(
        doc,
        ["规则", "阈值", "效果", "为什么"],
        [
            [
                "所缘硬地板 SUOYUAN_HARD_FLOOR",
                "0.10",
                "所缘分 < 0.1：直接出局，不打后三列",
                "所缘完全对不上，后面三列再高也是空转。",
            ],
            [
                "所缘+想双门控 GATE_THRESHOLD",
                "0.30",
                "所缘与生起念头都 < 0.3 才排除；否则继续评业与现代心理",
                "只靠所缘或只靠想不够判死。两边都弱才出局。",
            ],
            [
                "业 / 现代心理",
                "gate=False",
                "过双门控后两列全评，单列不再出局",
                "辅助列不应单独否决已经过关的心所。",
            ],
            [
                "对治分差 _DEFAULT_SCORE_MARGIN",
                "0.08",
                "Δ≥0.08 代码取高分；近平局才问 LLM",
                "明显一强一弱不必调模型；接近才需要语义裁决。",
            ],
        ],
    )

    heading(doc, "8.3 对治消解", 2)
    p(
        doc,
        "互斥心所成对检测。比分差 Δ ≥ 0.08 时取高分者，不调模型；近平局才裁决 winner。"
        "循环直到无冲突。消解在支内完成。",
    )

    heading(doc, "8.4 感知、组合、行动：每一个数为什么这么取", 2)
    p(doc, "下面按代码标识符列出。主路径在用的才算有效权重；标「旧/关」的仍留在代码里，当前不参与计算。")
    table(
        doc,
        ["常量", "值", "用在哪", "为什么这么设计"],
        [
            ["MAX_ALAMBANA_BRANCHES", "3", "常规感知最多留几条所缘", "同时认真抓取的视角很少。3 支够表达冲突，再多会把概率摊薄。"],
            ["ALAMBANA_MIN_PROBABILITY", "0.08", "过阈才开支；组合主路径主要作告警", "低于约 1/12 的视角几乎不会驱动行为。"],
            ["ALAMBANA_MIN_CONFIDENCE", "0.20", "兼容 Softmax 路径删支", "主路径 prior=P(C)，这条主要给无组合旧路用。"],
            ["ALAMBANA_SCORE_W_EVENT/SELF/WORLD", "0.4 / 0.3 / 0.3", "仅无组合时 Softmax", "事件贴合略重于「我」和「世界」。主路径已不用 Softmax。"],
            ["ALAMBANA_SOFTMAX_TEMPERATURE", "1.0", "兼容路径 Softmax T", "不额外锐化或抹平。"],
            ["COMBO_PRIOR_BLEND_ALPHA", "0.7（已不计算）", "旧：P(C) 与 Softmax 融合", "主路径改成 prior=P(C)，避免两套先验互相稀释。"],
            ["GRASPING_COMBO_ACTIVATION", "0.25", "执取 confidence 达此才进组合", "滤掉弱猜测，又不会只剩一条独断视角。"],
            ["COMBO_MAX_SELF/DHARMA_REPS", "各 3", "我执、法执代表上限", "各 3 条则组合最多 9 个，再交给门闸和去重。"],
            ["score(C)=I_c×D_c×alignment", "缺一侧填 0.5", "组合原始激活", "两侧都强且贴事件才算真激活。缺一侧用 0.5，避免单执归零。"],
            ["MIN_ALIGNMENT", "0.15", "combo 硬门闸：事件贴合下限", "历史执取再强，对不上当前事件也不能占支。禁止把 alignment=0 归一成 P=1。"],
            ["MIN_COMBO_RAW_SCORE", "0.05", "combo 硬门闸：原始分下限", "乘积太小说明两侧执取或贴合至少有一项几乎没了。"],
            ["COMBO_SECONDARY_RATIO", "0.50", "次组合须达到主组合一半强度", "真冲突通常强度接近，不是差一个数量级的回声。"],
            ["COMBO_ID_JACCARD_MAX", "0.50", "组合 id 过近则去同视角", "同一组执取换个说法不算新支。"],
            ["SEMANTIC_RECALL_MIN_LEN / JACCARD", "4 / 0.55", "弱规则近邻召回，过闸才问 LLM", "宜松：宁多召回，由 LLM 否决反义。短边至少 4 字。"],
            ["SEMANTIC_CONFIRM_MAX_PAIRS", "12", "语义确认每批对数；多批问完", "控单次成本，但不截断漏问。"],
            ["ACCOMMODATION_MIN_FIT", "0.25", "最高 alignment 低于此则 fresh-only 重抽", "比硬门闸 0.15 更严：过闸但仍整体贴不上，就承认旧 H 失效。"],
            ["ACCOMMODATION_MAX_RETRIES", "1", "只重抽一次", "再抽容易编。然后才允许 neutral。"],
            ["neutral 执 / 所缘 conf", "0.4 / 0.35", "最后兜底中性执取", "故意偏低：能出报告，但不能装成高确信。不得提前占支。"],
            ["ALAMBANA_DEFAULT_CONFIDENCE", "0.5", "所缘缺 confidence 时", "中性默认，不再用 E/S/W 拼假信心。"],
            ["ALAMBANA_VALIDATION_MAX_RETRIES", "1", "L3 grounding 失败后重写一次", "给一次纠幻觉机会；再失败淘汰。"],
            ["ACTION_COMPAT_NEUTRAL", "0.50", "L 的支持/矛盾分界", "0.5 表示说得通也说不通。"],
            ["h_up = 0.5+0.5a", "支持侧", "compat≥0.5 时充分抬后验", "高激活且行动对得上，后验应明显抬升。"],
            ["h_down = 0.25+0.25a", "矛盾侧", "compat<0.5 时少压", "表里不一不等于视角死了，避免一次反常行动打穿。"],
            ["ACTION_EXPAND_MAX_COMPAT", "0.45", "低于此且支数<4 才扩支", "旧所缘都解释不好这个行动，才允许行动条件下长新所缘。"],
            ["ACTION_EXPAND_MAX_BRANCHES", "4", "含扩支后总上限", "常规 3 支再加 1 条行动生成支。"],
            ["ACTION_EXPAND_MIN/MAX_NEW_PRIOR", "0.08 / 0.35", "新支 prior 夹紧", "下限避幽灵支；上限避免新支一口吃掉全部质量。新支不再二次打 L。"],
            ["land 默认行动置信", "0.7", "概率≤0 时写入行动证据", "必须非零才能落地；有真实 Top-1 概率时用真实值。"],
            ["BEHAVIOR_MIN_DISTINCT / RETRIES", "2 / 1", "多条表面形式塌成同一动作时才再要一次", "一开始只给 1 条视为情境确定。此时 p=1.0 只表示集合坍缩。"],
            ["BRANCH / AGGREGATED ACTION TOP_K", "3 / 5", "支内 / 跨支行为条数", "支内 3 条对齐「最多 3 个可观察行动」；跨支 5 条给融合留余量。"],
            ["MENTAL_FACTOR_PRESENT_THRESHOLD", "0.30", "score≥0.3 才计入在场", "和双门控同一量级。"],
        ],
    )

    heading(doc, "九、连续更新机制", 1)
    p(doc, "两套独立机制：H 管「以后怎么理解」，M 管「心所数字如何被历史拖住」。二者都可单独开关，默认都开。")

    heading(doc, "9.1 认知偏置 H", 2)
    p(
        doc,
        "每条 BiasFrame：text（解释倾向）+ mass∈[0,1]。"
        "正文不是 LLM 摘要，而是该最终支的主观所缘 + 我执 + 法执；mass = 该支后验。"
        "不衰减、不按 mass 下限淘汰、不限制条数。近义则合并加重，不同维度全部保留。",
    )
    p(doc, "事件结束后（解释事件当时写；预测事件在落地时写）：")
    p(doc, "（1）每个最终保留的所缘支直接登记为一条 incoming frame。")
    p(doc, "（2）与已有帧合并：弱规则召回 → 批量语义确认（每批 12 对，问完所有对）→ 近义则强化，否则追加。强化：")
    formula(doc, "mass ← mass + 0.35 × (1 − mass) × mass_incoming")
    p(doc, "若 incoming 文本更长，用更长的覆盖。合并后按 mass 重新编号 id。")
    p(
        doc,
        "注入下一事件：对全部 H 按与当前刺激的相关度排序，取前 5 条交给 Historical L2；"
        "并列时 mass 高者优先。H 不改下一事件的所缘公式，也不直接写入 Fresh L2。",
    )
    p(
        doc,
        "旧常量 COGNITIVE_MAX_FRAMES=8、COGNITIVE_DECAY_DELTA=0.08、COGNITIVE_MIN_MASS=0.15"
        "仅保留给测试/文档引用，H 更新路径不再用它们删条。",
    )

    heading(doc, "9.2 心理惯性 M", 2)
    p(doc, "状态里是 MentalEpisode 列表，最多 20 条（超出丢最旧）。每条存：")
    table(
        doc,
        ["字段", "含义"],
        [
            ["scores", "该事件跨支期望 M_event（未掺历史）"],
            ["theta θ", "该事件对主体的重要性 ∈[0,1]"],
            ["gap_from_previous_days", "相对上一事件的叙事间隔（天）；首事件或未启用时间维为 0"],
            ["timestamp / text_summary", "落库与 meta 提示"],
        ],
    )

    heading(doc, "元数据", 3)
    p(
        doc,
        "每事件最多一次抽取：time_dimension_applies、delta_t_days、theta、relevance R。"
        "resolve_mental_history 已写入这些键时，落库不再二次抽取。"
        "默认不算时间维：只有原文明确写出间隔或可明显推断「不是同一天」才打开。打开后 gap 才非 0。",
    )

    heading(doc, "历史加权 M_hist", 3)
    formula(doc, "τ(θ) = 3 + 57·θ     （天，约 3～60）")
    formula(doc, "w_i = θ_i · exp( − age_i / τ(θ_i) )")
    p(doc, "未启用时间维：全体 age=0，故 w_i = θ_i。启用时间维：最近一条 age = 当前 Δt；更早的 age = 后面各段 gap 累加。")
    formula(doc, "M_hist(f) = ( Σ_i w_i · scores_i(f) ) / ( Σ_i w_i )")
    p(doc, "缺的心所按 0。")

    heading(doc, "融合强度 λ", 3)
    formula(doc, "λ = min( 0.50 ,  0.30 · clip(Σw, 0, 1) · R )")
    p(doc, "R 为当前事件与历史心理残留的情境相关度。λ 封顶 0.5，当前事件至少占一半。")

    heading(doc, "掺法：前序 vs 预测", 3)
    formula(doc, "(1−λ)·当前 + λ·M_hist")
    p(doc, "前序：对跨支向量 M_event 做混合，得到报告用 scores_final；落库仍是 M_event。")
    p(
        doc,
        "预测：对每一支 s_b 做同一混合，得到支级 final，再预测行为；"
        "顶层 scores_final 仅等于 event（兼容），报告标记 mental_report_mode=per_branch_only。",
    )
    p(
        doc,
        "线性意义下「先跨支再掺」与「先支级掺再按 a 加权」对期望向量等价；"
        "预测事件故意不塌缩心所，因为行为要按支、按多种候选来。",
    )

    heading(doc, "9.3 三事件数值直觉", 2)
    p(doc, "时间维关闭（常见同场叙事）：")
    p(doc, "E1：无历史，λ=0，报告等于 M_event¹，落地后落库 (θ₁, M_event¹)，并按最终支登记 H。")
    p(doc, "E2 若为前序：M_hist=M_event¹，λ=0.3·clip(θ₁)·R₂，报告混合、落库 M_event²；H 近义加重或追加。L2 走 Fresh∥Historical。")
    p(doc, "E3 若为预测：M_hist 为已落库事件按 θ 加权；Historical L2 吃与当前刺激相关的 H；先掺进各支再预测；落地时再按带行动的解释事件写新 episode 与 H。")
    p(doc, "若打开时间维且原文有「三天后」：旧事件权重被 exp(−age/τ) 压低。H 仍不随时间衰减。")

    heading(doc, "9.4 与已删除机制的边界", 2)
    p(doc, "心理粒子已删除：不再用上一事件 a_prev 做 persist_boost=1+0.5·a_prev。")
    p(doc, "M 不改下一事件感知（不注入所缘/二执）。H 不负责心所向量；H 通过 Historical L2 竞争，不过闸就不能占支。")
    p(doc, "H 不再 LLM 压缩成短摘要，不再衰减，不再丢低 mass / 超 8 条。")

    heading(doc, "十、报告怎么读", 1)
    table(
        doc,
        ["字段", "前序事件", "预测事件（返回值）"],
        [
            ["mental_factors_event_nonzero", "跨支期望（落库）", "跨支期望；不是行为输入"],
            ["mental_factors_final_nonzero", "掺历史后的跨支心所", "无（per_branch_only）"],
            ["branches[].mental_factors_nonzero", "支内未掺历史", "支内已掺 M_hist"],
            ["aggregated_behaviors", "无", "跨支行为分布"],
            ["active_xinsuo", "可取自期望", "None"],
            ["self/dharma_grasping[].source_type", "fresh / historical / accommodation / neutral / mixed", "同左"],
            ["subjective_alambanas[].grounding_valid", "当前事件 grounding 是否通过", "同左"],
            ["subjective_alambanas[].action_informed", "true=扩支新支，跳过 L", "恒 false（预测不扩支）"],
            ["cognition_candidates", "Fresh/Historical 条数、去重后、过闸 combo 数", "同左"],
            ["accommodation / used_neutral_fallback", "是否触发重抽 / 是否落到中性执", "同左"],
            ["behavior_diagnostics", "无", "raw/distinct 条数、是否补多样性、是否坍缩"],
            ["continuity.apply_mode", "cross_branch_event", "per_branch_prediction"],
            ["continuity.landed", "无", "主入口落地成功则为 true"],
        ],
    )
    p(
        doc,
        "默认 CLI 输出 to_event_report_dict()。"
        "--verbose 才展开完整调试树。"
        "落地写入的 H/M 在模型 continuity_state 上；预测报告里的分支仍是预测当下的所缘与心所。"
        "candidate_collapsed=true 且条件概率=1.0 时，读成「只剩一种可观察动作」，不要读成高确信。",
    )

    heading(doc, "十一、入口与配置", 1)
    p(doc, "入口：")
    code_block(
        doc,
        "python3 -m mental_world_model.cli evaluate \\\n"
        "  --focus-person 分析对象姓名 \\\n"
        "  --file 原文.txt \\\n"
        "  -o reports/out.json",
    )
    p(doc, "LLM：.env 中 MWM_LLM_BASE_URL / MWM_LLM_MODEL / MWM_LLM_API_KEY。不加 --mock 即真实接口。")
    p(
        doc,
        "并发：默认 MWM_LLM_MAX_CONCURRENCY=160（同一评分波内全部心所；四列仍分波）。"
        "心所与想、预测事件各支行为可并行；感知 L0–L3 串行；Fresh 与 Historical L2 可并行；"
        "行动扩支子路径在 compat 之后串行插入。",
    )
    p(doc, "单事件评估也可直接调用 evaluate_text_with_cognition_branches（可关 update_continuity），再按需 apply_predicted_action。")
    p(doc, "提示词版本：感知 v21，认知 v14。改提示词时同步升版本号。")

    heading(doc, "十二、常量速查（连续性、编排、种子库）", 1)
    table(
        doc,
        ["常量", "值", "含义", "为什么"],
        [
            ["COGNITIVE_TOP_K_INJECT", "5", "交给 Historical L2 的 H 条数（按当前刺激相关度）", "只注入此刻相关的偏置，不是 mass 最高的 5 条，避免无关旧执取抢支。"],
            ["COGNITIVE_REINFORCE_ALPHA", "0.35", "近义合并：m←m+0.35(1−m)m_new", "加重但不封顶到 1；同一解释反复出现会变沉，不同维度全部保留。"],
            ["COGNITIVE_MAX_FRAMES / DECAY / MIN_MASS", "8 / 0.08 / 0.15（H 更新不用）", "旧：条数帽、时间衰减、低 mass 淘汰", "现路径 H 只加不减，这些仅留测试引用。"],
            ["MENTAL_MAX_EPISODES", "20", "M 历史条数", "超出丢最旧。心所惯性不需要无限长记忆。"],
            ["MENTAL_LAMBDA0", "0.30", "λ 基数：λ=0.30·clip(Σw)·R", "历史能拖住当前，但不能反过来主导。"],
            ["MENTAL_LAMBDA_MAX", "0.50", "λ 上限", "当前事件至少占一半。"],
            ["MENTAL_TAU_MIN/MAX_DAYS", "3 / 60", "τ(θ)=3+57θ（天）", "只作用于 M。重要事件衰减慢，琐事快。默认时间维关闭则 age=0、w=θ。"],
            ["_MAX_HIST_EVENTS / _MAX_BODY_CHARS", "3 / 600", "多事件 role_context 只带最近 3 条、各截 600 字", "背景备忘，不是 H/M。避免上下文无限膨胀。"],
            ["SEED_LIBRARY_ENABLED", "False", "种子库 / Top-6 路由 / EMA 默认关", "主路径用 40 心所全评 + 分支 a，不再用画像路由筛心所。"],
            ["ROUTER_RELEVANCE_ALPHA / TOP_K / MIN", "0.75 / 6 / 0.35（仅种子库开）", "combined=0.75·相关度+0.25·W_seed，取 Top-6", "开种子库时才用。语义相关为主，画像为辅。"],
            ["SEED_EMA_ALPHA / ZERO_DECAY", "0.3 / 0.05（仅种子库开）", "W←(1-α)W+α·score；零分缓慢衰减", "开种子库时的画像更新，不乘进心所 final。"],
        ],
    )
    p(
        doc,
        "默认开关：认知分支开、H 开（不衰减）、M 开、种子库关。"
        "心理粒子 persist_boost 已删除。H 管「以后怎么理解」，M 管「心所数字如何被历史拖住」。",
    )

    p(doc, "—— 完 ——", center=True, size=10, space_after=0)

    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    out = root / "模型最新架构.docx"
    build(out)
    print(out)


if __name__ == "__main__":
    main()
