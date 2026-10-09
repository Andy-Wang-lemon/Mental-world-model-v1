#!/usr/bin/env python3
"""生成最外层《双人闭环测试运行说明》Word 文档（覆盖旧版）。"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

_HERE = Path(__file__).resolve()
_OUTER = _HERE.parents[3]
_OUT = _OUTER / "双人闭环测试运行说明.docx"


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
    para.paragraph_format.space_before = Pt(4)
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
    p(doc, "双人闭环测试运行说明", bold=True, size=16, center=True, space_after=4)
    p(
        doc,
        "给代跑同学：配置 DeepSeek、改 API Key、跑脚本、中断后续跑、打包结果",
        size=11,
        center=True,
        space_after=4,
    )
    p(
        doc,
        "对应引擎：感知 v18 / 认知 v11　·　2026-09-07",
        size=10,
        center=True,
        space_after=16,
    )

    heading(doc, "一、你要做什么", 1)
    p(
        doc,
        "请帮忙跑一次「陈默 × 林可」双人闭环测试。脚本会调用真实大模型（DeepSeek V4 Flash），"
        "按固定协议交替预测两人行动，并把每一步报告写到本地文件夹。跑完后把输出文件夹打包发回即可。",
    )
    p(
        doc,
        "不需要改代码。需要：Python 3.10 或以上、能访问 DeepSeek 的网络、一个有效的 API Key。"
        "整次测试 LLM 调用很多，建议预留 1～3 小时，中途不要关电脑、不要断网。"
        "如果中途断了，可以用 --resume 从断点续跑，不必从头再来。",
        bold=True,
    )
    p(
        doc,
        "引擎内部刚翻新过（Fresh/Historical 双路径 L2、alignment 硬门闸、L3 grounding、行为先去重再估概率），"
        "但闭环脚本的跑法、目录、参数没有变。不要改代码，按本文命令执行即可。",
    )

    heading(doc, "二、目录在哪", 1)
    p(
        doc,
        "你拿到的最外层文件夹里有 mental-world-model。所有命令都要在这个子目录里执行，不是最外层。",
    )
    code_block(
        doc,
        "心智世界模型（代码）/          ← 最外层（本文档也在这里）\n"
        "  双人闭环测试运行说明.docx\n"
        "  mental-world-model/         ← 请进入这里再操作\n"
        "    .env.example              ← 配置模板\n"
        "    .env                      ← 你要新建/修改这个（不要发给别人）\n"
        "    scripts/experiments/dual_agent_closed_loop.py   ← 要跑的脚本\n"
        "    data/陈默事件（负面版）.txt\n"
        "    data/林可负面事件.txt\n"
        "    reports/                  ← 结果会写到这里",
    )

    heading(doc, "三、环境准备（只需一次）", 1)
    heading(doc, "3.1 确认 Python", 2)
    p(doc, "终端里执行：")
    code_block(doc, "python3 --version")
    p(doc, "应显示 3.10 或更高。若命令不存在，先安装 Python 3。")

    heading(doc, "3.2 进入项目并安装依赖", 2)
    p(doc, "macOS / Linux：")
    code_block(
        doc,
        "cd \"mental-world-model\"\n"
        "python3 -m venv .venv\n"
        "source .venv/bin/activate\n"
        "pip install -e .",
    )
    p(doc, "Windows（PowerShell）：")
    code_block(
        doc,
        "cd mental-world-model\n"
        "python -m venv .venv\n"
        ".venv\\Scripts\\Activate.ps1\n"
        "pip install -e .",
    )
    p(
        doc,
        "以后每次重新开终端，都要先进入 mental-world-model，再激活虚拟环境（source .venv/bin/activate 或 .venv\\Scripts\\Activate.ps1）。",
    )

    heading(doc, "四、改 API Key 和模型（必做）", 1)
    p(
        doc,
        "密钥写在 mental-world-model/.env，脚本启动时会自动读取。不要把 Key 写进命令行历史里到处转发，也不要把 .env 发到微信群。",
    )

    heading(doc, "4.1 新建或打开 .env", 2)
    p(doc, "如果还没有 .env，从模板复制一份：")
    code_block(
        doc,
        "cd mental-world-model\n"
        "cp .env.example .env",
    )
    p(
        doc,
        "Windows 可用资源管理器：复制 .env.example，改名为 .env。然后用任意文本编辑器打开 .env。",
    )

    heading(doc, "4.2 填这三行（本次指定 DeepSeek V4 Flash）", 2)
    p(doc, "请改成下面这样（Key 换成你的；不要留 your-api-key-here）：")
    code_block(
        doc,
        "MWM_LLM_BASE_URL=https://api.deepseek.com\n"
        "MWM_LLM_MODEL=deepseek-v4-flash\n"
        "MWM_LLM_API_KEY=sk-这里换成你的密钥",
    )
    table(
        doc,
        ["变量", "本次取值", "说明"],
        [
            [
                "MWM_LLM_BASE_URL",
                "https://api.deepseek.com",
                "与仓库 .env.example 一致。若报 404，再改成 https://api.deepseek.com/v1 试一次。",
            ],
            [
                "MWM_LLM_MODEL",
                "deepseek-v4-flash",
                "模型名必须和平台控制台里的一致。不要写成 DeepSeek-V4-Flash 或 deepseek-chat（除非另有要求）。",
            ],
            [
                "MWM_LLM_API_KEY",
                "sk-…",
                "在 https://platform.deepseek.com 登录后，进入 API keys 创建。粘贴时不要加引号、不要多空格。",
            ],
        ],
    )
    p(
        doc,
        "改完务必保存文件。若你本机已经 export 过同名环境变量，会优先于 .env；冲突时先关掉终端重开，或在当前终端 unset 掉旧变量。",
    )

    heading(doc, "4.3 建议写上的运行参数（.env）", 2)
    p(
        doc,
        "这些不是命令行参数，而是写在 .env 里、影响每一次 LLM 调用的配置。正式跑请按下面填（或保持默认）。",
    )
    code_block(
        doc,
        "MWM_LLM_TEMPERATURE=0\n"
        "MWM_LLM_TOP_P=1\n"
        "MWM_LLM_SEED=42\n"
        "MWM_LLM_MAX_CONCURRENCY=160\n"
        "MWM_LLM_TIMEOUT=120\n"
        "MWM_LLM_PROGRESS=1",
    )
    table(
        doc,
        ["变量", "建议值", "什么时候改"],
        [
            [
                "MWM_LLM_MAX_CONCURRENCY",
                "160（默认，同一评分波全部心所）",
                "频繁 429 限流时改成 2 或 3，保存后用同一 -o 加 --resume 续跑。",
            ],
            [
                "MWM_LLM_TIMEOUT",
                "120",
                "经常超时可改成 180 或 240。改完同样用 --resume，不必重开新目录。",
            ],
            [
                "MWM_LLM_TEMPERATURE / TOP_P / SEED",
                "0 / 1 / 42",
                "正式测试不要改。保证可复现。",
            ],
            [
                "MWM_LLM_PROGRESS",
                "1",
                "终端会打 [MWM] 进度。设 0 会变安静，排障时不要关。",
            ],
        ],
    )

    heading(doc, "4.4 换 Key / 换模型时怎么改", 2)
    p(doc, "只改 .env 对应行，保存后再跑脚本，不必重装依赖。")
    p(doc, "换自己的 Key：只改 MWM_LLM_API_KEY。若正在跑到一半因 401 失败，改完 Key 后必须用 --resume，不要新开目录。")
    p(doc, "若必须换模型：只改 MWM_LLM_MODEL。本次请用 deepseek-v4-flash。")
    p(doc, "改完可用下面命令确认文件里是新值（不会在屏幕上打印完整 Key）：")
    code_block(doc, "grep '^MWM_LLM_' .env")

    heading(doc, "五、怎么跑脚本（完整命令）", 1)
    heading(doc, "5.1 正式测试（请跑这个）", 2)
    p(doc, "确认已激活虚拟环境，当前目录是 mental-world-model，然后：")
    code_block(
        doc,
        "python3 scripts/experiments/dual_agent_closed_loop.py -o reports/dual_loop_run",
    )
    p(doc, "Windows 若 python3 不可用，改用：")
    code_block(
        doc,
        "python scripts/experiments/dual_agent_closed_loop.py -o reports/dual_loop_run",
    )
    p(
        doc,
        "这是唯一需要的正式命令。-o 指定输出目录，后面续跑、打包都认这个路径。",
        bold=True,
    )
    p(
        doc,
        "不要加 --mock（假模型，结果不能当正式测试）。不要加 --rebuild-prologue（旧模式，H/M 规则不一样）。",
        bold=True,
    )
    p(
        doc,
        "终端里会出现 [MWM] 进度和 [S01_A1] 这类步骤日志。先做两人的前序预热（陈默 7 条、林可 8 条），再跑 14 步：A1→B1→…→B7。看到「汇总已写」才算成功。",
    )

    heading(doc, "5.2 命令行参数说明（代跑必看）", 2)
    table(
        doc,
        ["参数", "正式跑怎么用", "注意"],
        [
            [
                "-o / --output-dir",
                "必须写：-o reports/dual_loop_run",
                "输出目录。续跑必须与第一次完全相同。建议用相对路径，放在 mental-world-model/reports/ 下。不要放到桌面 iCloud / OneDrive 同步盘。",
            ],
            [
                "--resume",
                "中断后续跑才加",
                "必须同时带同一个 -o。目录里要有 progress.json。忘了加且目录已有进度时，脚本会直接报错退出，不会覆盖；这时补上 --resume 即可。",
            ],
            [
                "--mock",
                "正式跑不要加",
                "只用假模型验证脚本能启动。Mock 结果不用发回来。",
            ],
            [
                "--rebuild-prologue",
                "正式跑不要加",
                "旧行为：每步重跑前序并重置 H/M。与 --resume 互斥。",
            ],
            [
                "--chenmo-file / --linke-file",
                "不要改",
                "默认已指向 data/ 里的负面事件文件。不要换成正面版。",
            ],
            [
                "--definitions",
                "不要改",
                "定义目录，保持默认。",
            ],
        ],
    )
    p(
        doc,
        "没有 --resume 时：如果 -o 目录里已经有 progress.json，脚本会拒绝启动（FileExistsError），"
        "防止把半成品当新跑覆盖掉。正确做法是：同一目录加 --resume，或换一个新的 -o 目录重新开跑。",
        bold=True,
    )

    heading(doc, "5.3 可选：先用假模型确认脚本能启动", 2)
    p(doc, "不花钱、不调 API，只验证代码和数据文件在你电脑上能跑通：")
    code_block(
        doc,
        "python3 scripts/experiments/dual_agent_closed_loop.py --mock -o reports/dual_loop_mock_check",
    )
    p(doc, "Mock 通过后，再跑 5.1 的正式命令。Mock 结果不用发回来。不要把 mock 目录和正式目录混用。")

    heading(doc, "5.4 中途断了怎么续跑（重要）", 2)
    p(
        doc,
        "脚本支持从断点续跑。H/M 连续性账本和 progress.json 会一起读回来，不会清空。",
        bold=True,
    )
    p(doc, "续跑命令（-o 必须与第一次完全一样）：")
    code_block(
        doc,
        "python3 scripts/experiments/dual_agent_closed_loop.py -o reports/dual_loop_run --resume",
    )
    p(doc, "续跑前请确认：")
    p(doc, "1. 当前目录仍是 mental-world-model，虚拟环境已激活。")
    p(doc, "2. reports/dual_loop_run/progress.json 还在；runtime/continuity_chenmo.json 与 continuity_linke.json 还在。不要手动删这些文件。")
    p(doc, "3. 不要加 --rebuild-prologue。不要换一个新的 -o。")
    p(doc, "4. 若是 401（Key 无效），先改 .env 里的 Key 并保存，再执行上面的续跑命令。")
    p(doc, "5. 若是 429（限流），先把 .env 里 MWM_LLM_MAX_CONCURRENCY 改成 2 或 3 并保存，再续跑。")
    p(
        doc,
        "如果终端提示「输出目录已有 progress.json。续跑请加 --resume」：说明你忘了加 --resume，按 5.4 的命令再跑即可，目录是安全的。",
    )
    p(
        doc,
        "如果提示 continuity 为空或 event_count 对不上：不要自行清空 runtime。把整个 dual_loop_run 文件夹和终端报错发回。",
    )

    heading(doc, "5.5 输出目录不要放云同步盘", 2)
    p(
        doc,
        "不要把 -o 指到 iCloud Drive、坚果云、OneDrive 等正在同步的文件夹。"
        "同步过程可能写出半截 JSON，导致无法续跑。默认的 reports/dual_loop_run 在项目里即可。",
        bold=True,
    )

    heading(doc, "六、跑的是什么（方便你对照日志）", 1)
    p(doc, "两个独立模型实例：A=陈默，B=林可。各有自己的连续性账本。数据文件不用改。")
    table(
        doc,
        ["阶段", "发生什么"],
        [
            ["前序预热", "陈默事件1–7、林可事件1–8 当作已发生事件写入 H/M，只算一次"],
            ["S01–S05", "A1→B1→A2→B2→A3；A3 的预测不转给林可"],
            ["外部→林可", "脚本写入：发现陈默曾私下补数据"],
            ["S06–S10", "B3→A4→B4→A5→B5；B5 的预测不转给陈默"],
            ["外部→陈默", "脚本写入：CEO 要求 48 小时共交方案"],
            ["S11–S14", "A6→B6→A7→B7 结束"],
        ],
    )
    p(
        doc,
        "交叉给对方的是「观察者可见」的行动投影，不是完整内心独白。"
        "若 Top-1 是空预测（例如「（无行为预测）」），脚本会主动中止，这是正常保护；把终端报错原文和 dual_loop_run 发回即可，不要改脚本硬跑。",
    )
    p(
        doc,
        "步骤 JSON 里会多几组引擎诊断，打包时不用删，方便回看："
        "self/dharma_grasping 的 source_type（fresh / historical / accommodation / neutral）、"
        "所缘的 grounding_valid、cognition_candidates、accommodation、used_neutral_fallback、"
        "behavior_diagnostics（raw/distinct 条数、是否补多样性、是否坍缩）。"
        "若 candidate_collapsed 为 true 且概率为 1.0，表示只剩一种可观察动作，不是「模型很确信」。",
    )

    heading(doc, "七、跑完请发回这些文件", 1)
    p(doc, "正式跑成功后，核心目录是：")
    code_block(doc, "mental-world-model/reports/dual_loop_run/")
    p(doc, "请把整个 dual_loop_run 文件夹压缩打包发回。其中最重要的是：")
    table(
        doc,
        ["文件", "用途"],
        [
            ["dual_loop_summary.json", "14 步摘要、最终连续性、未转交的 A3/B5"],
            ["progress.json", "步骤进度；续跑依赖它"],
            ["final_yuanwen_chenmo.txt", "陈默侧最终事件账本"],
            ["final_yuanwen_linke.txt", "林可侧最终事件账本"],
            ["S01_A1_陈默.json 等 14 个步骤报告", "每步完整评估"],
            ["runtime/continuity_chenmo.json", "陈默连续性落盘"],
            ["runtime/continuity_linke.json", "林可连续性落盘"],
            ["abort.json（若有）", "中途中止原因；有的话请一并打包"],
        ],
    )
    p(
        doc,
        "另外请附带：① 你用的 MWM_LLM_MODEL 实际值（不要发 API Key）；② 大约开始/结束时间；③ 若有报错，把终端最后 50 行复制过来；④ 是否用过 --resume、改过并发/超时。",
    )

    heading(doc, "八、常见问题", 1)
    table(
        doc,
        ["现象", "处理"],
        [
            [
                "未设置 MWM_LLM_BASE_URL / MODEL",
                "没读到 .env。确认文件在 mental-world-model/.env（不是最外层），且当前就是在这个目录跑脚本。",
            ],
            [
                "401 / invalid api key",
                "先停。到 DeepSeek 控制台重新创建 Key，改 .env 并保存，然后用同一 -o 加 --resume。不要新开目录。",
            ],
            [
                "404",
                "BASE_URL 路径不对。先用 https://api.deepseek.com，不行再改 https://api.deepseek.com/v1。MODEL 名必须和平台一致。改完 --resume。",
            ],
            [
                "429 / 限流",
                "把 .env 里 MWM_LLM_MAX_CONCURRENCY 改为 2，保存后同一 -o 加 --resume。不要同时开多个脚本。",
            ],
            [
                "超时",
                "把 MWM_LLM_TIMEOUT 改为 180 或 240，保存后 --resume。检查网络是否能访问 api.deepseek.com。",
            ],
            [
                "输出目录已有 progress.json",
                "这是保护，不是坏了。补上 --resume，或换新的 -o 目录。不要删 progress.json 再当新跑（会和已有 continuity 对不上）。",
            ],
            [
                "--resume 需要已有进度文件",
                "你加了 --resume 但 -o 目录不对，或 progress.json 被删了。核对路径。",
            ],
            [
                "continuity 为空 / event_count 不符",
                "runtime 被清空或目录混用了。不要手动改 JSON。把整个输出目录打包发回。",
            ],
            [
                "ModuleNotFoundError: dotenv",
                "虚拟环境没激活，或没执行 pip install -e .。",
            ],
            [
                "闭环中止 / DualLoopAbort",
                "模型没给出可用行为。不要自行改脚本硬跑。把 dual_loop_run（含 abort.json）和终端日志发回。",
            ],
        ],
    )

    heading(doc, "九、请不要做的事", 1)
    p(doc, "1. 不要把 .env 或 API Key 发到聊天软件、邮件正文、截图里。")
    p(doc, "2. 不要改 data/ 里的事件原文，也不要改 scripts/experiments/dual_agent_closed_loop.py。")
    p(doc, "3. 不要用 --mock 当正式结果，不要加 --rebuild-prologue。")
    p(doc, "4. 不要在跑的过程中改模型名或乱改 .env（401/429 时按第八节改 Key/并发是可以的）。不要休眠电脑过久。")
    p(doc, "5. 不要提交 git commit / 不要 push。只跑脚本、打包 reports 即可。")
    p(doc, "6. 不要把输出目录放到 iCloud / 网盘同步路径。")
    p(doc, "7. 不要同时开两个脚本写同一个 -o 目录。")
    p(doc, "8. 中断后续跑时，不要换 -o，不要漏 --resume，不要删 progress.json 或 runtime。")

    heading(doc, "十、命令速查（可直接转发给代跑同学）", 1)
    code_block(
        doc,
        "# 1) 进入目录并激活环境\n"
        "cd mental-world-model\n"
        "source .venv/bin/activate          # Windows: .venv\\Scripts\\Activate.ps1\n"
        "\n"
        "# 2) 配置（首次）\n"
        "cp .env.example .env\n"
        "# 编辑 .env：\n"
        "#   MWM_LLM_BASE_URL=https://api.deepseek.com\n"
        "#   MWM_LLM_MODEL=deepseek-v4-flash\n"
        "#   MWM_LLM_API_KEY=sk-你的密钥\n"
        "#   MWM_LLM_MAX_CONCURRENCY=160\n"
        "#   MWM_LLM_TIMEOUT=120\n"
        "\n"
        "# 3) 正式跑（第一次）\n"
        "python3 scripts/experiments/dual_agent_closed_loop.py -o reports/dual_loop_run\n"
        "\n"
        "# 4) 中断后续跑（-o 必须相同）\n"
        "python3 scripts/experiments/dual_agent_closed_loop.py -o reports/dual_loop_run --resume\n"
        "\n"
        "# 5) 打包（macOS / Linux）\n"
        "cd reports && zip -r dual_loop_run.zip dual_loop_run\n"
        "\n"
        "# 不要加：--mock   --rebuild-prologue",
    )

    p(
        doc,
        "有问题先看第八节；仍卡住请把完整报错原文发回，不要只说「跑不通」。",
        space_after=4,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    print(f"已写入：{path}")


if __name__ == "__main__":
    build(_OUT)
