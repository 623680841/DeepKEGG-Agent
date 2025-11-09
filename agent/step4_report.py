import os, glob, base64, pandas as pd
from datetime import datetime

# [EN] TODO: translate comment from Chinese.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(PROJECT_ROOT, "runs")
OUT_HTML = os.path.join(RUNS, "summary.html")
ALL_CSV = os.path.join(RUNS, "all_models_metrics.csv")


def b64img(path):
    """将图片文件转换为 Base64 编码的字符串"""
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")

def main():
    # [EN] TODO: translate comment from Chinese.
    # [EN] TODO: translate comment from Chinese.
    if os.path.exists(ALL_CSV):
        df = pd.read_csv(ALL_CSV)
        # [EN] TODO: translate comment from Chinese.
        cols_to_move_last = ["Timestamp", "Run Dir"]
        ordered_cols = [c for c in df.columns if c not in cols_to_move_last] + \
                       [c for c in cols_to_move_last if c in df.columns]
        df = df[ordered_cols]
    else:
        df = pd.DataFrame() # 创建一个空 DataFrame

    # [EN] TODO: translate comment from Chinese.
    # [EN] TODO: translate comment from Chinese.
    blocks = []
    if not df.empty and 'Run Dir' in df.columns:
        for index, row in df.iterrows():
            run_dir_name = row['Run Dir']
            run_dir_path = os.path.join(RUNS, run_dir_name)

            if not os.path.isdir(run_dir_path):
                print(f"[warn] Report: Directory not found for row {index}: {run_dir_path}")
                continue
            
            # [EN] TODO: translate comment from Chinese.
            title_parts = [
                row.get('Model', 'N/A'),
                row.get('Cancer', 'N/A'),
                row.get('LLM Provider', 'N/A')
            ]
            card_title = ' | '.join(map(str, filter(lambda x: x != 'N/A', title_parts)))
            
            roc64 = b64img(os.path.join(run_dir_path, "roc.png"))
            pr64  = b64img(os.path.join(run_dir_path, "pr.png"))
            
            blocks.append((card_title, run_dir_name, roc64, pr64))

    # [EN] TODO: translate comment from Chinese.
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # [EN] TODO: translate comment from Chinese.
    table_html = ""
    if not df.empty:
        # [EN] TODO: translate comment from Chinese.
        styler = df.style.format(precision=4).hide(axis="index")
        table_html = styler.to_html()
    else:
        table_html = "<p><i>No summary data found in 'all_models_metrics.csv'. Try running some models first.</i></p>"

    imgs_html = ""
    for title, d_name, roc64, pr64 in blocks:
        imgs_html += f"""
        <div class="card">
          <div class="card-title">{title}</div>
          <div class="card-subtitle">Run Directory: {d_name}</div>
          <div class="row">
            <div class="col">
              <div class="img-title">ROC Curve</div>
              {'<img src="'+roc64+'" />' if roc64 else '<div class="placeholder">roc.png not found</div>'}
            </div>
            <div class="col">
              <div class="img-title">PR Curve</div>
              {'<img src="'+pr64+'" />' if pr64 else '<div class="placeholder">pr.png not found</div>'}
            </div>
          </div>
        </div>
        """
    if not blocks:
        imgs_html = "<p><i>No model result images found. Check if 'Run Dir' in the CSV matches actual directories.</i></p>"

    # [EN] TODO: translate comment from Chinese.
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>DeepKEGG-Agent Summary Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 24px auto; max-width: 1200px; background-color: #f9fafb; color: #111827; }}
h1 {{ margin: 0 0 8px 0; font-size: 28px; }}
.subtitle {{ color:#6b7280; margin-bottom: 24px; }}
h2 {{ border-bottom: 1px solid #e5e7eb; padding-bottom: 8px; margin-top: 32px; font-size: 20px;}}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); background: white; }}
th, td {{ border: 1px solid #e5e7eb; padding: 10px 12px; text-align: left; }}
th {{ background: #f3f4f6; font-weight: 600; text-align: center; }}
td {{ text-align: center; }}
/* [EN] TODO: translate block comment from Chinese. */
.card {{ background-color: #fff; border:1px solid #e5e7eb; border-radius:12px; padding:16px; margin-bottom:18px; box-shadow:0 2px 6px rgba(0,0,0,0.04); }}
.card-title {{ font-size: 16px; font-weight:600; margin-bottom:4px; color: #1f2937; }}
.card-subtitle {{ font-size: 12px; color: #6b7280; margin-bottom: 12px; font-family: monospace; }}
.row {{ display:flex; gap:16px; }}
.col {{ flex:1; min-width: 0; }}
.img-title {{ font-size:14px; color:#4b5563; margin: 6px 0; text-align: center;}}
img {{ width:100%; height:auto; border:1px solid #d1d5db; border-radius:8px; }}
.placeholder {{ height:200px; border:1px dashed #d1d5db; border-radius:8px; display:flex; align-items:center; justify-content:center; color:#9ca3af; background-color: #f9fafb; }}
.footer {{ color:#9ca3af; font-size:12px; text-align: center; margin-top:32px; }}
</style>
</head>
<body>
  <h1>DeepKEGG-Agent &middot; 一页式报告</h1>
  <div class="subtitle">生成时间：{now} &middot; 数据源：{ALL_CSV}</div>

  <h2>汇总表 (Metrics Summary)</h2>
  {table_html}

  <h2>各模型运行曲线 (ROC / PR Curves)</h2>
  {imgs_html}

  <div class="footer">报告完全基于 `all_models_metrics.csv` 生成。每一行数据对应一组下方图片。</div>
</body>
</html>"""

    # [EN] TODO: translate comment from Chinese.
    os.makedirs(RUNS, exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ 报告已生成： file://{os.path.abspath(OUT_HTML)}")

if __name__ == "__main__":
    main()
