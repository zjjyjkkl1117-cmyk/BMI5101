# Four-model Results plotting / 四模型结果绘图

绘制 LR B、RF B + LASSO、GBM B + LASSO、XGBoost B 的校准、DCA、SHAP重要性、蜂群/瀑布解释、ROC、PR及RF校准比较图。

本目录仅发布代码。没有患者数据、真实逐人结局/概率、SHAP数组、训练模型或研究结果图片。`make_synthetic_demo.py` 完全通过随机数产生演示输入，不读取临床数据。演示图带有 **SYNTHETIC DEMO - NOT STUDY RESULTS** 水印，不能用于报告研究发现。

## 快速运行（Python 3）

```bash
cd four_model_plots
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python make_synthetic_demo.py --output-dir demo_data
python plot_all.py --data-dir demo_data --output-dir output
```

Windows 激活命令为 `.venv\Scripts\activate`。程序导出两份三页PDF、六张PNG和曲线汇总CSV，不联网、不训练模型、不拟合校准器，也不重新计算真实SHAP。

## 获授权成员使用真实数据

将真实输入保存在仓库之外的获授权本地目录，再使用 `--data-dir` 指向该目录。`--output-dir` 也建议指向仓库之外。不要将本地结果上传。

输入契约：

| 文件 | 字段/数组 | 用途 |
|---|---|---|
| predictions.csv | model, cohort, observed, p, row_index | 校准、DCA、ROC、PR |
| shap_importance.csv | model, feature, mean_abs_shap, scale, n | 每方案Top12重要性 |
| XGB_B_SHAP.npz | values, base, data, names, probability | XGBoost蜂群与瀑布 |
| RF_A_calibrated.csv / RF_B_calibrated.csv | observed, raw, isotonic, temperature | RF校准补充图 |

当前布局专用于训练994、测试426的研究设计；演示沿用相同形状，仅为测试布局。`cohort` 必须是 `Train apparent` 或 `Test`；`model` 是 `LR_B`、`RF_B`、`GBM_B_lasso`、`XGB_B`。`observed` 为0/1，`p` 在0到1之间。`row_index` 只是各方案各队列内的排序号，不是患者标识。

NPZ中 `values` 和 `data` 为426×特征数，`base`/`probability` 为426元素，`names` 为特征名。所有数组行顺序必须一致。真实输入应通过SHAP加和与预测概率核验。当前图需要至少15项特征。

RF真实SHAP需按原始字段汇总：逐人合并该字段对应编码列的贡献，再计算绝对值均值；单位为概率。其他模型当前采用编码特征级、log-odds贡献。面板长度不可直接跨模型比较。

## 方法与边界

- 校准按模型预测风险十分位分箱，Wilson 95%区间。
- DCA为未标准化净获益 TP/n − FP/n × t/(1−t)，展示阈值0.05至0.50。
- AP使用average precision，不是梯形积分PR-AUC。
- 训练面板是表观预测，不是OOF验证。
- 单例按中位预测风险选择，不按真实结局筛选。
- 演示SHAP/校准列仅是人工构造的绘图夹具，不来自拟合解释器或校准器，不可推断模型性能。
- 图例和样本数针对本项目固定；用于其他研究前需要修改并验证。

## 数据保护

不要提交原始数据、训练/测试患者名单、逐患者概率/结局、SHAP/编码特征矩阵、模型/检查点、单例临床图或含患者输出的笔记本。删除患者ID不意味着完全匿名。`.gitignore` 仅作辅助，不会删除已被跟踪的文件或历史提交。提交前应检查 `git diff --cached --stat` 及实际文件内容。

程序本身不自动发布图表。真实输出须按项目和机构的数据使用规则处理。
