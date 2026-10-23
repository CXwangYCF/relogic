# 来源与修改说明

- Auto-CoT 上游：https://github.com/amazon-science/auto-cot 。本包从用户当前工作区的修改版本提取；原 Apache-2.0 许可证及 NOTICE 保留于 `licenses/auto-cot/`。非上游原始发布版。
- BLEC 上游：https://github.com/chatc/BLEC 。源代码从当前工作区提取；保留原文说明 `BLEC-main/README.upstream.md`。当前本地 BLEC 副本没有附带 LICENSE 文件，本包不为这部分重新声明授权。
- reLogic / Snowball 数据来源：本地 `relogic-main`，对应论文 https://aclanthology.org/2021.findings-acl.388/ 。数据从当前工作区 `auto-cot-main/dataset/relogic_data` 提取，保留训练/测试划分，只删去本任务不读取的字段。保留本地 reLogic 副本的许可证于 `licenses/relogic/`；本包不为第三方数据重新声明授权，来源文件及导出文件 SHA-256 见清单。

相对当前工作区文件，本包的调整：

1. `run_inference.py` 新增可选 `--dataset_path`，允许沿用原入口读取训练集；两套任务提示词、答案提取、DataLoader、生成上限均保持原样。
2. `evaluate_blec.py` 将 BLEC 路径改为本仓库内的 `BLEC-main`，避免依赖仓库外的兄弟目录。
3. `run_demo.py` 修复 JSON/JSONL 读取，自动创建示例输出目录，候选不足或最终示例为空时报错；聚类与筛选规则保持原样。
4. 保留当前工作区的本地解码器及 BLEC 关键词大小写修改。新增快捷脚本、使用文档、校验清单和集成检查。

不包含个人评测方案文档及历史模型输出。
