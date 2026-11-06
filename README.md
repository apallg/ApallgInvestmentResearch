# AShareSignalAdvisor（Apallg 投研 V2）

A 股研究与交易辅助项目。前端采用 Vue 3、Vite 和 ECharts；后端采用 FastAPI，包含行情采集、数据质量检查、因子研究、选股、回测、组合与风险管理等模块。项目仍在持续开发，研究结果和回测表现不构成实盘收益保证。

## 目录

| 路径 | 内容 |
| --- | --- |
| `frontend/` | Vue 页面、组件与前端测试 |
| `backend/api/` | HTTP API |
| `backend/service/` | 用例编排 |
| `core/domain/` | 领域计算与回测逻辑 |
| `core/data/`、`core/adapters/` | 数据访问与外部系统适配 |
| `db/` | 数据库初始化 |
| `scripts/`、`tests/` | 运维脚本与测试 |

## 本地运行

建议使用 Windows 11、Python 3.11、Node.js、MySQL 8。部分数据采集和模型功能还依赖相应的数据源凭证及本地历史数据。

```powershell
python -m pip install -r requirements.txt
cd frontend
npm ci
cd ..
Copy-Item .env.example .env
```

编辑 `.env`，至少设置可用的 `MYSQL_PASSWORD` 和 `ADMIN_TOKEN`；需要外部数据、LLM 或通知功能时再填写对应凭证。初始化数据库：

```powershell
python db/setup.py
```

分别启动后端和前端：

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
python -m backend.main
```

```powershell
cd frontend
npm run dev
```

访问 `http://localhost:5174`，后端监听 `http://localhost:8001`。仓库中的 `start.bat` 和 `_run_backend.bat` 含原开发机的 Python 绝对路径，换机器使用前需修改。

## 验证

```powershell
pytest tests/ -m unit -v
pytest tests/ -m p0 -v
cd frontend
npm run test
npm run build
```

修改 `core/domain/` 后运行 `python scripts/check_l3_purity.py`。

## 数据与配置

仓库只包含源代码、测试、基础配置和少量 `data/optuna_params/` 参数文件。历史行情、Qlib 数据、模型产物、MLflow 记录、实验输出、日志和本机 `.env` 均不随代码发布。首次运行前需要自行配置数据源并完成数据采集；缺少历史数据时，相关研究、回测或预测功能可能不可用。

默认券商类型为 `fake`。涉及真实账户或交易前，应独立核对数据时点、复权口径、回测与下单约束，并完成自己的环境验收。
