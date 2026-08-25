# deployment/

Phase 1 重构范围明确不包含部署基础设施（不引入 Docker / docker-compose / K8s /
微服务拆分 —— 见重构任务要求第 6 条"不要为了企业级引入不必要的微服务"）。

这个目录当前只是占位符，保持与 `app/ agent/ knowledge/ mcp_service/ distill/
evaluation/ web/ tests/ configs/ docs/` 同级的目录骨架完整，便于后续阶段（若
确实需要私有化部署时）在这里加 `Dockerfile` / `docker-compose.yaml`，而不影响
当前项目的运行方式。

**当前项目的运行方式仍然是本地直接跑 Python 脚本 / uvicorn**，见根目录
README.md 的"如何运行"一节。
