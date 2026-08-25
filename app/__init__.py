"""app/ —— 跨模块共享的应用核心。

Phase 1 重构新增。这里不放业务逻辑，只放所有模块都会用到的横切关注点：

- ``app.logging_config``：统一 logging 配置（替代原先散落各处的 ``print``）。
- ``app.exceptions``：统一异常体系（替代原先"裸 except / 返回错误字符串"的
  容错方式）。

业务代码（agent/、knowledge/、mcp/、distill/、evaluation/、web/）从这里导入，
而不是反过来 —— 避免循环依赖。
"""
