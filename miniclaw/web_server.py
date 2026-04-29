from __future__ import annotations

from .app import app, create_app


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    # 兼容旧启动方式；现在更推荐直接用 uvicorn miniclaw.app:app。
    import uvicorn

    uvicorn.run("miniclaw.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
