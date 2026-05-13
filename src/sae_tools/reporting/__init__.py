"""HTML reporting and read-only dashboard service for SAE tools."""

__all__ = ["ReportServerConfig", "serve_report"]


def __getattr__(name: str):
    if name in __all__:
        from .server import ReportServerConfig, serve_report

        return {"ReportServerConfig": ReportServerConfig, "serve_report": serve_report}[name]
    raise AttributeError(name)
