"""HTML reporting and read-only dashboard service for SAE tools."""

__all__ = ["ArtifactReportResult", "ReportServerConfig", "generate_artifact_report", "serve_report"]


def __getattr__(name: str):
    if name in __all__:
        if name in {"ArtifactReportResult", "generate_artifact_report"}:
            from .artifact_report import ArtifactReportResult, generate_artifact_report

            return {
                "ArtifactReportResult": ArtifactReportResult,
                "generate_artifact_report": generate_artifact_report,
            }[name]
        from .server import ReportServerConfig, serve_report

        return {"ReportServerConfig": ReportServerConfig, "serve_report": serve_report}[name]
    raise AttributeError(name)
