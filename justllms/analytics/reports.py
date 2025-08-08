"""Report export functionality for analytics dashboard."""

import csv
import io
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Union

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import pandas as pd

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    pd = None

from justllms.analytics.models import AnalyticsReport


class ReportExporter(ABC):
    """Abstract base class for report exporters."""

    @abstractmethod
    def export(
        self, report: AnalyticsReport, output_path: Optional[Union[str, Path]] = None
    ) -> Union[str, bytes]:
        """Export report to specified format."""
        pass


class CSVExporter(ReportExporter):
    """Export analytics reports to CSV format."""

    def export(
        self, report: AnalyticsReport, output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """Export report to CSV format."""
        csv_data = []

        # Summary section
        csv_data.append(["=== ANALYTICS REPORT SUMMARY ==="])
        csv_data.append(["Report ID", report.report_id])
        csv_data.append(["Generated At", report.generated_at.isoformat()])
        csv_data.append(["Period Start", report.period_start.isoformat()])
        csv_data.append(["Period End", report.period_end.isoformat()])
        csv_data.append(["Period Duration (hours)", f"{report.period_duration_hours:.1f}"])
        csv_data.append([])

        # Cross-provider metrics
        csv_data.append(["=== CROSS-PROVIDER METRICS ==="])
        metrics = report.cross_provider_metrics
        csv_data.extend(
            [
                ["Total Requests", str(metrics.total_requests)],
                ["Total Tokens", str(metrics.total_tokens)],
                ["Total Cost", f"${metrics.total_cost:.4f}"],
                ["Total Errors", str(metrics.total_errors)],
                ["Success Rate", f"{metrics.overall_success_rate:.2f}%"],
                ["Average Cost per Request", f"${metrics.average_cost_per_request:.4f}"],
                ["Average Cost per Token", f"${metrics.average_cost_per_token:.6f}"],
                ["Unique Providers", str(metrics.unique_providers)],
                ["Unique Models", str(metrics.unique_models)],
                ["Cache Hit Rate", f"{metrics.cache_hit_rate:.2f}%"],
                ["Average Latency (ms)", f"{metrics.average_latency_ms:.2f}"],
                ["Most Used Provider", metrics.most_used_provider or "N/A"],
                ["Most Expensive Provider", metrics.most_expensive_provider or "N/A"],
                ["Fastest Provider", metrics.fastest_provider or "N/A"],
                ["Most Reliable Provider", metrics.most_reliable_provider or "N/A"],
            ]
        )
        csv_data.append([])

        # Provider breakdown
        csv_data.append(["=== PROVIDER BREAKDOWN ==="])
        csv_data.append(
            [
                "Provider",
                "Requests",
                "Tokens",
                "Cost",
                "Errors",
                "Success Rate",
                "Avg Latency (ms)",
                "Cost per Token",
                "Models Used",
            ]
        )

        for provider, stats in report.usage_breakdown.by_provider.items():
            csv_data.append(
                [
                    provider,
                    str(stats.total_requests),
                    str(stats.total_tokens),
                    f"${stats.total_cost:.4f}",
                    str(stats.error_count),
                    f"{stats.success_rate:.2f}%",
                    f"{stats.average_latency_ms:.2f}",
                    f"${stats.cost_per_token:.6f}",
                    ", ".join(stats.models_used),
                ]
            )
        csv_data.append([])

        # Model breakdown
        csv_data.append(["=== MODEL BREAKDOWN ==="])
        csv_data.append(
            [
                "Provider",
                "Model",
                "Requests",
                "Tokens",
                "Cost",
                "Errors",
                "Success Rate",
                "Avg Latency (ms)",
                "Cost per Token",
            ]
        )

        for _model_key, model_stats in report.usage_breakdown.by_model.items():
            csv_data.append(
                [
                    model_stats.provider,
                    model_stats.model,
                    str(model_stats.total_requests),
                    str(model_stats.total_tokens),
                    f"${model_stats.total_cost:.4f}",
                    str(model_stats.error_count),
                    f"{model_stats.success_rate:.2f}%",
                    f"{model_stats.average_latency_ms:.2f}",
                    f"${model_stats.cost_per_token:.6f}",
                ]
            )
        csv_data.append([])

        # Time series data
        if report.time_series:
            csv_data.append(["=== TIME SERIES DATA ==="])
            csv_data.append(
                ["Timestamp", "Requests", "Tokens", "Cost", "Errors", "Avg Latency (ms)"]
            )

            for point in report.time_series:
                csv_data.append(
                    [
                        point.timestamp.isoformat(),
                        str(point.requests),
                        str(point.tokens),
                        f"${point.cost:.4f}",
                        str(point.errors),
                        f"{point.latency_ms:.2f}",
                    ]
                )
            csv_data.append([])

        # Top models by usage
        if report.top_models_by_usage:
            csv_data.append(["=== TOP MODELS BY USAGE ==="])
            csv_data.append(["Provider", "Model", "Requests", "Tokens", "Cost", "Success Rate"])

            for model in report.top_models_by_usage:
                csv_data.append(
                    [
                        model.provider,
                        model.model,
                        str(model.total_requests),
                        str(model.total_tokens),
                        f"${model.total_cost:.4f}",
                        f"{model.success_rate:.2f}%",
                    ]
                )

        # Generate CSV string
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(csv_data)
        csv_content = output.getvalue()
        output.close()

        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                f.write(csv_content)

        return csv_content


class PDFExporter(ReportExporter):
    """Export analytics reports to PDF format."""

    def __init__(self, include_charts: bool = True):
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "ReportLab is required for PDF export. Install with: pip install reportlab"
            )

        self.include_charts = include_charts and MATPLOTLIB_AVAILABLE
        self.styles = getSampleStyleSheet()

        # Create custom styles
        self.styles.add(
            ParagraphStyle(
                name="CustomTitle",
                parent=self.styles["Title"],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.darkblue,
            )
        )

        self.styles.add(
            ParagraphStyle(
                name="SectionHeader",
                parent=self.styles["Heading2"],
                fontSize=16,
                spaceAfter=12,
                textColor=colors.darkblue,
                borderWidth=1,
                borderColor=colors.lightgrey,
                borderPadding=5,
            )
        )

    def export(
        self, report: AnalyticsReport, output_path: Optional[Union[str, Path]] = None
    ) -> bytes:
        """Export report to PDF format."""
        if output_path is None:
            output_path = f"analytics_report_{report.report_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create PDF document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )

        # Build PDF content
        story: List[Any] = []

        # Title page
        self._add_title_page(story, report)

        # Executive summary
        self._add_executive_summary(story, report)

        # Provider analysis
        self._add_provider_analysis(story, report)

        # Model analysis
        self._add_model_analysis(story, report)

        # Time series analysis
        if report.time_series:
            self._add_time_series_analysis(story, report)

        # Performance insights
        self._add_performance_insights(story, report)

        # Build PDF
        doc.build(story)

        # Read the file and return bytes
        with open(output_path, "rb") as f:
            pdf_content = f.read()

        return pdf_content

    def _add_title_page(self, story: List, report: AnalyticsReport) -> None:
        """Add title page to PDF."""
        story.append(Paragraph("JustLLMs Analytics Report", self.styles["CustomTitle"]))
        story.append(Spacer(1, 12))

        # Report metadata
        metadata = [
            ["Report ID:", report.report_id],
            ["Generated:", report.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC")],
            [
                "Period:",
                f"{report.period_start.strftime('%Y-%m-%d %H:%M')} to {report.period_end.strftime('%Y-%m-%d %H:%M')}",
            ],
            ["Duration:", f"{report.period_duration_hours:.1f} hours"],
        ]

        table = Table(metadata, colWidths=[2 * inch, 4 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ]
            )
        )

        story.append(table)
        story.append(PageBreak())

    def _add_executive_summary(self, story: List, report: AnalyticsReport) -> None:
        """Add executive summary section."""
        story.append(Paragraph("Executive Summary", self.styles["SectionHeader"]))
        story.append(Spacer(1, 12))

        metrics = report.cross_provider_metrics

        # Key metrics table
        key_metrics = [
            ["Metric", "Value"],
            ["Total Requests", f"{metrics.total_requests:,}"],
            ["Total Tokens", f"{metrics.total_tokens:,}"],
            ["Total Cost", f"${metrics.total_cost:.4f}"],
            ["Success Rate", f"{metrics.overall_success_rate:.2f}%"],
            ["Average Latency", f"{metrics.average_latency_ms:.2f} ms"],
            ["Cache Hit Rate", f"{metrics.cache_hit_rate:.2f}%"],
            ["Providers Used", str(metrics.unique_providers)],
            ["Models Used", str(metrics.unique_models)],
        ]

        table = Table(key_metrics, colWidths=[3 * inch, 2 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 11),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )

        story.append(table)
        story.append(Spacer(1, 20))

        # Key insights
        insights = []
        if metrics.most_used_provider:
            insights.append(f"• Most used provider: {metrics.most_used_provider}")
        if metrics.fastest_provider:
            insights.append(f"• Fastest provider: {metrics.fastest_provider}")
        if metrics.most_reliable_provider:
            insights.append(f"• Most reliable provider: {metrics.most_reliable_provider}")
        if metrics.cache_hit_rate > 0:
            insights.append(f"• Cache hit rate: {metrics.cache_hit_rate:.1f}%")

        if insights:
            story.append(Paragraph("Key Insights:", self.styles["Heading3"]))
            for insight in insights:
                story.append(Paragraph(insight, self.styles["Normal"]))

        story.append(Spacer(1, 20))

    def _add_provider_analysis(self, story: List, report: AnalyticsReport) -> None:
        """Add provider analysis section."""
        story.append(Paragraph("Provider Analysis", self.styles["SectionHeader"]))
        story.append(Spacer(1, 12))

        # Provider comparison table
        provider_data = [["Provider", "Requests", "Tokens", "Cost", "Success Rate", "Avg Latency"]]

        for provider, stats in report.usage_breakdown.by_provider.items():
            provider_data.append(
                [
                    provider,
                    f"{stats.total_requests:,}",
                    f"{stats.total_tokens:,}",
                    f"${stats.total_cost:.4f}",
                    f"{stats.success_rate:.1f}%",
                    f"{stats.average_latency_ms:.1f} ms",
                ]
            )

        if len(provider_data) > 1:  # Has data beyond header
            table = Table(
                provider_data,
                colWidths=[1.2 * inch, 1 * inch, 1 * inch, 1 * inch, 1 * inch, 1 * inch],
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )

            story.append(table)
            story.append(Spacer(1, 20))

    def _add_model_analysis(self, story: List, report: AnalyticsReport) -> None:
        """Add model analysis section."""
        story.append(Paragraph("Top Models by Usage", self.styles["SectionHeader"]))
        story.append(Spacer(1, 12))

        if report.top_models_by_usage:
            model_data = [["Provider", "Model", "Requests", "Cost", "Success Rate"]]

            for model in report.top_models_by_usage[:10]:  # Top 10
                model_data.append(
                    [
                        model.provider,
                        model.model,
                        f"{model.total_requests:,}",
                        f"${model.total_cost:.4f}",
                        f"{model.success_rate:.1f}%",
                    ]
                )

            table = Table(
                model_data, colWidths=[1.2 * inch, 2 * inch, 1 * inch, 1 * inch, 1 * inch]
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )

            story.append(table)
            story.append(Spacer(1, 20))

    def _add_time_series_analysis(self, story: List, report: AnalyticsReport) -> None:
        """Add time series analysis with charts if matplotlib available."""
        story.append(Paragraph("Usage Over Time", self.styles["SectionHeader"]))
        story.append(Spacer(1, 12))

        if self.include_charts and report.time_series:
            # Create matplotlib chart
            try:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

                timestamps = [point.timestamp for point in report.time_series]
                requests = [point.requests for point in report.time_series]
                costs = [point.cost for point in report.time_series]

                # Requests over time
                ax1.plot(timestamps, requests, marker="o", linewidth=2)
                ax1.set_title("Requests Over Time")
                ax1.set_ylabel("Requests")
                ax1.grid(True, alpha=0.3)

                # Cost over time
                ax2.plot(timestamps, costs, marker="o", linewidth=2, color="green")
                ax2.set_title("Cost Over Time")
                ax2.set_ylabel("Cost ($)")
                ax2.set_xlabel("Time")
                ax2.grid(True, alpha=0.3)

                # Format x-axis
                for ax in [ax1, ax2]:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
                    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

                plt.tight_layout()

                # Save chart to temporary file and add to PDF
                chart_path = f"/tmp/chart_{report.report_id}.png"
                plt.savefig(chart_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

                # Add image to story
                img = Image(chart_path, width=6 * inch, height=4 * inch)
                story.append(img)
                story.append(Spacer(1, 20))

            except Exception:
                # Fallback to table if chart creation fails
                pass

        # Always include tabular data as fallback
        if report.time_series:
            time_data = [["Time", "Requests", "Tokens", "Cost", "Errors"]]
            for point in report.time_series[-10:]:  # Last 10 points
                time_data.append(
                    [
                        point.timestamp.strftime("%m/%d %H:%M"),
                        str(point.requests),
                        f"{point.tokens:,}",
                        f"${point.cost:.4f}",
                        str(point.errors),
                    ]
                )

            table = Table(time_data, colWidths=[1.5 * inch, 1 * inch, 1 * inch, 1 * inch, 1 * inch])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )

            story.append(table)

    def _add_performance_insights(self, story: List, report: AnalyticsReport) -> None:
        """Add performance insights section."""
        story.append(Paragraph("Performance Insights", self.styles["SectionHeader"]))
        story.append(Spacer(1, 12))

        metrics = report.cross_provider_metrics

        insights = []

        # Cost efficiency insights
        if metrics.cost_efficiency_ranking:
            most_efficient = metrics.cost_efficiency_ranking[0]
            insights.append(f"Most cost-efficient provider: {most_efficient}")

        # Performance insights
        if metrics.performance_ranking:
            fastest = metrics.performance_ranking[0]
            insights.append(f"Fastest provider: {fastest}")

        # Reliability insights
        if metrics.reliability_ranking:
            most_reliable = metrics.reliability_ranking[0]
            insights.append(f"Most reliable provider: {most_reliable}")

        # Usage insights
        avg_cost_per_request = metrics.average_cost_per_request
        if avg_cost_per_request > 0:
            insights.append(f"Average cost per request: ${avg_cost_per_request:.4f}")

        # Cache insights
        if metrics.cache_hit_rate > 0:
            cache_savings = metrics.cache_hit_rate
            insights.append(f"Cache is saving {cache_savings:.1f}% of requests")

        for insight in insights:
            story.append(Paragraph(f"• {insight}", self.styles["Normal"]))
            story.append(Spacer(1, 6))

        story.append(Spacer(1, 20))
