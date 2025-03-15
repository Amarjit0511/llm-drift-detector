"""
Interactive dashboard for LLM drift visualization.

This module provides a Dash-based dashboard for visualizing and exploring
drift detection results, including trends, distributions, and alerts.
"""

import logging
import os
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import webbrowser
import time

logger = logging.getLogger(__name__)

class DashboardApp:
    """
    Interactive dashboard application for drift monitoring.
    
    This class wraps a Dash application for visualizing drift detection
    results with interactive plots and filters.
    """
    
    def __init__(
        self,
        title: str = "LLM Drift Monitor",
        host: str = "0.0.0.0",
        port: int = 8050,
        debug: bool = False,
        **kwargs
    ):
        """
        Initialize the dashboard application.
        
        Args:
            title: Dashboard title
            host: Host to serve dashboard on
            port: Port to serve dashboard on
            debug: Whether to run in debug mode
            **kwargs: Additional Dash app configuration
        """
        self.title = title
        self.host = host
        self.port = port
        self.debug = debug
        self.config = kwargs
        
        # Lazy load dash to avoid unnecessary dependencies
        self._app = None
        self._server = None
        self._is_running = False
        self._thread = None
        
        # Data sources
        self.metrics_df = pd.DataFrame()
        self.samples_df = pd.DataFrame()
        self.recent_alerts = []
        
        # Initialize the app
        self._init_app()
    
    def _init_app(self):
        """Initialize the Dash application."""
        try:
            # Import Dash
            import dash
            from dash import dcc, html, callback, Input, Output, State
            import plotly.express as px
            import plotly.graph_objects as go
            
            # Create Dash app
            self._app = dash.Dash(
                __name__,
                title=self.title,
                **self.config
            )
            
            # Get underlying Flask server
            self._server = self._app.server
            
            # Define layout
            self._app.layout = html.Div([
                # Header
                html.Div([
                    html.H1(self.title),
                    html.Div([
                        html.Span("Status: "),
                        html.Span(id="status-indicator", className="status-normal", children="Normal")
                    ], className="status-container"),
                    html.Div([
                        html.Button("Refresh", id="refresh-button", n_clicks=0),
                        dcc.Dropdown(
                            id="timeframe-dropdown",
                            options=[
                                {"label": "Last 24 hours", "value": "24h"},
                                {"label": "Last 7 days", "value": "7d"},
                                {"label": "Last 30 days", "value": "30d"}
                            ],
                            value="24h"
                        )
                    ], className="header-controls"),
                ], className="dashboard-header"),
                
                # Main content
                html.Div([
                    # Left panel with summary metrics
                    html.Div([
                        html.H2("Summary"),
                        html.Div(id="summary-metrics"),
                        html.Div([
                            html.H3("Recent Alerts"),
                            html.Div(id="alerts-container")
                        ]),
                        html.Div([
                            html.H3("Providers"),
                            dcc.Checklist(
                                id="provider-checklist",
                                options=[],
                                value=[]
                            )
                        ])
                    ], className="left-panel"),
                    
                    # Right panel with charts
                    html.Div([
                        html.Div([
                            html.H2("Drift Metrics Over Time"),
                            dcc.Graph(id="drift-time-graph")
                        ]),
                        html.Div([
                            html.H2("Drift by Detector Type"),
                            dcc.Graph(id="detector-comparison-graph")
                        ]),
                        html.Div([
                            html.H2("Distribution Changes"),
                            dcc.Dropdown(
                                id="distribution-metric-dropdown",
                                options=[
                                    {"label": "Response Length", "value": "response_length"},
                                    {"label": "Token Count", "value": "token_count"},
                                    {"label": "Response Time", "value": "response_time"}
                                ],
                                value="response_length"
                            ),
                            dcc.Graph(id="distribution-graph")
                        ])
                    ], className="right-panel")
                ], className="main-content"),
                
                # Footer
                html.Div([
                    html.P(f"LLM Drift Detector Dashboard - Last Updated: ", id="last-updated"),
                    dcc.Interval(
                        id="update-interval",
                        interval=60000,  # 1 minute in milliseconds
                        n_intervals=0
                    )
                ], className="dashboard-footer"),
                
                # CSS
                html.Style('''
                    .dashboard-header { display: flex; justify-content: space-between; align-items: center; padding: 1rem; background-color: #f8f9fa; }
                    .main-content { display: flex; }
                    .left-panel { width: 30%; padding: 1rem; }
                    .right-panel { width: 70%; padding: 1rem; }
                    .status-container { display: flex; align-items: center; }
                    .status-normal { color: green; font-weight: bold; }
                    .status-warning { color: orange; font-weight: bold; }
                    .status-critical { color: red; font-weight: bold; }
                    .alert-item { padding: 0.5rem; margin-bottom: 0.5rem; border-radius: 4px; }
                    .alert-warning { background-color: #fff3cd; }
                    .alert-critical { background-color: #f8d7da; }
                ''')
            ])
            
            # Define callbacks
            
            # Update data on refresh or interval
            @self._app.callback(
                [Output("last-updated", "children"),
                 Output("provider-checklist", "options"),
                 Output("provider-checklist", "value")],
                [Input("refresh-button", "n_clicks"),
                 Input("update-interval", "n_intervals")]
            )
            def update_data(n_clicks, n_intervals):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Get unique providers
                providers = []
                if not self.metrics_df.empty and "provider_name" in self.metrics_df.columns:
                    providers = self.metrics_df["provider_name"].unique().tolist()
                
                provider_options = [{"label": p, "value": p} for p in providers]
                provider_values = providers  # Select all by default
                
                return f"LLM Drift Detector Dashboard - Last Updated: {timestamp}", provider_options, provider_values
            
            # Update status indicator
            @self._app.callback(
                [Output("status-indicator", "children"),
                 Output("status-indicator", "className")],
                [Input("refresh-button", "n_clicks"),
                 Input("update-interval", "n_intervals")]
            )
            def update_status(n_clicks, n_intervals):
                # Check for drift alerts
                if not self.metrics_df.empty and "drift_detected" in self.metrics_df.columns:
                    recent_metrics = self._get_recent_metrics()
                    if recent_metrics.empty:
                        return "No Data", "status-normal"
                    
                    drift_detected = recent_metrics["drift_detected"].any()
                    
                    if drift_detected:
                        # Check severity
                        severe_drift = False
                        if "drift_score" in recent_metrics.columns and "threshold" in recent_metrics.columns:
                            severe_drift = (recent_metrics["drift_score"] > recent_metrics["threshold"] * 1.5).any()
                        
                        if severe_drift:
                            return "Critical", "status-critical"
                        else:
                            return "Warning", "status-warning"
                
                return "Normal", "status-normal"
            
            # Update summary metrics
            @self._app.callback(
                Output("summary-metrics", "children"),
                [Input("refresh-button", "n_clicks"),
                 Input("update-interval", "n_intervals"),
                 Input("provider-checklist", "value")]
            )
            def update_summary_metrics(n_clicks, n_intervals, selected_providers):
                if self.metrics_df.empty:
                    return html.P("No metrics data available")
                
                recent_metrics = self._get_recent_metrics(providers=selected_providers)
                
                if recent_metrics.empty:
                    return html.P("No recent metrics for selected providers")
                
                # Calculate summary metrics
                total_detections = len(recent_metrics)
                drift_count = recent_metrics["drift_detected"].sum() if "drift_detected" in recent_metrics.columns else 0
                drift_percentage = (drift_count / total_detections * 100) if total_detections > 0 else 0
                
                # Average drift score by detector
                detector_metrics = []
                if "detector_name" in recent_metrics.columns and "drift_score" in recent_metrics.columns:
                    for detector in recent_metrics["detector_name"].unique():
                        detector_data = recent_metrics[recent_metrics["detector_name"] == detector]
                        avg_score = detector_data["drift_score"].mean()
                        detector_metrics.append(html.Li(f"{detector}: {avg_score:.3f}"))
                
                return html.Div([
                    html.P(f"Total Detection Runs: {total_detections}"),
                    html.P(f"Drift Detected: {drift_count} ({drift_percentage:.1f}%)"),
                    html.H3("Average Drift Score by Detector"),
                    html.Ul(detector_metrics)
                ])
            
            # Update alerts container
            @self._app.callback(
                Output("alerts-container", "children"),
                [Input("refresh-button", "n_clicks"),
                 Input("update-interval", "n_intervals")]
            )
            def update_alerts(n_clicks, n_intervals):
                if not self.recent_alerts:
                    return html.P("No recent alerts")
                
                alert_items = []
                for alert in self.recent_alerts:
                    level = alert.get("level", "warning")
                    title = alert.get("title", "Unknown Alert")
                    message = alert.get("message", "")
                    timestamp = alert.get("timestamp", datetime.now())
                    
                    if isinstance(timestamp, datetime):
                        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        timestamp_str = str(timestamp)
                    
                    alert_items.append(html.Div([
                        html.H4(title),
                        html.P(message),
                        html.P(f"Time: {timestamp_str}", className="alert-time")
                    ], className=f"alert-item alert-{level}"))
                
                return alert_items
            
            # Update drift time graph
            @self._app.callback(
                Output("drift-time-graph", "figure"),
                [Input("refresh-button", "n_clicks"),
                 Input("update-interval", "n_intervals"),
                 Input("timeframe-dropdown", "value"),
                 Input("provider-checklist", "value")]
            )
            def update_drift_time_graph(n_clicks, n_intervals, timeframe, selected_providers):
                if self.metrics_df.empty:
                    return create_empty_figure("No metrics data available")
                
                # Filter by timeframe and providers
                filtered_df = self._filter_metrics(timeframe, selected_providers)
                
                if filtered_df.empty:
                    return create_empty_figure("No data for selected filters")
                
                # Create the figure
                return create_drift_plot(filtered_df)
            
            # Update detector comparison graph
            @self._app.callback(
                Output("detector-comparison-graph", "figure"),
                [Input("refresh-button", "n_clicks"),
                 Input("update-interval", "n_intervals"),
                 Input("timeframe-dropdown", "value"),
                 Input("provider-checklist", "value")]
            )
            def update_detector_comparison(n_clicks, n_intervals, timeframe, selected_providers):
                if self.metrics_df.empty:
                    return create_empty_figure("No metrics data available")
                
                # Filter by timeframe and providers
                filtered_df = self._filter_metrics(timeframe, selected_providers)
                
                if filtered_df.empty:
                    return create_empty_figure("No data for selected filters")
                
                # Check if we have necessary columns
                if "detector_name" not in filtered_df.columns or "drift_score" not in filtered_df.columns:
                    return create_empty_figure("Missing required columns for detector comparison")
                
                # Create the figure
                return create_heatmap_plot(filtered_df)
            
            # Update distribution graph
            @self._app.callback(
                Output("distribution-graph", "figure"),
                [Input("refresh-button", "n_clicks"),
                 Input("update-interval", "n_intervals"),
                 Input("distribution-metric-dropdown", "value"),
                 Input("provider-checklist", "value")]
            )
            def update_distribution_graph(n_clicks, n_intervals, metric, selected_providers):
                if self.samples_df.empty:
                    return create_empty_figure("No samples data available")
                
                # Check if metric column exists
                if metric not in self.samples_df.columns:
                    return create_empty_figure(f"Metric '{metric}' not found in samples data")
                
                # Filter by providers
                filtered_df = self.samples_df
                if selected_providers and "provider_name" in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df["provider_name"].isin(selected_providers)]
                
                if filtered_df.empty:
                    return create_empty_figure("No data for selected filters")
                
                # Create the figure
                return create_histogram_plot(filtered_df, metric)
            
            logger.info("Dashboard application initialized")
            
        except ImportError as e:
            self._app = None
            logger.error(f"Failed to initialize dashboard due to missing dependencies: {str(e)}")
            logger.error("To use the dashboard, install with: pip install llm-drift-detector[visualization]")
        except Exception as e:
            self._app = None
            logger.error(f"Failed to initialize dashboard: {str(e)}")
    
    def _get_recent_metrics(self, days: int = 1, providers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get recent metrics from the metrics DataFrame.
        
        Args:
            days: Number of days to consider as recent
            providers: Optional list of providers to filter by
            
        Returns:
            pd.DataFrame: Filtered metrics DataFrame
        """
        if self.metrics_df.empty:
            return pd.DataFrame()
        
        # Ensure timestamp column exists and is datetime
        if "timestamp" not in self.metrics_df.columns:
            return pd.DataFrame()
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(self.metrics_df["timestamp"]):
            try:
                self.metrics_df["timestamp"] = pd.to_datetime(self.metrics_df["timestamp"])
            except:
                return pd.DataFrame()
        
        # Filter by date
        cutoff = datetime.now() - timedelta(days=days)
        recent_df = self.metrics_df[self.metrics_df["timestamp"] >= cutoff]
        
        # Filter by providers if specified
        if providers and "provider_name" in recent_df.columns:
            recent_df = recent_df[recent_df["provider_name"].isin(providers)]
        
        return recent_df
    
    def _filter_metrics(self, timeframe: str, providers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Filter metrics DataFrame by timeframe and providers.
        
        Args:
            timeframe: Timeframe string ('24h', '7d', '30d')
            providers: Optional list of providers to filter by
            
        Returns:
            pd.DataFrame: Filtered metrics DataFrame
        """
        if self.metrics_df.empty:
            return pd.DataFrame()
        
        # Convert timeframe to days
        days = 1  # default
        if timeframe == "7d":
            days = 7
        elif timeframe == "30d":
            days = 30
        
        return self._get_recent_metrics(days=days, providers=providers)
    
    def update_data(
        self,
        metrics: Optional[pd.DataFrame] = None,
        samples: Optional[pd.DataFrame] = None,
        alerts: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Update the dashboard data.
        
        Args:
            metrics: New metrics DataFrame
            samples: New samples DataFrame
            alerts: New list of alerts
        """
        if metrics is not None:
            self.metrics_df = metrics
        
        if samples is not None:
            self.samples_df = samples
        
        if alerts is not None:
            self.recent_alerts = alerts[:10]  # Keep only the 10 most recent alerts
    
    def run(self, open_browser: bool = True):
        """
        Run the dashboard server.
        
        Args:
            open_browser: Whether to open a browser automatically
        """
        if self._app is None:
            logger.error("Dashboard not initialized, cannot run")
            return
        
        if self._is_running:
            logger.warning("Dashboard is already running")
            return
        
        # Open browser if requested
        if open_browser:
            def open_browser_tab():
                time.sleep(1)  # Give the server a second to start
                url = f"http://localhost:{self.port}"
                webbrowser.open(url)
            
            threading.Thread(target=open_browser_tab).start()
        
        # Run the server
        self._is_running = True
        self._app.run_server(
            host=self.host,
            port=self.port,
            debug=self.debug
        )
        self._is_running = False
    
    def start_background(self):
        """Start the dashboard server in a background thread."""
        if self._app is None:
            logger.error("Dashboard not initialized, cannot start")
            return
        
        if self._is_running:
            logger.warning("Dashboard is already running")
            return
        
        def run_server():
            self._app.run_server(
                host=self.host,
                port=self.port,
                debug=False
            )
        
        self._thread = threading.Thread(target=run_server)
        self._thread.daemon = True
        self._thread.start()
        self._is_running = True
        
        logger.info(f"Dashboard started in background at http://{self.host}:{self.port}")
    
    def stop(self):
        """Stop the dashboard server if running in the background."""
        if not self._is_running:
            return
        
        # There's no clean way to stop a Dash server from code
        # This is a somewhat hacky solution, but it works
        import requests
        try:
            requests.get(f"http://{self.host}:{self.port}/_shutdown")
        except:
            pass
        
        self._is_running = False
        logger.info("Dashboard stopped")


def create_empty_figure(message: str = "No data available") -> Dict[str, Any]:
    """
    Create an empty figure with a message.
    
    Args:
        message: Message to display
        
    Returns:
        Dict[str, Any]: Empty figure dict
    """
    try:
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig
    except ImportError:
        return {}


def create_drift_plot(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a line plot of drift scores over time.
    
    Args:
        df: DataFrame with drift metrics
        
    Returns:
        Dict[str, Any]: Plotly figure
    """
    try:
        import plotly.express as px
        
        # Check required columns
        if "timestamp" not in df.columns or "drift_score" not in df.columns:
            return create_empty_figure("Missing required columns for drift plot")
        
        # Create color column based on drift_detected if available
        if "drift_detected" in df.columns:
            df = df.copy()
            df["status"] = df["drift_detected"].apply(lambda x: "Drift Detected" if x else "Normal")
            color_column = "status"
        else:
            color_column = None
        
        # Create the figure
        fig = px.line(
            df,
            x="timestamp",
            y="drift_score",
            color=color_column if color_column else None,
            line_group="detector_name" if "detector_name" in df.columns else None,
            facet_row="detector_name" if "detector_name" in df.columns else None,
            hover_data=["provider_name", "model_name"] if "provider_name" in df.columns and "model_name" in df.columns else None,
            title="Drift Score Over Time",
            labels={"drift_score": "Drift Score", "timestamp": "Time"}
        )
        
        # Add threshold line if available
        if "threshold" in df.columns:
            # Add a horizontal line at the average threshold value
            avg_threshold = df["threshold"].mean()
            fig.add_hline(
                y=avg_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold ({avg_threshold:.2f})",
                annotation_position="bottom right"
            )
        
        # Update layout
        fig.update_layout(
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    except ImportError:
        return create_empty_figure("Plotly not installed")
    except Exception as e:
        logger.error(f"Error creating drift plot: {str(e)}")
        return create_empty_figure(f"Error: {str(e)}")


def create_histogram_plot(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Create a histogram of a numeric column.
    
    Args:
        df: DataFrame with samples
        column: Column to plot
        
    Returns:
        Dict[str, Any]: Plotly figure
    """
    try:
        import plotly.express as px
        
        # Check if column exists and has numeric data
        if column not in df.columns:
            return create_empty_figure(f"Column '{column}' not found")
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            return create_empty_figure(f"Column '{column}' is not numeric")
        
        # Create the figure
        color_column = "provider_name" if "provider_name" in df.columns else None
        
        fig = px.histogram(
            df,
            x=column,
            color=color_column,
            marginal="box",
            title=f"Distribution of {column.replace('_', ' ').title()}",
            labels={column: column.replace('_', ' ').title()}
        )
        
        # Update layout
        fig.update_layout(
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    except ImportError:
        return create_empty_figure("Plotly not installed")
    except Exception as e:
        logger.error(f"Error creating histogram plot: {str(e)}")
        return create_empty_figure(f"Error: {str(e)}")


def create_heatmap_plot(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a heatmap of drift scores by detector and provider.
    
    Args:
        df: DataFrame with drift metrics
        
    Returns:
        Dict[str, Any]: Plotly figure
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        
        # Check required columns
        required_columns = ["detector_name", "drift_score"]
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            return create_empty_figure(f"Missing columns: {', '.join(missing)}")
        
        # Create pivot table
        pivot_columns = ["detector_name"]
        if "provider_name" in df.columns:
            pivot_columns.append("provider_name")
        
        try:
            pivot_df = df.pivot_table(
                values="drift_score",
                index=pivot_columns[0],
                columns=pivot_columns[1] if len(pivot_columns) > 1 else None,
                aggfunc="mean"
            )
        except:
            # Fall back to groupby if pivot fails
            if len(pivot_columns) > 1:
                pivot_df = df.groupby(pivot_columns).mean(numeric_only=True)["drift_score"].unstack()
            else:
                pivot_df = df.groupby(pivot_columns[0]).mean(numeric_only=True)[["drift_score"]]
        
        # Create heatmap
        if len(pivot_columns) > 1:
            fig = px.imshow(
                pivot_df,
                title="Average Drift Score by Detector and Provider",
                labels=dict(x="Provider", y="Detector", color="Drift Score"),
                color_continuous_scale="Viridis"
            )
        else:
            # Create bar chart if only one dimension
            detector_name = pivot_columns[0]
            fig = px.bar(
                pivot_df.reset_index(),
                x=detector_name,
                y="drift_score",
                title="Average Drift Score by Detector",
                labels={detector_name: "Detector", "drift_score": "Drift Score"}
            )
        
        # Update layout
        fig.update_layout(height=400)
        
        return fig
    except ImportError:
        return create_empty_figure("Plotly not installed")
    except Exception as e:
        logger.error(f"Error creating heatmap plot: {str(e)}")
        return create_empty_figure(f"Error: {str(e)}")


def create_line_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    color_column: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a line plot.
    
    Args:
        df: DataFrame with data
        x_column: Column for x-axis
        y_column: Column for y-axis
        title: Plot title
        color_column: Optional column for line color
        
    Returns:
        Dict[str, Any]: Plotly figure
    """
    try:
        import plotly.express as px
        
        # Check required columns
        if x_column not in df.columns or y_column not in df.columns:
            missing = []
            if x_column not in df.columns:
                missing.append(x_column)
            if y_column not in df.columns:
                missing.append(y_column)
            return create_empty_figure(f"Missing columns: {', '.join(missing)}")
        
        # Create the figure
        fig = px.line(
            df,
            x=x_column,
            y=y_column,
            color=color_column if color_column and color_column in df.columns else None,
            title=title,
            labels={
                x_column: x_column.replace('_', ' ').title(),
                y_column: y_column.replace('_', ' ').title()
            }
        )
        
        # Update layout
        fig.update_layout(
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    except ImportError:
        return create_empty_figure("Plotly not installed")
    except Exception as e:
        logger.error(f"Error creating line plot: {str(e)}")
        return create_empty_figure(f"Error: {str(e)}")


def create_scatter_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    color_column: Optional[str] = None,
    size_column: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a scatter plot.
    
    Args:
        df: DataFrame with data
        x_column: Column for x-axis
        y_column: Column for y-axis
        title: Plot title
        color_column: Optional column for point color
        size_column: Optional column for point size
        
    Returns:
        Dict[str, Any]: Plotly figure
    """
    try:
        import plotly.express as px
        
        # Check required columns
        if x_column not in df.columns or y_column not in df.columns:
            missing = []
            if x_column not in df.columns:
                missing.append(x_column)
            if y_column not in df.columns:
                missing.append(y_column)
            return create_empty_figure(f"Missing columns: {', '.join(missing)}")
        
        # Create the figure
        fig = px.scatter(
            df,
            x=x_column,
            y=y_column,
            color=color_column if color_column and color_column in df.columns else None,
            size=size_column if size_column and size_column in df.columns else None,
            hover_name="provider_name" if "provider_name" in df.columns else None,
            title=title,
            labels={
                x_column: x_column.replace('_', ' ').title(),
                y_column: y_column.replace('_', ' ').title()
            }
        )
        
        # Update layout
        fig.update_layout(
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    except ImportError:
        return create_empty_figure("Plotly not installed")
    except Exception as e:
        logger.error(f"Error creating scatter plot: {str(e)}")
        return create_empty_figure(f"Error: {str(e)}")


def create_embedding_plot(
    embeddings: Union[np.ndarray, Dict[str, np.ndarray]],
    labels: Optional[List[str]] = None,
    method: str = "pca",
    title: str = "Embedding Space Visualization",
    point_size: int = 5
) -> Dict[str, Any]:
    """
    Create a visualization of embeddings after dimensionality reduction.
    
    Args:
        embeddings: Array of embeddings or dictionary mapping IDs to embeddings
        labels: Optional labels for each embedding
        method: Dimensionality reduction method ('pca', 'tsne', or 'umap')
        title: Plot title
        point_size: Size of points in the visualization
        
    Returns:
        Dict[str, Any]: Plotly figure
    """
    try:
        import plotly.express as px
        import numpy as np
        
        # Convert dictionary to array if needed
        if isinstance(embeddings, dict):
            ids = list(embeddings.keys())
            embedding_array = np.array(list(embeddings.values()))
            
            # Generate default labels from keys if not provided
            if labels is None:
                labels = ids
        else:
            embedding_array = embeddings
        
        # Check if we have embeddings
        if embedding_array.size == 0:
            return create_empty_figure("No embeddings to visualize")
        
        # Apply dimensionality reduction
        reduced_embeddings = np.empty((len(embedding_array), 2))
        
        if method == "pca":
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                reduced_embeddings = pca.fit_transform(embedding_array)
                method_name = "PCA"
            except ImportError:
                # Fall back to simple PCA
                cov = np.cov(embedding_array.T)
                evals, evecs = np.linalg.eigh(cov)
                # Sort eigenvectors by eigenvalues in descending order
                idx = np.argsort(evals)[::-1]
                evecs = evecs[:, idx]
                # Project data onto first two principal components
                reduced_embeddings = np.dot(embedding_array, evecs[:, :2])
                method_name = "Simple PCA"
        
        elif method == "tsne":
            try:
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embedding_array)-1))
                reduced_embeddings = tsne.fit_transform(embedding_array)
                method_name = "t-SNE"
            except ImportError:
                logger.warning("sklearn not installed. Falling back to PCA for embedding visualization.")
                # Fall back to PCA
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                reduced_embeddings = pca.fit_transform(embedding_array)
                method_name = "PCA (fallback)"
        
        elif method == "umap":
            try:
                import umap
                reducer = umap.UMAP(random_state=42)
                reduced_embeddings = reducer.fit_transform(embedding_array)
                method_name = "UMAP"
            except ImportError:
                logger.warning("umap-learn not installed. Falling back to PCA for embedding visualization.")
                # Fall back to PCA
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                reduced_embeddings = pca.fit_transform(embedding_array)
                method_name = "PCA (fallback)"
        
        else:
            # Default to PCA for unknown method
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embedding_array)
            method_name = "PCA"
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1]
        })
        
        # Add labels if provided
        if labels is not None:
            if len(labels) == len(df):
                df['label'] = labels
            else:
                logger.warning(f"Number of labels ({len(labels)}) doesn't match number of embeddings ({len(df)})")
                df['label'] = [f"Point {i}" for i in range(len(df))]
        else:
            df['label'] = [f"Point {i}" for i in range(len(df))]
        
        # Create the scatter plot
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='label' if 'label' in df.columns else None,
            hover_name='label' if 'label' in df.columns else None,
            title=f"{title} ({method_name})",
            labels={'x': f"{method_name} Dimension 1", 'y': f"{method_name} Dimension 2"}
        )
        
        # Update marker size
        fig.update_traces(marker=dict(size=point_size))
        
        # Update layout
        fig.update_layout(
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ) if 'label' in df.columns else None
        )
        
        return fig
    except ImportError as e:
        logger.error(f"Error importing required packages for embedding plot: {str(e)}")
        return create_empty_figure("Required packages not installed")
    except Exception as e:
        logger.error(f"Error creating embedding plot: {str(e)}")
        return create_empty_figure(f"Error: {str(e)}")


def create_dashboard(config: Optional[Dict[str, Any]] = None) -> DashboardApp:
    """
    Create a dashboard application with default settings.
    
    Args:
        config: Optional dashboard configuration
        
    Returns:
        DashboardApp: Dashboard application
    """
    from ..config import get_config
    
    # Get dashboard settings from config
    llm_config = get_config()
    dashboard_config = llm_config.get("monitoring.dashboard", {})
    
    # Override with provided config
    if config:
        dashboard_config.update(config)
    
    # Create dashboard
    dashboard = DashboardApp(
        title=dashboard_config.get("title", "LLM Drift Monitor"),
        host=dashboard_config.get("host", "0.0.0.0"),
        port=dashboard_config.get("port", 8050),
        debug=dashboard_config.get("debug", False)
    )
    
    return dashboard


def launch_dashboard(
    metrics: Optional[pd.DataFrame] = None,
    samples: Optional[pd.DataFrame] = None,
    alerts: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
    background: bool = False,
    open_browser: bool = True
) -> DashboardApp:
    """
    Launch a dashboard with the provided data.
    
    Args:
        metrics: Metrics DataFrame to display
        samples: Samples DataFrame to display
        alerts: List of alerts to display
        config: Optional dashboard configuration
        background: Whether to run the dashboard in the background
        open_browser: Whether to open a browser automatically
        
    Returns:
        DashboardApp: Dashboard application
    """
    dashboard = create_dashboard(config)
    
    # Update data
    dashboard.update_data(metrics, samples, alerts)
    
    # Run dashboard
    if background:
        dashboard.start_background()
        if open_browser:
            import webbrowser
            import time
            time.sleep(1)  # Give the server a moment to start
            webbrowser.open(f"http://localhost:{dashboard.port}")
    else:
        dashboard.run(open_browser=open_browser)
    
    return dashboard