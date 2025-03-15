"""
Alerting system for LLM drift detection.

This module provides functionality for sending alerts when
significant drift is detected in LLM outputs.
"""

import logging
import time
import threading
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import os
import requests
from dataclasses import dataclass, field, asdict

from ..config import Config, get_config

logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """
    Alert notification for detected drift.
    """
    # Basic information
    timestamp: datetime = field(default_factory=datetime.now)
    level: str = "warning"  # warning, critical
    title: str = ""
    message: str = ""
    
    # Source information
    provider_name: str = ""
    model_name: str = ""
    detector_name: Optional[str] = None
    metric_name: Optional[str] = None
    
    # Drift details
    drift_score: float = 0.0
    threshold: float = 0.0
    
    # Additional data
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the alert
        """
        result = asdict(self)
        result["timestamp"] = result["timestamp"].isoformat()
        return result


class AlertManager:
    """
    Manages and sends alerts for detected drift.
    
    This class handles alert throttling, delivery via different
    channels, and alert history management.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the alert manager.
        
        Args:
            config: Configuration object, uses global config if None
        """
        self.config = config or get_config()
        
        # Check if alerting is enabled
        self.enabled = self.config.get("monitoring.alerts.enabled", True)
        
        if not self.enabled:
            logger.info("Alert manager is disabled in configuration")
            return
        
        # Alert throttling settings
        self.cooldown_minutes = self.config.get("monitoring.alerts.cooldown_minutes", 60)
        
        # Alert thresholds
        self.thresholds = self.config.get("monitoring.alerts.thresholds", {
            "warning": 0.7,  # 70% of detector threshold
            "critical": 1.0   # 100% of detector threshold
        })
        
        # Configure alert channels
        self._configure_email()
        self._configure_slack()
        self._configure_http()
        
        # Alert history
        self.alert_history = []
        self.max_history_size = 100
        
        # Last alert timestamp by key (for throttling)
        self.last_alert_times = {}
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def _configure_email(self):
        """Configure email alerting."""
        email_config = self.config.get("monitoring.alerts.channels.email", {})
        self.email_enabled = email_config.get("enabled", False)
        
        if self.email_enabled:
            self.smtp_server = email_config.get("smtp_server", "")
            self.smtp_port = email_config.get("smtp_port", 587)
            self.smtp_username = email_config.get("username", "")
            
            # Get password from environment variable
            password_env_var = email_config.get("password_env_var", "")
            self.smtp_password = os.environ.get(password_env_var, "")
            
            self.email_from = email_config.get("from_address", "")
            self.email_to = email_config.get("to_addresses", [])
            
            if not self.smtp_server or not self.smtp_username or not self.smtp_password or not self.email_from or not self.email_to:
                logger.warning("Email alerting enabled but configuration is incomplete")
                self.email_enabled = False
    
    def _configure_slack(self):
        """Configure Slack alerting."""
        slack_config = self.config.get("monitoring.alerts.channels.slack", {})
        self.slack_enabled = slack_config.get("enabled", False)
        
        if self.slack_enabled:
            # Get webhook URL from environment variable
            webhook_env_var = slack_config.get("webhook_url_env_var", "")
            self.slack_webhook = os.environ.get(webhook_env_var, "")
            
            self.slack_channel = slack_config.get("channel", "#llm-drift-alerts")
            self.slack_username = slack_config.get("username", "LLM Drift Monitor")
            
            if not self.slack_webhook:
                logger.warning("Slack alerting enabled but webhook URL is missing")
                self.slack_enabled = False
    
    def _configure_http(self):
        """Configure HTTP webhook alerting."""
        http_config = self.config.get("monitoring.alerts.channels.http", {})
        self.http_enabled = http_config.get("enabled", False)
        
        if self.http_enabled:
            self.http_endpoint = http_config.get("endpoint", "")
            self.http_method = http_config.get("method", "POST")
            self.http_headers = http_config.get("headers", {})
            
            if not self.http_endpoint:
                logger.warning("HTTP alerting enabled but endpoint is missing")
                self.http_enabled = False
    
    def send_alert(self, alert: Alert) -> bool:
        """
        Send an alert through configured channels.
        
        Args:
            alert: Alert to send
            
        Returns:
            bool: True if alert was sent, False otherwise
        """
        if not self.enabled:
            logger.debug("Alert manager is disabled, not sending alert")
            return False
        
        with self._lock:
            # Check throttling
            if not self._should_send_alert(alert):
                logger.debug(f"Alert throttled: {alert.title}")
                return False
            
            # Add to history
            self.alert_history.append(alert)
            if len(self.alert_history) > self.max_history_size:
                self.alert_history = self.alert_history[-self.max_history_size:]
            
            # Update last alert time
            alert_key = self._get_alert_key(alert)
            self.last_alert_times[alert_key] = time.time()
        
        # Send through each channel
        success = False
        
        if self.email_enabled:
            email_success = self._send_email_alert(alert)
            success = success or email_success
        
        if self.slack_enabled:
            slack_success = self._send_slack_alert(alert)
            success = success or slack_success
        
        if self.http_enabled:
            http_success = self._send_http_alert(alert)
            success = success or http_success
        
        if success:
            logger.info(f"Alert sent: {alert.title}")
        else:
            logger.warning(f"Failed to send alert: {alert.title}")
        
        return success
    
    def _should_send_alert(self, alert: Alert) -> bool:
        """
        Check if an alert should be sent based on throttling rules.
        
        Args:
            alert: Alert to check
            
        Returns:
            bool: True if alert should be sent, False if it should be throttled
        """
        # Get alert key for throttling
        alert_key = self._get_alert_key(alert)
        
        # Check if we've sent an alert recently
        if alert_key in self.last_alert_times:
            last_time = self.last_alert_times[alert_key]
            elapsed_seconds = time.time() - last_time
            cooldown_seconds = self.cooldown_minutes * 60
            
            # Allow critical alerts to bypass throttling for warning alerts
            if alert.level == "critical" and elapsed_seconds > cooldown_seconds / 2:
                return True
            
            if elapsed_seconds < cooldown_seconds:
                return False
        
        return True
    
    def _get_alert_key(self, alert: Alert) -> str:
        """
        Get a unique key for an alert for throttling purposes.
        
        Args:
            alert: Alert to get key for
            
        Returns:
            str: Alert key
        """
        return f"{alert.provider_name}:{alert.model_name}:{alert.detector_name or ''}:{alert.level}"
    
    def _send_email_alert(self, alert: Alert) -> bool:
        """
        Send an alert via email.
        
        Args:
            alert: Alert to send
            
        Returns:
            bool: True if alert was sent successfully, False otherwise
        """
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_from
            msg['To'] = ', '.join(self.email_to)
            
            # Set subject based on alert level
            if alert.level == "critical":
                msg['Subject'] = f"[CRITICAL] {alert.title}"
            else:
                msg['Subject'] = f"[WARNING] {alert.title}"
            
            # Build email body
            body = f"{alert.message}\n\n"
            body += f"Provider: {alert.provider_name}\n"
            body += f"Model: {alert.model_name}\n"
            body += f"Detector: {alert.detector_name or 'N/A'}\n"
            body += f"Metric: {alert.metric_name or 'N/A'}\n"
            body += f"Drift Score: {alert.drift_score:.2f} (Threshold: {alert.threshold:.2f})\n"
            body += f"Timestamp: {alert.timestamp.isoformat()}\n\n"
            
            if alert.details:
                body += "Details:\n"
                body += json.dumps(alert.details, indent=2)
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending email alert: {str(e)}")
            return False
    
    def _send_slack_alert(self, alert: Alert) -> bool:
        """
        Send an alert via Slack webhook.
        
        Args:
            alert: Alert to send
            
        Returns:
            bool: True if alert was sent successfully, False otherwise
        """
        try:
            # Set color based on alert level
            color = "#ff0000" if alert.level == "critical" else "#ffcc00"
            
            # Build Slack message
            message = {
                "username": self.slack_username,
                "channel": self.slack_channel,
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": color,
                        "title": alert.title,
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Provider",
                                "value": alert.provider_name,
                                "short": True
                            },
                            {
                                "title": "Model",
                                "value": alert.model_name,
                                "short": True
                            },
                            {
                                "title": "Detector",
                                "value": alert.detector_name or "N/A",
                                "short": True
                            },
                            {
                                "title": "Metric",
                                "value": alert.metric_name or "N/A",
                                "short": True
                            },
                            {
                                "title": "Drift Score",
                                "value": f"{alert.drift_score:.2f} (Threshold: {alert.threshold:.2f})",
                                "short": False
                            }
                        ],
                        "footer": f"LLM Drift Detector • {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                    }
                ]
            }
            
            # Add details if available
            if alert.details:
                message["attachments"][0]["fields"].append({
                    "title": "Details",
                    "value": f"```{json.dumps(alert.details, indent=2)}```",
                    "short": False
                })
            
            # Send to Slack
            response = requests.post(
                self.slack_webhook,
                json=message,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                logger.warning(f"Slack API error: {response.status_code} - {response.text}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending Slack alert: {str(e)}")
            return False
    
    def _send_http_alert(self, alert: Alert) -> bool:
        """
        Send an alert via HTTP webhook.
        
        Args:
            alert: Alert to send
            
        Returns:
            bool: True if alert was sent successfully, False otherwise
        """
        try:
            # Convert alert to JSON
            payload = alert.to_dict()
            
            # Send HTTP request
            if self.http_method.upper() == "POST":
                response = requests.post(
                    self.http_endpoint,
                    json=payload,
                    headers=self.http_headers,
                    timeout=10
                )
            elif self.http_method.upper() == "PUT":
                response = requests.put(
                    self.http_endpoint,
                    json=payload,
                    headers=self.http_headers,
                    timeout=10
                )
            else:
                logger.warning(f"Unsupported HTTP method: {self.http_method}")
                return False
            
            # Check response
            if response.status_code < 200 or response.status_code >= 300:
                logger.warning(f"HTTP webhook error: {response.status_code} - {response.text}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending HTTP alert: {str(e)}")
            return False
    
    def create_alert_from_drift_result(self,
                                     provider_name: str,
                                     model_name: str,
                                     result: Any,  # DriftResult
                                     custom_message: Optional[str] = None) -> Optional[Alert]:
        """
        Create an alert from a drift detection result.
        
        Args:
            provider_name: Name of the LLM provider
            model_name: Name of the model
            result: Drift detection result
            custom_message: Optional custom message
            
        Returns:
            Optional[Alert]: Created alert, or None if no alert should be sent
        """
        # Check if result indicates drift
        if not result.drift_detected:
            return None
        
        # Determine alert level based on drift score
        level = "warning"
        if result.drift_score >= self.thresholds.get("critical", 1.0) * result.threshold:
            level = "critical"
        elif result.drift_score < self.thresholds.get("warning", 0.7) * result.threshold:
            return None  # Below warning threshold
        
        # Create title and message
        detector_name = result.detector_name
        metric_name = result.metric_name
        
        title = f"LLM Drift Detected: {provider_name}/{model_name}"
        
        if custom_message:
            message = custom_message
        else:
            message = f"Drift detected in {detector_name} detector"
            if metric_name:
                message += f" for metric '{metric_name}'"
            message += f" with score {result.drift_score:.2f} (threshold: {result.threshold:.2f})"
        
        # Create alert
        alert = Alert(
            level=level,
            title=title,
            message=message,
            provider_name=provider_name,
            model_name=model_name,
            detector_name=detector_name,
            metric_name=metric_name,
            drift_score=result.drift_score,
            threshold=result.threshold,
            details=result.analysis
        )
        
        return alert
    
    def get_alert_history(self,
                         hours: int = 24,
                         level: Optional[str] = None,
                         provider_name: Optional[str] = None,
                         model_name: Optional[str] = None,
                         detector_name: Optional[str] = None) -> List[Alert]:
        """
        Get alert history with optional filtering.
        
        Args:
            hours: Number of hours to look back
            level: Optional alert level filter
            provider_name: Optional provider name filter
            model_name: Optional model name filter
            detector_name: Optional detector name filter
            
        Returns:
            List[Alert]: Filtered list of alerts
        """
        with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(hours=hours)
            
            filtered_alerts = []
            
            for alert in self.alert_history:
                # Apply time filter
                if alert.timestamp < cutoff:
                    continue
                
                # Apply level filter
                if level and alert.level != level:
                    continue
                
                # Apply provider filter
                if provider_name and alert.provider_name != provider_name:
                    continue
                
                # Apply model filter
                if model_name and alert.model_name != model_name:
                    continue
                
                # Apply detector filter
                if detector_name and alert.detector_name != detector_name:
                    continue
                
                filtered_alerts.append(alert)
            
            return filtered_alerts