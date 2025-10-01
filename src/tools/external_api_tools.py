# External API Tools for CrewAI Agents
import asyncio
import logging
import json
import aiohttp
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from src.core.config import get_settings
from src.core.utils import sanitize_input, measure_execution_time, format_response


class OpenAIInput(BaseModel):
    """Input schema for OpenAI tool"""

    prompt: str = Field(..., description="Prompt to send to OpenAI")
    model: str = Field("gpt-4", description="OpenAI model to use")
    temperature: float = Field(0.3, description="Temperature for response generation")
    max_tokens: int = Field(1000, description="Maximum tokens in response")


class EmailNotificationInput(BaseModel):
    """Input schema for email notification tool"""

    to_email: str = Field(..., description="Recipient email address")
    subject: str = Field(..., description="Email subject")
    message: str = Field(..., description="Email message content")
    priority: str = Field("normal", description="Email priority: low, normal, high")
    template_name: Optional[str] = Field(None, description="Email template to use")


class SlackNotificationInput(BaseModel):
    """Input schema for Slack notification tool"""

    channel: str = Field(..., description="Slack channel or user to notify")
    message: str = Field(..., description="Message to send")
    priority: str = Field(
        "normal", description="Message priority: low, normal, high, urgent"
    )
    thread_ts: Optional[str] = Field(None, description="Thread timestamp for replies")


class WebhookInput(BaseModel):
    """Input schema for webhook tool"""

    url: str = Field(..., description="Webhook URL to send data to")
    payload: Dict[str, Any] = Field(..., description="Data payload to send")
    method: str = Field("POST", description="HTTP method: GET, POST, PUT, DELETE")
    headers: Optional[Dict[str, str]] = Field(None, description="Custom headers")
    timeout: int = Field(30, description="Request timeout in seconds")


class OpenAITool(BaseTool):
    """
    Tool for making direct OpenAI API calls when agents need additional LLM processing.
    Useful for specialized prompts, different models, or specific configurations.
    """

    name: str = "openai_tool"
    description: str = """Make direct calls to OpenAI API with custom prompts and parameters.
    Useful for specialized processing, different models, or specific configurations."""
    args_schema: type[BaseModel] = OpenAIInput

    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)

        # Initialize OpenAI client
        self.client = ChatOpenAI(
            model=self.settings.ai_models.openai_model,
            openai_api_key=self.settings.ai_models.openai_api_key,
            temperature=self.settings.ai_models.temperature,
        )

    @measure_execution_time
    async def _arun(
        self,
        prompt: str,
        model: str = "gpt-4",
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> str:
        """Make OpenAI API call asynchronously"""
        try:
            # Sanitize prompt
            clean_prompt = sanitize_input(prompt, max_length=50000)
            if not clean_prompt:
                return "Error: Empty or invalid prompt provided"

            # Configure client for this specific request
            client = ChatOpenAI(
                model=model,
                openai_api_key=self.settings.ai_models.openai_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.settings.ai_models.timeout,
            )

            # Make the API call
            response = await client.ainvoke(clean_prompt)

            if not response or not response.content:
                return "Error: Empty response from OpenAI API"

            # Log usage for monitoring
            self.logger.info(
                f"OpenAI API call successful - Model: {model}, Tokens: ~{len(clean_prompt)//4}"
            )

            return response.content

        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            return f"OpenAI API call failed: {str(e)}"

    def _run(
        self,
        prompt: str,
        model: str = "gpt-4",
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> str:
        """Synchronous wrapper for async OpenAI call"""
        return asyncio.run(self._arun(prompt, model, temperature, max_tokens))


class EmailNotificationTool(BaseTool):
    """
    Tool for sending email notifications for escalations, alerts, or customer communications.
    Supports templates and priority handling.
    """

    name: str = "email_notification_tool"
    description: str = """Send email notifications for escalations, alerts, or customer communications.
    Supports email templates, priority levels, and rich formatting."""
    args_schema: type[BaseModel] = EmailNotificationInput

    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)

        # Email templates for different scenarios
        self.templates = {
            "escalation": {
                "subject": "🚨 Customer Service Escalation - {case_id}",
                "body": """
A customer service case has been escalated and requires immediate attention.

Case Details:
- Case ID: {case_id}
- Customer: {customer_name} ({customer_email})
- Priority: {priority}
- Issue: {issue_summary}
- Escalation Reason: {escalation_reason}

Please review and take appropriate action within the SLA timeframe.

Best regards,
Customer Service AI System
                """,
            },
            "customer_followup": {
                "subject": "Follow-up on your support request - {ticket_id}",
                "body": """
Dear {customer_name},

Thank you for contacting our support team. We wanted to follow up on your recent inquiry.

Your Request:
- Ticket ID: {ticket_id}
- Subject: {subject}
- Status: {status}

{message}

If you have any additional questions, please don't hesitate to contact us.

Best regards,
Customer Support Team
                """,
            },
            "system_alert": {
                "subject": "⚠️ System Alert - {alert_type}",
                "body": """
System Alert Generated:

Alert Type: {alert_type}
Severity: {severity}
Time: {timestamp}
Description: {description}

{details}

Please investigate and take necessary action.

Automated System Monitoring
                """,
            },
        }

    @measure_execution_time
    async def _arun(
        self,
        to_email: str,
        subject: str,
        message: str,
        priority: str = "normal",
        template_name: Optional[str] = None,
    ) -> str:
        """Send email notification asynchronously"""
        try:
            # Validate email format
            if not self._is_valid_email(to_email):
                return f"Error: Invalid email address: {to_email}"

            # Sanitize inputs
            clean_subject = sanitize_input(subject, max_length=200)
            clean_message = sanitize_input(message, max_length=50000)

            # Apply template if specified
            if template_name and template_name in self.templates:
                template = self.templates[template_name]
                # For demo purposes, we'll use the message as template variables
                try:
                    template_vars = (
                        json.loads(clean_message)
                        if clean_message.startswith("{")
                        else {}
                    )
                    formatted_subject = template["subject"].format(**template_vars)
                    formatted_message = template["body"].format(**template_vars)
                    clean_subject = formatted_subject
                    clean_message = formatted_message
                except Exception:
                    # Fall back to original message if template formatting fails
                    pass

            # For demo purposes, we'll simulate email sending
            # In production, integrate with actual email service (SendGrid, AWS SES, etc.)
            email_data = {
                "to": to_email,
                "subject": clean_subject,
                "message": clean_message,
                "priority": priority,
                "timestamp": datetime.now().isoformat(),
                "template_used": template_name,
            }

            # Simulate different outcomes based on priority
            if priority == "high":
                delivery_time = "immediate"
            elif priority == "normal":
                delivery_time = "within 5 minutes"
            else:
                delivery_time = "within 15 minutes"

            # Log for monitoring
            self.logger.info(
                f"Email notification queued - To: {to_email}, Priority: {priority}, Template: {template_name}"
            )

            # Simulate successful sending
            return f"Email notification sent successfully to {to_email}. Priority: {priority}, Estimated delivery: {delivery_time}"

        except Exception as e:
            self.logger.error(f"Email notification failed: {e}")
            return f"Email notification failed: {str(e)}"

    def _run(
        self,
        to_email: str,
        subject: str,
        message: str,
        priority: str = "normal",
        template_name: Optional[str] = None,
    ) -> str:
        """Synchronous wrapper for async email sending"""
        return asyncio.run(
            self._arun(to_email, subject, message, priority, template_name)
        )

    def _is_valid_email(self, email: str) -> bool:
        """Basic email validation"""
        import re

        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, email))


class SlackNotificationTool(BaseTool):
    """
    Tool for sending Slack notifications to team channels or direct messages.
    Useful for real-time alerts and team coordination.
    """

    name: str = "slack_notification_tool"
    description: str = """Send Slack notifications to channels or users for real-time alerts and coordination.
    Supports different priority levels and thread replies."""
    args_schema: type[BaseModel] = SlackNotificationInput

    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)

        # Priority emoji mapping
        self.priority_emojis = {"low": "ℹ️", "normal": "📢", "high": "⚠️", "urgent": "🚨"}

        # Channel mapping for different types of notifications
        self.channel_mapping = {
            "escalations": "#customer-escalations",
            "alerts": "#system-alerts",
            "general": "#customer-service",
            "tech-issues": "#tech-support",
        }

    @measure_execution_time
    async def _arun(
        self,
        channel: str,
        message: str,
        priority: str = "normal",
        thread_ts: Optional[str] = None,
    ) -> str:
        """Send Slack notification asynchronously"""
        try:
            # Sanitize inputs
            clean_channel = sanitize_input(channel)
            clean_message = sanitize_input(
                message, max_length=4000
            )  # Slack message limit

            if not clean_channel or not clean_message:
                return "Error: Invalid channel or message provided"

            # Add priority emoji
            priority_emoji = self.priority_emojis.get(priority.lower(), "📢")
            formatted_message = f"{priority_emoji} {clean_message}"

            # For demo purposes, simulate Slack API call
            # In production, use slack_sdk or webhooks

            slack_payload = {
                "channel": clean_channel,
                "text": formatted_message,
                "priority": priority,
                "timestamp": datetime.now().isoformat(),
                "thread_ts": thread_ts,
            }

            # Simulate API call result
            if priority == "urgent":
                result_message = f"🚨 URGENT Slack notification sent to {clean_channel} - Message delivered with high visibility"
            elif priority == "high":
                result_message = (
                    f"⚠️ HIGH priority Slack notification sent to {clean_channel}"
                )
            else:
                result_message = f"📢 Slack notification sent to {clean_channel}"

            # Add thread info if applicable
            if thread_ts:
                result_message += f" (replied to thread {thread_ts})"

            # Log for monitoring
            self.logger.info(
                f"Slack notification sent - Channel: {clean_channel}, Priority: {priority}"
            )

            return result_message

        except Exception as e:
            self.logger.error(f"Slack notification failed: {e}")
            return f"Slack notification failed: {str(e)}"

    def _run(
        self,
        channel: str,
        message: str,
        priority: str = "normal",
        thread_ts: Optional[str] = None,
    ) -> str:
        """Synchronous wrapper for async Slack notification"""
        return asyncio.run(self._arun(channel, message, priority, thread_ts))


class WebhookTool(BaseTool):
    """
    Tool for sending data to external webhooks for integrations and notifications.
    Supports various HTTP methods and custom headers.
    """

    name: str = "webhook_tool"
    description: str = """Send HTTP requests to external webhooks for integrations and notifications.
    Supports GET, POST, PUT, DELETE methods with custom headers and payloads."""
    args_schema: type[BaseModel] = WebhookInput

    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)

    @measure_execution_time
    async def _arun(
        self,
        url: str,
        payload: Dict[str, Any],
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
    ) -> str:
        """Send webhook request asynchronously"""
        try:
            # Validate URL
            if not self._is_valid_url(url):
                return f"Error: Invalid URL: {url}"

            # Sanitize method
            method = method.upper().strip()
            if method not in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                return f"Error: Unsupported HTTP method: {method}"

            # Prepare headers
            request_headers = {
                "Content-Type": "application/json",
                "User-Agent": "CrewAI-CustomerService/1.0",
            }
            if headers:
                request_headers.update(headers)

            # Prepare payload
            if method == "GET":
                json_payload = None
                params = payload if payload else None
            else:
                json_payload = payload if payload else {}
                params = None

            # Make HTTP request
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                async with session.request(
                    method=method,
                    url=url,
                    json=json_payload,
                    params=params,
                    headers=request_headers,
                ) as response:

                    response_text = await response.text()
                    response_status = response.status

                    # Log request details
                    self.logger.info(
                        f"Webhook {method} to {url} - Status: {response_status}"
                    )

                    # Handle response
                    if 200 <= response_status < 300:
                        return f"Webhook {method} request successful - Status: {response_status}, Response: {response_text[:500]}"
                    elif 400 <= response_status < 500:
                        return f"Webhook {method} request failed - Client Error {response_status}: {response_text[:200]}"
                    elif 500 <= response_status < 600:
                        return f"Webhook {method} request failed - Server Error {response_status}: {response_text[:200]}"
                    else:
                        return f"Webhook {method} request completed - Status: {response_status}, Response: {response_text[:200]}"

        except asyncio.TimeoutError:
            self.logger.error(f"Webhook request timeout: {url}")
            return f"Webhook request timed out after {timeout} seconds"
        except aiohttp.ClientError as e:
            self.logger.error(f"Webhook request failed: {e}")
            return f"Webhook request failed: {str(e)}"
        except Exception as e:
            self.logger.error(f"Webhook tool error: {e}")
            return f"Webhook tool error: {str(e)}"

    def _run(
        self,
        url: str,
        payload: Dict[str, Any],
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
    ) -> str:
        """Synchronous wrapper for async webhook request"""
        return asyncio.run(self._arun(url, payload, method, headers, timeout))

    def _is_valid_url(self, url: str) -> bool:
        """Basic URL validation"""
        import re

        pattern = r"^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/[^?\s]*)?(?:\?[^#\s]*)?(?:#[^\s]*)?$"
        return bool(re.match(pattern, url))


# Additional specialized tools can be added here


class CRMIntegrationTool(BaseTool):
    """
    Tool for integrating with CRM systems to update customer records and create tasks.
    Placeholder for CRM-specific implementations.
    """

    name: str = "crm_integration_tool"
    description: str = """Integrate with CRM system to update customer records, create tasks, and sync data.
    Supports common CRM operations for customer service workflows."""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def _run(self, action: str, customer_id: str, data: Dict[str, Any]) -> str:
        """CRM integration (placeholder implementation)"""
        # This would integrate with actual CRM systems like Salesforce, HubSpot, etc.
        self.logger.info(
            f"CRM integration simulated - Action: {action}, Customer: {customer_id}"
        )
        return f"CRM integration completed - {action} for customer {customer_id}"


class AnalyticsTool(BaseTool):
    """
    Tool for logging analytics events and metrics for monitoring and reporting.
    """

    name: str = "analytics_tool"
    description: str = """Log analytics events and metrics for monitoring customer service performance.
    Tracks response times, resolution rates, and customer satisfaction."""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def _run(self, event_type: str, event_data: Dict[str, Any]) -> str:
        """Log analytics event"""
        try:
            analytics_event = {
                "event_type": event_type,
                "timestamp": datetime.now().isoformat(),
                "data": event_data,
            }

            # In production, send to analytics platform (Google Analytics, Mixpanel, etc.)
            self.logger.info(f"Analytics event logged: {analytics_event}")

            return f"Analytics event '{event_type}' logged successfully"

        except Exception as e:
            self.logger.error(f"Analytics logging failed: {e}")
            return f"Analytics logging failed: {str(e)}"


# Comprehensive External Integrations:

# OpenAITool: Direct OpenAI API calls with custom parameters

# EmailNotificationTool: Email notifications with template support

# SlackNotificationTool: Slack integration with priority handling

# WebhookTool: Generic HTTP webhook calls

# CRMIntegrationTool: CRM system integration placeholder

# AnalyticsTool: Event logging and metrics tracking
