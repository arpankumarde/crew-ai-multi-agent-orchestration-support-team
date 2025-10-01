# Escalation Agent
from crewai import Agent
from crewai.tools import tool
from langchain_openai import ChatOpenAI
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from src.core.config import settings


class EscalationAgent:
    """
    Agent responsible for identifying complex cases that require human intervention
    and managing the escalation process with appropriate routing and prioritization.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.ai_models.openai_model,
            temperature=0.2,
            api_key=settings.ai_models.openai_api_key,
        )

        # Escalation rules and thresholds
        self.escalation_rules = {
            "low_confidence_threshold": 0.6,
            "complex_intents": ["legal", "complaint", "refund", "cancel"],
            "urgent_keywords": [
                "emergency",
                "critical",
                "urgent",
                "immediately",
                "asap",
            ],
            "vip_tiers": ["platinum", "enterprise", "gold"],
            "technical_keywords": ["bug", "outage", "data loss", "security", "breach"],
            "billing_keywords": ["charged twice", "unauthorized", "fraud", "dispute"],
        }

        # Routing queues
        self.routing_queues = {
            "technical": "Technical Support Team",
            "billing": "Billing Specialist Team",
            "legal": "Legal Affairs Department",
            "vip": "VIP Customer Success Team",
            "general": "Tier 2 Support Team",
            "urgent": "Emergency Response Team",
        }

    @tool
    def escalation_assessment_tool(
        self,
        intent_data: Dict[str, Any],
        response_data: Dict[str, Any],
        customer_data: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Assess whether a case requires escalation based on multiple factors.

        Args:
            intent_data: Intent classification results
            response_data: Generated response information
            customer_data: Customer information and history

        Returns:
            Dictionary containing escalation assessment and recommendations
        """
        try:
            # Initialize escalation assessment
            escalation_score = 0.0
            escalation_reasons = []
            escalation_decision = False

            # Factor 1: Intent confidence
            confidence = intent_data.get("confidence", 1.0)
            if confidence < self.escalation_rules["low_confidence_threshold"]:
                escalation_score += 0.3
                escalation_reasons.append(
                    f"Low intent classification confidence: {confidence:.2f}"
                )

            # Factor 2: Intent complexity
            intent = intent_data.get("intent", "general")
            if intent in self.escalation_rules["complex_intents"]:
                escalation_score += 0.2
                escalation_reasons.append(f"Complex intent category: {intent}")

            # Factor 3: Urgency level
            urgency = intent_data.get("urgency", "medium")
            if urgency == "urgent":
                escalation_score += 0.4
                escalation_reasons.append(f"High urgency level: {urgency}")
            elif urgency == "high":
                escalation_score += 0.2
                escalation_reasons.append(f"Elevated urgency level: {urgency}")

            # Factor 4: Customer sentiment
            sentiment = intent_data.get("sentiment", "neutral")
            if sentiment == "negative":
                escalation_score += 0.2
                escalation_reasons.append("Negative customer sentiment detected")

            # Factor 5: Customer tier/VIP status
            customer_tier = "bronze"
            if customer_data and "customer" in customer_data:
                customer_tier = customer_data["customer"].get("customer_tier", "bronze")

            if customer_tier in self.escalation_rules["vip_tiers"]:
                escalation_score += 0.3
                escalation_reasons.append(f"VIP customer tier: {customer_tier}")

            # Factor 6: Response generation issues
            if response_data.get("requires_escalation", False):
                escalation_score += 0.4
                escalation_reasons.append("Response generator flagged for escalation")

            # Factor 7: Keyword analysis
            query = intent_data.get("original_query", "")
            escalation_score += self._analyze_escalation_keywords(
                query, escalation_reasons
            )

            # Factor 8: Historical context
            if customer_data and "support_tickets" in customer_data:
                recent_tickets = customer_data.get("recent_tickets", [])
                if len(recent_tickets) > 2:  # Multiple recent tickets
                    escalation_score += 0.2
                    escalation_reasons.append("Multiple recent support tickets")

            # Make escalation decision
            escalation_decision = escalation_score >= 0.5

            # Determine routing
            routing_recommendation = self._determine_routing(
                intent_data, customer_data, escalation_reasons
            )

            # Set priority
            priority = self._determine_priority(
                escalation_score, urgency, customer_tier
            )

            return {
                "escalation_needed": escalation_decision,
                "escalation_score": round(escalation_score, 2),
                "confidence": round(1 - escalation_score, 2),
                "escalation_reasons": escalation_reasons,
                "routing_queue": routing_recommendation,
                "priority_level": priority,
                "estimated_resolution_time": self._estimate_resolution_time(
                    intent, priority
                ),
                "recommended_agent_skills": self._recommend_agent_skills(
                    intent_data, escalation_reasons
                ),
                "escalation_context": {
                    "customer_tier": customer_tier,
                    "intent": intent,
                    "urgency": urgency,
                    "sentiment": sentiment,
                },
            }

        except Exception as e:
            print(f"Error in escalation assessment: {e}")
            return {
                "escalation_needed": True,
                "error": str(e),
                "routing_queue": "general",
                "priority_level": "medium",
            }

    @tool
    def priority_routing_tool(self, escalation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route escalated cases to appropriate teams with priority handling.

        Args:
            escalation_data: Escalation assessment results

        Returns:
            Dictionary containing routing information and next steps
        """
        try:
            routing_queue = escalation_data.get("routing_queue", "general")
            priority_level = escalation_data.get("priority_level", "medium")

            # Generate routing instructions
            routing_info = {
                "target_team": self.routing_queues.get(
                    routing_queue, "General Support"
                ),
                "priority": priority_level,
                "routing_code": self._generate_routing_code(
                    routing_queue, priority_level
                ),
                "sla_target": self._get_sla_target(priority_level),
                "escalation_path": self._get_escalation_path(routing_queue),
                "required_approvals": self._get_required_approvals(escalation_data),
                "next_steps": self._generate_next_steps(escalation_data),
                "handoff_notes": self._generate_handoff_notes(escalation_data),
            }

            return routing_info

        except Exception as e:
            print(f"Error in priority routing: {e}")
            return {
                "error": str(e),
                "target_team": "General Support",
                "priority": "medium",
            }

    def _analyze_escalation_keywords(
        self, query: str, escalation_reasons: List[str]
    ) -> float:
        """Analyze query for escalation trigger keywords"""
        score_increase = 0.0
        query_lower = query.lower()

        # Check for urgent keywords
        for keyword in self.escalation_rules["urgent_keywords"]:
            if keyword in query_lower:
                score_increase += 0.1
                escalation_reasons.append(f"Urgent keyword detected: {keyword}")

        # Check for technical keywords
        for keyword in self.escalation_rules["technical_keywords"]:
            if keyword in query_lower:
                score_increase += 0.1
                escalation_reasons.append(f"Technical issue keyword: {keyword}")

        # Check for billing keywords
        for keyword in self.escalation_rules["billing_keywords"]:
            if keyword in query_lower:
                score_increase += 0.2
                escalation_reasons.append(f"Billing concern keyword: {keyword}")

        return min(score_increase, 0.4)  # Cap at 0.4

    def _determine_routing(
        self, intent_data: Dict, customer_data: Dict, reasons: List[str]
    ) -> str:
        """Determine appropriate routing queue"""
        intent = intent_data.get("intent", "general")
        urgency = intent_data.get("urgency", "medium")

        # Check for VIP routing first
        if customer_data and "customer" in customer_data:
            customer_tier = customer_data["customer"].get("customer_tier", "bronze")
            if customer_tier in self.escalation_rules["vip_tiers"]:
                return "vip"

        # Route by intent
        if intent in ["billing", "refund"]:
            return "billing"
        elif intent in ["technical", "bug"]:
            return "technical"
        elif intent in ["legal", "complaint"]:
            return "legal"
        elif urgency == "urgent":
            return "urgent"
        else:
            return "general"

    def _determine_priority(
        self, escalation_score: float, urgency: str, customer_tier: str
    ) -> str:
        """Determine priority level for the escalated case"""
        if escalation_score >= 0.8 or urgency == "urgent":
            return "critical"
        elif escalation_score >= 0.6 or customer_tier in ["platinum", "enterprise"]:
            return "high"
        elif escalation_score >= 0.4:
            return "medium"
        else:
            return "low"

    def _estimate_resolution_time(self, intent: str, priority: str) -> str:
        """Estimate resolution time based on intent and priority"""
        base_times = {
            "billing": 2,
            "technical": 4,
            "account": 1,
            "product": 2,
            "legal": 24,
            "complaint": 4,
        }

        priority_multipliers = {"critical": 0.5, "high": 0.7, "medium": 1.0, "low": 1.5}

        base_time = base_times.get(intent, 3)
        multiplier = priority_multipliers.get(priority, 1.0)
        estimated_hours = base_time * multiplier

        if estimated_hours < 1:
            return "< 1 hour"
        elif estimated_hours < 24:
            return f"{int(estimated_hours)} hours"
        else:
            return f"{int(estimated_hours // 24)} days"

    def _recommend_agent_skills(
        self, intent_data: Dict, escalation_reasons: List[str]
    ) -> List[str]:
        """Recommend required agent skills for handling the escalated case"""
        skills = []
        intent = intent_data.get("intent", "general")

        skill_mapping = {
            "billing": ["billing_expertise", "payment_processing"],
            "technical": ["technical_troubleshooting", "product_knowledge"],
            "legal": ["legal_knowledge", "policy_expertise"],
            "complaint": ["conflict_resolution", "customer_retention"],
            "cancel": ["retention_specialist", "billing_expertise"],
        }

        if intent in skill_mapping:
            skills.extend(skill_mapping[intent])

        # Add based on escalation reasons
        if any("VIP" in reason for reason in escalation_reasons):
            skills.append("vip_handling")

        if any("negative sentiment" in reason.lower() for reason in escalation_reasons):
            skills.append("conflict_resolution")

        return skills

    def _generate_routing_code(self, queue: str, priority: str) -> str:
        """Generate a routing code for tracking"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        queue_code = queue[:3].upper()
        priority_code = priority[:1].upper()
        return f"ESC-{queue_code}-{priority_code}-{timestamp}"

    def _get_sla_target(self, priority: str) -> str:
        """Get SLA target based on priority"""
        sla_targets = {
            "critical": "1 hour response, 4 hours resolution",
            "high": "4 hours response, 24 hours resolution",
            "medium": "24 hours response, 72 hours resolution",
            "low": "72 hours response, 1 week resolution",
        }
        return sla_targets.get(priority, "24 hours response, 72 hours resolution")

    def _get_escalation_path(self, queue: str) -> List[str]:
        """Get escalation path for the queue"""
        escalation_paths = {
            "technical": [
                "L2 Technical Support",
                "Engineering Team",
                "Senior Engineering Manager",
            ],
            "billing": ["Billing Specialist", "Billing Manager", "Finance Director"],
            "legal": ["Legal Affairs", "Senior Legal Counsel", "Legal Director"],
            "vip": [
                "VIP Success Manager",
                "Customer Success Director",
                "VP Customer Success",
            ],
            "general": ["L2 Support", "Support Manager", "Support Director"],
        }
        return escalation_paths.get(queue, escalation_paths["general"])

    def _get_required_approvals(self, escalation_data: Dict) -> List[str]:
        """Determine required approvals for the escalation"""
        approvals = []
        priority = escalation_data.get("priority_level", "medium")
        reasons = escalation_data.get("escalation_reasons", [])

        if priority == "critical":
            approvals.append("Manager Approval Required")

        if any("refund" in reason.lower() for reason in reasons):
            approvals.append("Billing Manager Approval")

        if any("legal" in reason.lower() for reason in reasons):
            approvals.append("Legal Review Required")

        return approvals

    def _generate_next_steps(self, escalation_data: Dict) -> List[str]:
        """Generate next steps for the escalated case"""
        next_steps = [
            "Route to appropriate specialized team",
            "Notify customer of escalation and expected timeline",
            "Gather additional context and documentation",
        ]

        priority = escalation_data.get("priority_level", "medium")
        if priority == "critical":
            next_steps.insert(0, "Immediate manager notification required")

        return next_steps

    def _generate_handoff_notes(self, escalation_data: Dict) -> str:
        """Generate detailed handoff notes for the receiving team"""
        return f"""
        ESCALATION HANDOFF NOTES:
        
        Escalation Score: {escalation_data.get('escalation_score', 'N/A')}
        Priority: {escalation_data.get('priority_level', 'medium').upper()}
        Customer Tier: {escalation_data.get('escalation_context', {}).get('customer_tier', 'unknown')}
        
        Escalation Reasons:
        {chr(10).join('- ' + reason for reason in escalation_data.get('escalation_reasons', []))}
        
        Recommended Agent Skills:
        {', '.join(escalation_data.get('recommended_agent_skills', ['general_support']))}
        
        Please handle with appropriate priority and follow SLA guidelines.
        """

    def create_agent(self) -> Agent:
        """Create and configure the CrewAI agent"""
        return Agent(
            role="Support Case Manager",
            goal="Identify complex cases that require human intervention and manage escalation process",
            backstory="""You are a senior support case manager with expertise in identifying when 
            cases need human attention. You excel at prioritizing urgent issues and ensuring proper 
            handoffs to specialized teams. You understand the nuances of customer tiers, issue 
            complexity, and business impact. Your decisions ensure that customers receive appropriate 
            levels of support while optimizing team resources.""",
            llm=self.llm,
            tools=[self.escalation_assessment_tool, self.priority_routing_tool],
            max_iter=2,
            verbose=True,
            allow_delegation=False,
        )


# Role: Complex case identification and routing

# Capabilities:

# Multi-factor escalation scoring

# Priority-based routing decisions

# SLA target assignments

# Handoff documentation generation

# Key Features:

# Comprehensive escalation rules engine

# Customer tier-based routing (VIP handling)

# Specialized queue assignments (technical, billing, legal)

# Risk assessment and approval requirements
