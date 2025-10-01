# Quality Assurance Agent
from crewai import Agent
from crewai.tools import tool
from langchain_openai import ChatOpenAI
from typing import Dict, Any, List, Optional
import json
import re
from datetime import datetime
from src.core.config import settings


class QualityAssuranceAgent:
    """
    Agent responsible for reviewing and validating response quality, accuracy,
    and brand compliance before delivery to customers.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.ai_models.openai_model,
            temperature=0.1,
            api_key=settings.ai_models.openai_api_key,
        )

        # Quality criteria and thresholds
        self.quality_criteria = {
            "minimum_quality_score": 7.0,
            "auto_approve_threshold": 8.5,
            "max_response_length": 1000,
            "min_response_length": 50,
            "required_elements": ["greeting", "solution", "closing"],
            "prohibited_words": ["terrible", "awful", "stupid", "dumb", "worst"],
            "professional_tone_keywords": [
                "please",
                "thank you",
                "appreciate",
                "assist",
                "help",
            ],
        }

        # Brand guidelines
        self.brand_guidelines = {
            "tone": "professional_friendly",
            "voice": "helpful_empathetic",
            "formality": "business_casual",
            "company_values": [
                "customer_first",
                "transparency",
                "reliability",
                "innovation",
            ],
        }

        # Grammar and style checks
        self.grammar_patterns = {
            "common_mistakes": [
                (r"\bi\b", "I"),  # lowercase 'i' should be uppercase
                (r"\bthier\b", "their"),  # common typo
                (r"\byou're\b.*\bproblem\b", "your problem"),  # your vs you're
                (r"\bits\s+[a-z]", "it's"),  # its vs it's context
            ]
        }

    @tool
    def quality_check_tool(
        self,
        response: str,
        intent_data: Dict[str, Any],
        knowledge_sources: List[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive quality assessment of generated response.

        Args:
            response: Generated customer service response
            intent_data: Original intent classification data
            knowledge_sources: Sources used to generate response

        Returns:
            Dictionary containing quality assessment and recommendations
        """
        try:
            # Initialize quality assessment
            quality_assessment = {
                "overall_score": 0.0,
                "individual_scores": {},
                "issues_found": [],
                "improvements_suggested": [],
                "compliance_status": "pending",
                "approval_status": "pending",
                "review_timestamp": datetime.now().isoformat(),
            }

            # Perform individual quality checks
            quality_assessment["individual_scores"]["accuracy"] = self._check_accuracy(
                response, knowledge_sources
            )
            quality_assessment["individual_scores"]["tone"] = (
                self._check_tone_appropriateness(response, intent_data)
            )
            quality_assessment["individual_scores"]["completeness"] = (
                self._check_completeness(response, intent_data)
            )
            quality_assessment["individual_scores"]["clarity"] = self._check_clarity(
                response
            )
            quality_assessment["individual_scores"]["grammar"] = (
                self._check_grammar_style(response)
            )
            quality_assessment["individual_scores"]["brand_compliance"] = (
                self._check_brand_compliance(response)
            )
            quality_assessment["individual_scores"]["length_appropriateness"] = (
                self._check_response_length(response)
            )

            # Calculate overall score
            scores = quality_assessment["individual_scores"]
            overall_score = sum(scores.values()) / len(scores)
            quality_assessment["overall_score"] = round(overall_score, 1)

            # Determine approval status
            if overall_score >= self.quality_criteria["auto_approve_threshold"]:
                quality_assessment["approval_status"] = "auto_approved"
                quality_assessment["compliance_status"] = "compliant"
            elif overall_score >= self.quality_criteria["minimum_quality_score"]:
                quality_assessment["approval_status"] = "approved_with_notes"
                quality_assessment["compliance_status"] = "compliant"
            else:
                quality_assessment["approval_status"] = "requires_revision"
                quality_assessment["compliance_status"] = "non_compliant"

            # Generate improvement suggestions if needed
            if overall_score < self.quality_criteria["auto_approve_threshold"]:
                quality_assessment["improvements_suggested"] = (
                    self._generate_improvement_suggestions(
                        response,
                        quality_assessment["individual_scores"],
                        quality_assessment["issues_found"],
                    )
                )

            return quality_assessment

        except Exception as e:
            print(f"Error in quality check: {e}")
            return {
                "overall_score": 5.0,
                "error": str(e),
                "approval_status": "manual_review_required",
            }

    @tool
    def compliance_validation_tool(
        self, response: str, customer_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Validate response compliance with company policies and regulations.

        Args:
            response: Generated response to validate
            customer_data: Customer information for context

        Returns:
            Dictionary containing compliance validation results
        """
        try:
            compliance_results = {
                "overall_compliance": True,
                "checks_performed": [],
                "violations_found": [],
                "recommendations": [],
                "risk_level": "low",
            }

            # Privacy compliance check
            privacy_check = self._check_privacy_compliance(response, customer_data)
            compliance_results["checks_performed"].append("privacy_protection")
            if not privacy_check["compliant"]:
                compliance_results["violations_found"].extend(
                    privacy_check["violations"]
                )
                compliance_results["overall_compliance"] = False

            # Policy compliance check
            policy_check = self._check_policy_compliance(response)
            compliance_results["checks_performed"].append("policy_adherence")
            if not policy_check["compliant"]:
                compliance_results["violations_found"].extend(
                    policy_check["violations"]
                )
                compliance_results["overall_compliance"] = False

            # Legal compliance check
            legal_check = self._check_legal_compliance(response)
            compliance_results["checks_performed"].append("legal_compliance")
            if not legal_check["compliant"]:
                compliance_results["violations_found"].extend(legal_check["violations"])
                compliance_results["overall_compliance"] = False
                compliance_results["risk_level"] = "high"

            # Generate recommendations
            if compliance_results["violations_found"]:
                compliance_results["recommendations"] = (
                    self._generate_compliance_recommendations(
                        compliance_results["violations_found"]
                    )
                )

            return compliance_results

        except Exception as e:
            print(f"Error in compliance validation: {e}")
            return {
                "overall_compliance": False,
                "error": str(e),
                "risk_level": "unknown",
            }

    def _check_accuracy(
        self, response: str, knowledge_sources: List[Dict] = None
    ) -> float:
        """Check factual accuracy of the response"""
        accuracy_score = 8.0  # Base score

        # Check if response contradicts known information
        if knowledge_sources:
            # This would involve more sophisticated fact-checking
            # For now, we'll do basic consistency checks
            pass

        # Check for definitive statements without sources
        definitive_patterns = [
            r"always",
            r"never",
            r"guaranteed",
            r"100%",
            r"impossible",
        ]

        for pattern in definitive_patterns:
            if re.search(pattern, response.lower()):
                accuracy_score -= 0.5

        return max(accuracy_score, 5.0)

    def _check_tone_appropriateness(
        self, response: str, intent_data: Dict[str, Any]
    ) -> float:
        """Check if tone matches customer sentiment and intent"""
        tone_score = 8.0  # Base score

        sentiment = intent_data.get("sentiment", "neutral")
        urgency = intent_data.get("urgency", "medium")

        response_lower = response.lower()

        # Check for empathy markers when needed
        if sentiment == "negative":
            empathy_markers = ["understand", "sorry", "apologize", "concern"]
            if not any(marker in response_lower for marker in empathy_markers):
                tone_score -= 1.0

        # Check for urgency handling
        if urgency == "urgent":
            urgency_markers = ["immediately", "priority", "urgent", "right away"]
            if not any(marker in response_lower for marker in urgency_markers):
                tone_score -= 0.5

        # Check for professional language
        professional_markers = self.quality_criteria["professional_tone_keywords"]
        professional_count = sum(
            1 for marker in professional_markers if marker in response_lower
        )

        if professional_count < 2:
            tone_score -= 1.0

        return max(tone_score, 5.0)

    def _check_completeness(self, response: str, intent_data: Dict[str, Any]) -> float:
        """Check if response completely addresses the customer query"""
        completeness_score = 8.0  # Base score

        intent = intent_data.get("intent", "general")

        # Intent-specific completeness checks
        if intent == "account":
            required_elements = ["login", "password", "account"]
            if not any(element in response.lower() for element in required_elements):
                completeness_score -= 1.5

        elif intent == "billing":
            required_elements = ["payment", "billing", "charge", "invoice"]
            if not any(element in response.lower() for element in required_elements):
                completeness_score -= 1.5

        elif intent == "technical":
            required_elements = ["steps", "solution", "troubleshoot"]
            if not any(element in response.lower() for element in required_elements):
                completeness_score -= 1.5

        # Check for required response elements
        required_elements = self.quality_criteria["required_elements"]
        response_lower = response.lower()

        for element in required_elements:
            element_patterns = {
                "greeting": ["hello", "hi", "thank you", "thanks"],
                "solution": ["here", "follow", "step", "solution", "resolve"],
                "closing": ["help", "assist", "anything else", "questions"],
            }

            if element in element_patterns:
                if not any(
                    pattern in response_lower for pattern in element_patterns[element]
                ):
                    completeness_score -= 0.5

        return max(completeness_score, 5.0)

    def _check_clarity(self, response: str) -> float:
        """Check response clarity and readability"""
        clarity_score = 8.0  # Base score

        # Check sentence length (long sentences reduce clarity)
        sentences = response.split(".")
        long_sentences = [s for s in sentences if len(s.split()) > 25]

        if len(long_sentences) > len(sentences) * 0.3:  # More than 30% long sentences
            clarity_score -= 1.0

        # Check for jargon or technical terms without explanation
        technical_terms = ["API", "database", "server", "protocol", "configuration"]
        for term in technical_terms:
            if (
                term in response and f"({term}" not in response
            ):  # No explanation provided
                clarity_score -= 0.3

        # Check paragraph structure
        paragraphs = response.split("\n\n")
        if len(paragraphs) == 1 and len(response) > 300:  # Wall of text
            clarity_score -= 0.5

        return max(clarity_score, 5.0)

    def _check_grammar_style(self, response: str) -> float:
        """Check grammar and writing style"""
        grammar_score = 8.0  # Base score

        # Check for common grammar mistakes
        for mistake_pattern, correction in self.grammar_patterns["common_mistakes"]:
            if re.search(mistake_pattern, response):
                grammar_score -= 0.5

        # Check for proper capitalization
        sentences = response.split(".")
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not sentence[0].isupper():
                grammar_score -= 0.3

        # Check for excessive repetition
        words = response.lower().split()
        word_counts = {}
        for word in words:
            if len(word) > 4:  # Only check longer words
                word_counts[word] = word_counts.get(word, 0) + 1

        excessive_repetition = any(count > 4 for count in word_counts.values())
        if excessive_repetition:
            grammar_score -= 1.0

        return max(grammar_score, 5.0)

    def _check_brand_compliance(self, response: str) -> float:
        """Check compliance with brand guidelines"""
        brand_score = 8.0  # Base score

        # Check for prohibited words
        response_lower = response.lower()
        for word in self.quality_criteria["prohibited_words"]:
            if word in response_lower:
                brand_score -= 1.0

        # Check brand voice consistency
        brand_voice_markers = ["customer", "help", "support", "service", "solution"]
        brand_marker_count = sum(
            1 for marker in brand_voice_markers if marker in response_lower
        )

        if brand_marker_count < 2:
            brand_score -= 0.5

        return max(brand_score, 5.0)

    def _check_response_length(self, response: str) -> float:
        """Check if response length is appropriate"""
        length_score = 8.0  # Base score

        word_count = len(response.split())

        if word_count < self.quality_criteria["min_response_length"]:
            length_score -= 2.0  # Too short
        elif word_count > self.quality_criteria["max_response_length"]:
            length_score -= 1.0  # Too long

        return max(length_score, 5.0)

    def _check_privacy_compliance(
        self, response: str, customer_data: Dict = None
    ) -> Dict[str, Any]:
        """Check for privacy compliance issues"""
        privacy_violations = []

        # Check for sensitive information exposure
        sensitive_patterns = [
            (r"\b\d{4}-\d{4}-\d{4}-\d{4}\b", "Credit card number detected"),
            (r"\b\d{3}-\d{2}-\d{4}\b", "SSN pattern detected"),
            (
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "Email address in response",
            ),
        ]

        for pattern, violation_msg in sensitive_patterns:
            if re.search(pattern, response):
                privacy_violations.append(violation_msg)

        return {
            "compliant": len(privacy_violations) == 0,
            "violations": privacy_violations,
        }

    def _check_policy_compliance(self, response: str) -> Dict[str, Any]:
        """Check compliance with company policies"""
        policy_violations = []

        # Check for policy violations (simplified)
        problematic_phrases = [
            "we don't care",
            "not our problem",
            "that's impossible",
            "we can't help",
        ]

        response_lower = response.lower()
        for phrase in problematic_phrases:
            if phrase in response_lower:
                policy_violations.append(
                    f"Policy violation: '{phrase}' not aligned with customer-first values"
                )

        return {
            "compliant": len(policy_violations) == 0,
            "violations": policy_violations,
        }

    def _check_legal_compliance(self, response: str) -> Dict[str, Any]:
        """Check for legal compliance issues"""
        legal_violations = []

        # Check for potential legal issues
        legal_concerns = [
            "guarantee",
            "promise",
            "always works",
            "never fails",
            "legal advice",
        ]

        response_lower = response.lower()
        for concern in legal_concerns:
            if concern in response_lower:
                legal_violations.append(
                    f"Legal concern: Avoid absolute statements like '{concern}'"
                )

        return {"compliant": len(legal_violations) == 0, "violations": legal_violations}

    def _generate_improvement_suggestions(
        self, response: str, scores: Dict[str, float], issues: List[str]
    ) -> List[str]:
        """Generate specific improvement suggestions"""
        suggestions = []

        for criterion, score in scores.items():
            if score < 7.0:
                if criterion == "tone":
                    suggestions.append(
                        "Consider adding more empathetic language and acknowledgment of customer concerns"
                    )
                elif criterion == "completeness":
                    suggestions.append(
                        "Ensure all aspects of the customer query are addressed"
                    )
                elif criterion == "clarity":
                    suggestions.append(
                        "Break down complex information into clearer, shorter sentences"
                    )
                elif criterion == "grammar":
                    suggestions.append("Review for grammar and spelling errors")
                elif criterion == "brand_compliance":
                    suggestions.append(
                        "Align language with brand voice and avoid prohibited terms"
                    )

        return suggestions

    def _generate_compliance_recommendations(self, violations: List[str]) -> List[str]:
        """Generate compliance recommendations based on violations"""
        recommendations = []

        for violation in violations:
            if "privacy" in violation.lower():
                recommendations.append(
                    "Remove or mask any personally identifiable information"
                )
            elif "policy" in violation.lower():
                recommendations.append("Revise language to align with company policies")
            elif "legal" in violation.lower():
                recommendations.append("Remove absolute statements and legal advice")

        return recommendations

    def create_agent(self) -> Agent:
        """Create and configure the CrewAI agent"""
        return Agent(
            role="Quality Control Specialist",
            goal="Review and validate response quality, accuracy, and brand compliance",
            backstory="""You are a quality assurance specialist focused on maintaining high 
            standards for customer communications. You ensure all responses are accurate, helpful, 
            and align with company guidelines. You have a keen eye for detail and understand the 
            importance of brand consistency, legal compliance, and customer satisfaction. You take 
            pride in maintaining the company's reputation through excellent communication quality.""",
            llm=self.llm,
            tools=[self.quality_check_tool, self.compliance_validation_tool],
            max_iter=3,
            verbose=True,
            allow_delegation=False,
        )


# Role: Response validation and compliance checking

# Capabilities:

# Multi-dimensional quality scoring

# Brand compliance validation

# Grammar and style checking

# Privacy and legal compliance verification

# Key Features:

# 7-point quality assessment system

# Automated approval workflows

# Compliance violation detection

# Improvement suggestion generation
