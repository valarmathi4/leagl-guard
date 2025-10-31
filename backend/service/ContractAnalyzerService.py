import json
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

from models.ContractAnalysisModel import ContractAnalysisRequest
from models.ContractAnalysisResponseModel import ContractAnalysisResponse, ClauseFlag, ComplianceFeedback
from models.ComplianceRiskScore import ComplianceRiskScore
from utils.law_loader import LawLoader
from service.RegulatoryEngineService import RegulatoryEngineService
from utils.ai_client import WatsonXClient, WatsonXConfig, GeminiClient, GeminiConfig
from utils.ai_client.exceptions import ConfigurationError, APIError, AuthenticationError 

logger = logging.getLogger(__name__)

class ContractAnalyzerService:
    def __init__(self):
        self.law_loader = LawLoader()
        self.regulatory_engine = RegulatoryEngineService(self.law_loader)
        self.watsonx_client = None
        self.gemini_client = None
        self.ai_provider = None  # Track which provider is active
        
        # Try to initialize Gemini client first (preferred)
        try:
            gemini_config = GeminiConfig.from_environment()
            self.gemini_client = GeminiClient(gemini_config)
            self.ai_provider = "gemini"
            logger.info(f"Gemini AI client initialized successfully with model: {gemini_config.model_name}")
        except ConfigurationError as e:
            logger.info(f"Gemini client not configured: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini client: {e}")
        
        # Fall back to IBM WatsonX if Gemini is not available
        if not self.gemini_client:
            try:
                config = WatsonXConfig.from_environment()
                self.watsonx_client = WatsonXClient(config)
                self.ai_provider = "watsonx"
                logger.info("IBM WatsonX Granite client initialized successfully.")
            except ConfigurationError as e:
                logger.warning(f"Failed to initialize WatsonX client due to configuration: {e}")
                self.watsonx_client = None
            except Exception as e:
                logger.error(f"Failed to initialize WatsonX client: {e}")
                self.watsonx_client = None
        
        # Log the active AI provider
        if self.ai_provider:
            logger.info(f"Active AI Provider: {self.ai_provider.upper()}")
        else:
            logger.warning("No AI provider available - will use fallback analysis")
                
    async def analyze_contract(self, request: ContractAnalysisRequest) -> ContractAnalysisResponse:
        """
        Main contract analysis orchestrator with enhanced content-aware analysis.
        """
        try:
            # 1. Pre-process and clean the contract text
            cleaned_contract = self._preprocess_contract_text(request.text)
            logger.info(f"Contract preprocessing complete. Original length: {len(request.text)}, Cleaned length: {len(cleaned_contract)}")
            
            # 2. Analyze contract structure and content type
            contract_metadata = self._analyze_contract_metadata(cleaned_contract)
            logger.info(f"Contract analysis: Type={contract_metadata['type']}, Sections={len(contract_metadata['sections'])}, Has_Data_Processing={contract_metadata['has_data_processing']}")
            
            # 3. Get the applicable compliance rules from our engine
            jurisdiction = request.jurisdiction or "MY"
            compliance_checklist = self.regulatory_engine.get_compliance_checklist(
                jurisdiction=jurisdiction,
                contract_type=contract_metadata['type']
            )

            # 4. Determine which AI service to use
            use_ai = self.ai_provider is not None
            
            if use_ai:
                try:
                    if self.ai_provider == "gemini":
                        logger.info("Using Google Gemini AI for contract analysis")
                        ai_response_text = self._get_gemini_analysis(
                            cleaned_contract, contract_metadata, compliance_checklist, jurisdiction
                        )
                        logger.info(f"Gemini AI Response received: {ai_response_text[:200]}...")
                    else:  # watsonx
                        logger.info("Using IBM WatsonX Granite AI for contract analysis")
                        ai_response_text = self._get_granite_analysis_with_context(
                            cleaned_contract, contract_metadata, compliance_checklist, jurisdiction
                        )
                        logger.info(f"IBM Granite AI Response received: {ai_response_text[:200]}...")
                    
                    # Validate the AI response
                    if self._is_ai_response_minimal(ai_response_text):
                        logger.info(f"{self.ai_provider.upper()} response appears minimal, enhancing with domain expertise")
                        ai_response_text = self._get_intelligent_mock_analysis(
                            cleaned_contract, contract_metadata, compliance_checklist, jurisdiction
                        )
                    else:
                        logger.info(f"{self.ai_provider.upper()} provided comprehensive analysis - using AI response directly")
                        
                except (APIError, AuthenticationError) as e:
                    logger.error(f"{self.ai_provider.upper()} API error: {e}")
                    ai_response_text = self._get_intelligent_mock_analysis(
                        cleaned_contract, contract_metadata, compliance_checklist, jurisdiction
                    )
                except Exception as e:
                    logger.error(f"Unexpected error calling {self.ai_provider.upper()}: {e}")
                    ai_response_text = self._get_intelligent_mock_analysis(
                        cleaned_contract, contract_metadata, compliance_checklist, jurisdiction
                    )
            else:
                logger.warning("No AI provider available - using intelligent fallback analysis")
                ai_response_text = self._get_intelligent_mock_analysis(
                    cleaned_contract, contract_metadata, compliance_checklist, jurisdiction
                )

            # 5. Parse and validate the AI's JSON response
            try:
                ai_json = json.loads(ai_response_text)
                ai_json = self._clean_ai_response(ai_json, jurisdiction, cleaned_contract)
                
                # Ensure we have meaningful analysis
                if not ai_json.get("compliance_issues") and not ai_json.get("flagged_clauses"):
                    logger.info("No issues found, generating comprehensive compliance analysis")
                    ai_json = self._generate_comprehensive_analysis(
                        cleaned_contract, contract_metadata, jurisdiction
                    )
                
                # Convert law_id to law for compatibility
                if "compliance_issues" in ai_json:
                    for issue in ai_json["compliance_issues"]:
                        if "law_id" in issue and "law" not in issue:
                            issue["law"] = issue.pop("law_id")
                
                # Ensure required fields
                ai_json.setdefault("summary", "Analysis complete.")
                ai_json.setdefault("flagged_clauses", [])
                ai_json.setdefault("compliance_issues", [])
                
                return ContractAnalysisResponse(
                    summary=ai_json["summary"],
                    flagged_clauses=[ClauseFlag(**flag) for flag in ai_json["flagged_clauses"]],
                    compliance_issues=[ComplianceFeedback(**issue) for issue in ai_json["compliance_issues"]],
                    jurisdiction=jurisdiction
                )
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI JSON response: {ai_response_text}")
                return ContractAnalysisResponse(
                    summary="Error: AI response could not be parsed as valid JSON.", 
                    flagged_clauses=[],
                    compliance_issues=[],
                    jurisdiction=jurisdiction
                )
            except Exception as e:
                logger.error(f"Failed to create response model from AI data: {e}")
                return ContractAnalysisResponse(
                    summary="Error: Failed to process AI response into structured format.", 
                    flagged_clauses=[],
                    compliance_issues=[],
                    jurisdiction=jurisdiction
                )

        except Exception as e:
            logger.error(f"Contract analysis failed: {str(e)}")
            raise

    def _preprocess_contract_text(self, contract_text: str) -> str:
        """
        Enhanced preprocessing to remove formatting artifacts and focus on actual contract content.
        """
        # Remove markdown headers and formatting
        text = re.sub(r'^#{1,6}\s+.*$', '', contract_text, flags=re.MULTILINE)
        
        # Remove markdown emphasis
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        
        # Remove markdown lists that aren't contract content
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags if present
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove excessive whitespace but preserve paragraph structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove common non-contract content patterns
        non_contract_patterns = [
            r'(?i)^(contract analysis|legal review|summary|overview):.*$',
            r'(?i)^(note|disclaimer|warning):.*$',
            r'(?i)^(created by|generated by|analyzed by):.*$',
        ]
        
        for pattern in non_contract_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        
        # Clean up and normalize
        text = text.strip()
        
        # If text is too short after cleaning, it might not be a real contract
        if len(text.strip()) < 100:
            logger.warning("Contract text appears to be very short after preprocessing")
        
        return text
    
    def _analyze_contract_metadata(self, contract_text: str) -> Dict[str, Any]:
        """
        Analyze contract structure and content to understand what type of contract this is
        and what specific legal areas it touches.
        """
        text_lower = contract_text.lower()
        
        # Detect contract type based on content analysis
        contract_type = "General"
        type_indicators = {
            "Employment": ["employee", "employer", "employment", "salary", "wage", "termination", "workplace", "job duties", "position", "work schedule"],
            "Service": ["services", "service provider", "client", "deliverables", "scope of work", "performance"],
            "NDA": ["confidential", "non-disclosure", "proprietary", "trade secret", "confidentiality"],
            "Rental": ["tenant", "landlord", "rent", "lease", "property", "premises", "rental"],
            "Sales": ["purchase", "buyer", "seller", "goods", "products", "sale", "delivery"],
            "Partnership": ["partner", "partnership", "joint venture", "profit sharing", "collaboration"]
        }
        
        max_score = 0
        for contract_type_candidate, keywords in type_indicators.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > max_score:
                max_score = score
                contract_type = contract_type_candidate
        
        # Analyze content areas
        has_data_processing = any(term in text_lower for term in [
            "personal data", "data processing", "privacy", "information", "data subject", "gdpr", "pdpa"
        ])
        
        has_termination_clauses = any(term in text_lower for term in [
            "termination", "terminate", "end of contract", "expiry", "cancellation"
        ])
        
        has_payment_terms = any(term in text_lower for term in [
            "payment", "pay", "salary", "wage", "compensation", "fee", "amount", "money"
        ])
        
        has_liability_clauses = any(term in text_lower for term in [
            "liable", "liability", "damages", "indemnify", "responsibility", "loss"
        ])
        
        has_ip_clauses = any(term in text_lower for term in [
            "intellectual property", "copyright", "patent", "trademark", "work product", "invention"
        ])
        
        # Extract meaningful sections
        sections = self._extract_meaningful_sections(contract_text)
        
        # Detect jurisdiction indicators in the text
        jurisdiction_indicators = {
            "MY": ["malaysia", "malaysian", "kuala lumpur", "ringgit", "employment act 1955"],
            "SG": ["singapore", "singaporean", "sgd", "singapore dollar"],
            "US": ["united states", "usd", "dollar", "state of", "california", "new york"],
            "EU": ["european union", "gdpr", "euro", "eur"]
        }
        
        detected_jurisdictions = []
        for jurisdiction, indicators in jurisdiction_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                detected_jurisdictions.append(jurisdiction)
        
        return {
            "type": contract_type,
            "has_data_processing": has_data_processing,
            "has_termination_clauses": has_termination_clauses,
            "has_payment_terms": has_payment_terms,
            "has_liability_clauses": has_liability_clauses,
            "has_ip_clauses": has_ip_clauses,
            "sections": sections,
            "detected_jurisdictions": detected_jurisdictions,
            "word_count": len(contract_text.split()),
            "is_substantial": len(contract_text.strip()) > 500  # Flag for substantial contracts
        }
    
    def _extract_meaningful_sections(self, contract_text: str) -> List[Dict[str, str]]:
        """
        Extract meaningful contract sections, ignoring formatting artifacts.
        """
        sections = []
        
        # Try different section detection patterns
        section_patterns = [
            r'\n\s*(\d+)\.\s*([A-Z][^.\n]*[.:]?)\s*\n((?:[^\n]+\n?)*?)(?=\n\s*\d+\.|$)',  # Numbered sections
            r'\n\s*([A-Z][A-Z\s]{2,}):?\s*\n((?:[^\n]+\n?)*?)(?=\n\s*[A-Z][A-Z\s]{2,}:|$)',  # ALL CAPS headers
            r'\n\s*([A-Z][^.\n]*):?\s*\n((?:[^\n]+\n?)*?)(?=\n\s*[A-Z][^.\n]*:|$)'  # Title Case headers
        ]
        
        for pattern in section_patterns:
            matches = re.finditer(pattern, contract_text, re.MULTILINE | re.DOTALL)
            for match in matches:
                if len(match.groups()) >= 2:
                    title = match.group(1).strip() if len(match.groups()) > 1 else f"Section {match.group(1)}"
                    content = match.group(2).strip() if len(match.groups()) > 2 else match.group(2).strip()
                    
                    # Skip sections that are too short or look like formatting artifacts
                    if len(content) > 50 and not self._is_formatting_artifact(title, content):
                        sections.append({
                            "title": title,
                            "content": content,
                            "word_count": len(content.split())
                        })
            
            if sections:  # If we found sections with one pattern, use those
                break
        
        # Fallback: split by paragraphs if no clear sections found
        if not sections:
            paragraphs = re.split(r'\n\s*\n\s*', contract_text)
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph.strip()) > 100:  # Only substantial paragraphs
                    sections.append({
                        "title": f"Paragraph {i+1}",
                        "content": paragraph.strip(),
                        "word_count": len(paragraph.split())
                    })
        
        return sections
    
    def _is_formatting_artifact(self, title: str, content: str) -> bool:
        """
        Check if a section is likely a formatting artifact rather than actual contract content.
        """
        title_lower = title.lower()
        content_lower = content.lower()
        
        # Skip common non-contract sections
        artifact_indicators = [
            "summary", "analysis", "review", "note", "disclaimer", "generated", 
            "created", "overview", "introduction", "conclusion", "appendix",
            "table of contents", "index", "header", "footer"
        ]
        
        if any(indicator in title_lower for indicator in artifact_indicators):
            return True
        
        # Skip sections that are mostly formatting
        if len(content.strip()) < 20:
            return True
        
        # Skip sections with too many special characters (likely formatting)
        special_char_ratio = sum(1 for c in content if not c.isalnum() and not c.isspace()) / len(content)
        if special_char_ratio > 0.3:
            return True
        
        return False
    
    def _get_granite_analysis_with_context(self, contract_text: str, metadata: Dict[str, Any], 
                                         compliance_checklist: Dict[str, Any], jurisdiction: str) -> str:
        """
        Enhanced prompting for IBM Granite with contract context and intelligent analysis.
        Optimized for TechXchange Hackathon submission.
        """
        if not self.watsonx_client:
            logger.warning("IBM WatsonX client not available, falling back to intelligent analysis")
            return self._get_intelligent_mock_analysis(contract_text, metadata, compliance_checklist, jurisdiction)
        
        try:
            logger.info("Engaging IBM Granite model for advanced legal analysis")
            
            # Use the enhanced prompt designed for Granite
            granite_response = self.watsonx_client.analyze_contract(
                contract_text=contract_text,
                compliance_checklist=compliance_checklist
            )
            
            logger.info(f"IBM Granite analysis completed successfully: {len(granite_response)} characters")
            
            # Validate Granite response quality
            if self._is_granite_response_minimal(granite_response):
                logger.info("IBM Granite response needs enhancement, combining with domain expertise")
                # Enhance minimal Granite response with our intelligent analysis
                enhanced_response = self._enhance_granite_response(
                    granite_response, contract_text, metadata, jurisdiction
                )
                return enhanced_response
            
            return granite_response
            
        except (APIError, AuthenticationError) as e:
            logger.error(f"IBM Granite API error: {e}")
            return self._get_intelligent_mock_analysis(contract_text, metadata, compliance_checklist, jurisdiction)
        except Exception as e:
            logger.error(f"Unexpected error with IBM Granite: {e}")
            return self._get_intelligent_mock_analysis(contract_text, metadata, compliance_checklist, jurisdiction)
    
    def _get_gemini_analysis(self, contract_text: str, metadata: Dict[str, Any], 
                            compliance_checklist: Dict[str, Any], jurisdiction: str) -> str:
        """
        Enhanced analysis using Google Gemini AI with contract context and intelligent prompting.
        """
        if not self.gemini_client:
            logger.warning("Gemini client not available, falling back to intelligent analysis")
            return self._get_intelligent_mock_analysis(contract_text, metadata, compliance_checklist, jurisdiction)
        
        try:
            logger.info("Engaging Google Gemini model for advanced legal analysis")
            
            # Use the Gemini client's analyze_contract method
            gemini_response = self.gemini_client.analyze_contract(
                contract_text=contract_text,
                compliance_checklist=compliance_checklist
            )
            
            logger.info(f"Gemini analysis completed successfully: {len(gemini_response)} characters")
            
            # Validate Gemini response quality
            if self._is_ai_response_minimal(gemini_response):
                logger.info("Gemini response needs enhancement, combining with domain expertise")
                # Enhance minimal response with our intelligent analysis
                enhanced_response = self._enhance_ai_response(
                    gemini_response, contract_text, metadata, jurisdiction
                )
                return enhanced_response
            
            return gemini_response
            
        except (APIError, AuthenticationError) as e:
            logger.error(f"Gemini API error: {e}")
            return self._get_intelligent_mock_analysis(contract_text, metadata, compliance_checklist, jurisdiction)
        except Exception as e:
            logger.error(f"Unexpected error with Gemini: {e}")
            return self._get_intelligent_mock_analysis(contract_text, metadata, compliance_checklist, jurisdiction)
    
    def _build_enhanced_granite_prompt(self, contract_text: str, metadata: Dict[str, Any], jurisdiction: str) -> str:
        """
        Build an intelligent prompt for Granite that focuses on actual contract content.
        """
        jurisdiction_name = {
            "MY": "Malaysia", "SG": "Singapore", "EU": "European Union", "US": "United States"
        }.get(jurisdiction, jurisdiction)
        
        prompt = f"""Analyze this {metadata['type']} contract for {jurisdiction_name} compliance.

CONTRACT CONTEXT:
- Type: {metadata['type']} contract
- Word Count: {metadata['word_count']} words
- Sections Identified: {len(metadata['sections'])}
- Data Processing Elements: {'Yes' if metadata['has_data_processing'] else 'No'}
- Termination Clauses: {'Yes' if metadata['has_termination_clauses'] else 'No'}
- Payment Terms: {'Yes' if metadata['has_payment_terms'] else 'No'}

ANALYSIS REQUIREMENTS:
1. Focus ONLY on actual contract clauses, ignore any headers, titles, or formatting
2. For flagged clauses, extract the EXACT clause text that has the issue
3. Only flag issues that are contextually relevant to the specific clause content
4. Provide jurisdiction-specific compliance analysis for {jurisdiction_name}

IMPORTANT: Do not flag issues on section headers, markdown formatting, or non-contractual text.
Only analyze substantive contractual provisions and obligations.

Return analysis in JSON format with summary, flagged_clauses, and compliance_issues arrays."""
        
        return prompt
    
    def _get_intelligent_mock_analysis(self, contract_text: str, metadata: Dict[str, Any], 
                                     compliance_checklist: Dict[str, Any], jurisdiction: str) -> str:
        """
        Intelligent mock analysis that adapts to the specific contract content and avoids repetitive outputs.
        Enhanced for IBM Granite compatibility with rigorous legal analysis.
        """
        flagged_clauses = []
        compliance_issues = []
        seen_issues = set()  # Track unique issues to prevent duplicates
        
        logger.info(f"Starting rigorous intelligent analysis for {metadata['type']} contract with {len(metadata['sections'])} sections")
        
        # Analyze the entire contract holistically first to identify key areas
        contract_analysis = self._perform_comprehensive_contract_analysis(
            contract_text, metadata, jurisdiction
        )
        
        # Extract unique flagged clauses with strict criteria
        for issue in contract_analysis.get('flagged_clauses', []):
            issue_key = f"{issue['issue'][:50]}_{issue['severity']}"  # Create unique key
            if issue_key not in seen_issues and self._is_substantive_legal_issue(issue, contract_text):
                seen_issues.add(issue_key)
                flagged_clauses.append(issue)
        
        # Generate compliance issues based on actual contract content and gaps
        compliance_issues = self._generate_smart_compliance_issues(
            contract_text, metadata, jurisdiction
        )
        
        # Validate and clean compliance issues to prevent malformed data
        compliance_issues = self._validate_compliance_issues(compliance_issues, jurisdiction)
        
        # Apply critical analysis - only flag serious violations
        flagged_clauses = self._apply_critical_legal_analysis(flagged_clauses, metadata, jurisdiction)
        
        # Generate contextual summary
        summary = self._generate_contextual_summary(
            flagged_clauses, compliance_issues, metadata, jurisdiction
        )
        
        logger.info(f"Rigorous analysis complete: {len(flagged_clauses)} unique flagged clauses, {len(compliance_issues)} compliance issues")
        
        return json.dumps({
            "summary": summary,
            "flagged_clauses": flagged_clauses,
            "compliance_issues": compliance_issues
        })
    
    def _analyze_section_intelligently(self, section: Dict[str, str], metadata: Dict[str, Any], 
                                     jurisdiction: str) -> List[Dict[str, Any]]:
        """
        Intelligent section analysis that only flags relevant issues on appropriate content.
        """
        issues = []
        title = section['title'].lower()
        content = section['content'].lower()
        
        # Only analyze if this is substantial contract content
        if section['word_count'] < 10:
            return issues
        
        # Employment-specific analysis
        if metadata['type'] == 'Employment':
            if 'termination' in title or 'termination' in content:
                if 'without notice' in content and 'misconduct' not in content:
                    issues.append({
                        "clause_text": self._extract_relevant_clause(section['content'], 'without notice'),
                        "issue": f"Termination without notice may not comply with {jurisdiction} employment law minimum notice requirements",
                        "severity": "high"
                    })
                
                # Check for inadequate notice periods
                notice_match = re.search(r'(\d+)\s*(day|week|month)', content)
                if notice_match:
                    notice_period = int(notice_match.group(1))
                    period_type = notice_match.group(2)
                    
                    if period_type == 'day' and notice_period < 7:
                        issues.append({
                            "clause_text": self._extract_relevant_clause(section['content'], notice_match.group(0)),
                            "issue": f"Notice period of {notice_period} days may be insufficient under {jurisdiction} employment standards",
                            "severity": "medium"
                        })
            
            if ('wage' in content or 'salary' in content) and 'overtime' not in content:
                issues.append({
                    "clause_text": self._extract_relevant_clause(section['content'], 'wage salary'),
                    "issue": f"Compensation clause lacks overtime provisions required under {jurisdiction} employment law",
                    "severity": "medium"
                })
        
        # Data processing analysis (only for contracts that actually process data)
        if metadata['has_data_processing'] and ('data' in content or 'information' in content):
            if 'personal data' in content and 'consent' not in content:
                issues.append({
                    "clause_text": self._extract_relevant_clause(section['content'], 'personal data'),
                    "issue": f"Data processing clause lacks explicit consent mechanisms required under {jurisdiction} privacy law",
                    "severity": "high"
                })
        
        # Liability analysis
        if 'liability' in content or 'damages' in content:
            # Look for liability caps that might be too low
            amount_match = re.search(r'(\d+(?:,\d+)*)', content)
            if amount_match:
                amount = int(amount_match.group(1).replace(',', ''))
                if 'liability' in content and amount < 10000:
                    issues.append({
                        "clause_text": self._extract_relevant_clause(section['content'], amount_match.group(0)),
                        "issue": f"Liability limitation of {amount} may be insufficient for this type of contract",
                        "severity": "low"
                    })
        
        return issues
    
    def _extract_relevant_clause(self, section_content: str, search_terms: str) -> str:
        """
        Extract the most relevant sentence or clause that contains the search terms.
        """
        sentences = re.split(r'[.!?]+', section_content)
        
        # Find the sentence containing the search terms
        for sentence in sentences:
            if any(term.lower() in sentence.lower() for term in search_terms.split()):
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 10:
                    return clean_sentence + "."
        
        # Fallback: return first substantial sentence
        for sentence in sentences:
            if len(sentence.strip()) > 20:
                return sentence.strip() + "."
        
        # Last resort: return truncated content
        return section_content[:150] + "..." if len(section_content) > 150 else section_content
    
    def _generate_smart_compliance_issues(self, contract_text: str, metadata: Dict[str, Any], 
                                        jurisdiction: str) -> List[Dict[str, Any]]:
        """
        Generate critical compliance issues that are specific to the contract type and jurisdiction.
        Enhanced for IBM Granite compatibility with specific statutory references.
        Only generates issues for laws that are actually applicable to the contract and jurisdiction.
        """
        issues = []
        text_lower = contract_text.lower()
        
        logger.info(f"Generating compliance issues for {metadata['type']} contract in {jurisdiction} jurisdiction")
        
        # Employment contract compliance (ONLY for employment contracts in Malaysia)
        if metadata['type'] == 'Employment' and jurisdiction == 'MY':
            requirements = []
            recommendations = []
            
            # Enhanced Employment Act 1955 compliance checks
            
            # 1. Termination notice provisions (Section 12)
            notice_found = re.search(r'(?:notice|termination).*(?:\d+.*(?:week|month|day))', text_lower)
            if not notice_found:
                requirements.append("Termination notice provisions do not meet Employment Act 1955 Section 12 minimum requirements")
                recommendations.append("Add termination clause specifying minimum notice: 2 weeks for <2 years service, 4 weeks for >2 years service")
            
            # 2. Working hours limitations (Section 60A)
            hours_violation = False
            hours_matches = re.finditer(r'(\d+).*hours?.*(?:per|each).*(?:day|week)', text_lower)
            for match in hours_matches:
                hours = int(match.group(1))
                period = match.group(0)
                if ('day' in period and hours > 8) or ('week' in period and hours > 48):
                    hours_violation = True
                    requirements.append(f"Working hours exceed Employment Act 1955 Section 60A maximum (8 hours/day, 48 hours/week)")
                    recommendations.append("Adjust working hours to comply with statutory maximums under Section 60A")
                    break
            
            # 3. Overtime compensation (Section 60A)
            if not re.search(r'overtime.*(?:compensation|payment|rate|1\.5|time.*half)', text_lower):
                requirements.append("Missing overtime compensation violates Employment Act 1955 Section 60A")
                recommendations.append("Include overtime payment at minimum 1.5x normal hourly rate as mandated by Section 60A")
            
            # 4. Annual leave entitlement (Section 60E)
            leave_found = re.search(r'annual.*leave.*(\d+).*day|(\d+).*day.*annual.*leave', text_lower)
            if not leave_found:
                requirements.append("Missing annual leave entitlement violates Employment Act 1955 Section 60E")
                recommendations.append("Specify annual leave entitlement: 8 days (<2 years), 12 days (2-5 years), 16 days (>5 years)")
            elif leave_found:
                # Check if leave days are sufficient
                leave_days = int(leave_found.group(1) or leave_found.group(2))
                if leave_days < 8:
                    requirements.append(f"Annual leave of {leave_days} days below Employment Act 1955 Section 60E minimum")
                    recommendations.append("Increase annual leave to statutory minimum of 8 days as required by Section 60E")
            
            # 5. Rest days and public holidays (Sections 60C, 60D)
            if not re.search(r'rest.*day|public.*holiday|gazetted.*holiday', text_lower):
                requirements.append("Missing rest day and public holiday provisions required under Employment Act 1955 Sections 60C, 60D")
                recommendations.append("Include provisions for weekly rest days and gazetted public holidays as mandated")
            
            # 6. Probation period limits (Section 11)
            probation_match = re.search(r'probation.*(\d+).*month|(\d+).*month.*probation', text_lower)
            if probation_match:
                probation_months = int(probation_match.group(1) or probation_match.group(2))
                if probation_months > 6:
                    requirements.append(f"Probation period of {probation_months} months exceeds Employment Act 1955 Section 11 maximum")
                    recommendations.append("Reduce probation period to maximum 6 months as required by Section 11")
            
            # 7. Minimum wage compliance
            salary_matches = re.finditer(r'salary.*rm\s*(\d+(?:,\d+)*)|rm\s*(\d+(?:,\d+)*).*salary', text_lower)
            for match in salary_matches:
                salary_str = (match.group(1) or match.group(2)).replace(',', '')
                salary_amount = int(salary_str)
                if salary_amount < 1500:
                    requirements.append(f"Monthly salary of RM{salary_amount} below minimum wage of RM1,500")
                    recommendations.append("Adjust salary to meet Minimum Wages Order 2022 requirement of RM1,500")
                    break
            
            # 8. EPF and SOCSO contributions
            if not re.search(r'epf|employees.*provident.*fund', text_lower):
                requirements.append("Missing EPF contribution provisions required under EPF Act 1991")
                recommendations.append("Include EPF contribution clause (11% employee, 12-13% employer)")
            
            if not re.search(r'socso|social.*security|employment.*injury', text_lower):
                requirements.append("Missing SOCSO contribution provisions required under SOCSO Act 1969")
                recommendations.append("Include SOCSO contribution clause for employment injury and invalidity coverage")
            
            if requirements:
                issues.append({
                    "law": "EMPLOYMENT_ACT_MY",
                    "missing_requirements": requirements,
                    "recommendations": recommendations
                })
                logger.info(f"Generated comprehensive Employment Act MY compliance issue with {len(requirements)} requirements")
        
        # Data protection compliance (only for contracts that actually process personal data)
        if metadata['has_data_processing']:
            # Determine the correct data protection law based on jurisdiction
            law_mapping = {
                "MY": "PDPA_MY",
                "SG": "PDPA_SG", 
                "EU": "GDPR_EU",
                "US": "CCPA_US"
            }
            
            law_id = law_mapping.get(jurisdiction)
            if not law_id:
                logger.warning(f"No data protection law defined for jurisdiction '{jurisdiction}'")
                return issues
            
            law_name = {
                "MY": "Personal Data Protection Act 2010", 
                "SG": "Personal Data Protection Act 2012", 
                "EU": "GDPR", 
                "US": "CCPA"
            }.get(jurisdiction, "privacy law")
            
            requirements = []
            recommendations = []
            
            # Critical data protection violations only
            if not re.search(r'consent.*(?:explicit|written|informed)', text_lower):
                requirements.append(f"Missing explicit consent mechanisms required under {law_name}")
                recommendations.append(f"Implement clear, informed consent procedures before collecting personal data")
            
            if jurisdiction in ['MY', 'SG'] and not re.search(r'data subject.*rights', text_lower):
                requirements.append(f"Missing data subject rights provisions required under {law_name}")
                recommendations.append("Include data subject rights: access, correction, and withdrawal of consent")
            
            if jurisdiction == 'EU' and not re.search(r'(?:access|rectification|erasure|portability)', text_lower):
                requirements.append("Missing GDPR data subject rights (access, rectification, erasure, portability)")
                recommendations.append("Implement all GDPR data subject rights as mandated by Articles 15-20")
            
            if jurisdiction == 'US' and not re.search(r'(?:consumer.*rights|privacy.*rights|opt.*out)', text_lower):
                # Enhanced CCPA-specific violation detection
                requirements, recommendations = self._detect_ccpa_violations(contract_text, text_lower)
                if requirements:
                    requirements.append("Missing consumer privacy rights disclosure required under CCPA")
                    recommendations.append("Add consumer privacy notice with opt-out mechanisms for data selling")
                else:
                    requirements = ["Missing consumer privacy rights disclosure required under CCPA"]
                    recommendations = ["Add consumer privacy notice with opt-out mechanisms for data selling"]
            
            if requirements:
                issues.append({
                    "law": law_id,
                    "missing_requirements": requirements,
                    "recommendations": recommendations
                })
                logger.info(f"Generated {law_id} compliance issue with {len(requirements)} requirements")
        
        # For US service contracts, check general contract law compliance (not employment law!)
        if jurisdiction == 'US' and not metadata['has_data_processing'] and metadata['type'] in ['Service', 'General']:
            # For US service contracts, focus on consumer protection and general contract law
            # Do NOT apply employment law or foreign privacy laws
            logger.info(f"US {metadata['type']} contract detected - checking general contract compliance only")
            
            # Only generate CCPA issues if there's actual data processing
            # Don't generate any employment law issues for service contracts
            
        logger.info(f"Generated {len(issues)} total compliance issues for {jurisdiction} {metadata['type']} contract")
        return issues
    
    def _generate_contextual_summary(self, flagged_clauses: List[Dict], compliance_issues: List[Dict], 
                                   metadata: Dict[str, Any], jurisdiction: str) -> str:
        """
        Generate a contextual summary based on the specific contract and findings.
        """
        jurisdiction_name = {
            "MY": "Malaysia", "SG": "Singapore", "EU": "European Union", "US": "United States"
        }.get(jurisdiction, jurisdiction)
        
        total_issues = len(flagged_clauses) + len(compliance_issues)
        contract_type = metadata['type']
        
        if total_issues == 0:
            return f"Review of this {contract_type.lower()} contract for {jurisdiction_name} compliance found no significant issues requiring immediate attention."
        
        # Count severity levels
        high_severity = sum(1 for clause in flagged_clauses if clause.get('severity') == 'high')
        medium_severity = sum(1 for clause in flagged_clauses if clause.get('severity') == 'medium')
        
        summary_parts = []
        
        # Main assessment
        if high_severity > 0:
            summary_parts.append(f"This {contract_type.lower()} contract contains {high_severity} high-priority compliance issue{'s' if high_severity != 1 else ''} for {jurisdiction_name}")
        elif medium_severity > 0:
            summary_parts.append(f"This {contract_type.lower()} contract has {medium_severity} moderate compliance concern{'s' if medium_severity != 1 else ''} for {jurisdiction_name}")
        else:
            summary_parts.append(f"This {contract_type.lower()} contract has minor compliance gaps for {jurisdiction_name}")
        
        # Add specific areas of concern
        concern_areas = []
        if compliance_issues:
            laws_affected = [issue['law'] for issue in compliance_issues]
            if 'EMPLOYMENT_ACT_MY' in laws_affected:
                concern_areas.append("employment law compliance")
            if any('PDPA' in law for law in laws_affected):
                concern_areas.append("data protection requirements")
            if any('GDPR' in law for law in laws_affected):
                concern_areas.append("GDPR compliance")
        
        if concern_areas:
            summary_parts.append(f"requiring attention in: {', '.join(concern_areas)}")
        
        # Add recommendation level
        if high_severity > 0 or len(compliance_issues) > 2:
            summary_parts.append("Recommend legal review before contract execution.")
        elif total_issues > 0:
            summary_parts.append("Consider addressing identified issues to ensure full compliance.")
        
        return " ".join(summary_parts)
    
    def _generate_comprehensive_analysis(self, contract_text: str, metadata: Dict[str, Any], 
                                       jurisdiction: str) -> Dict[str, Any]:
        """
        Generate comprehensive analysis when no issues are initially found.
        """
        # This should rarely be called with the new intelligent analysis
        return {
            "summary": f"Comprehensive review of this {metadata['type'].lower()} contract completed for {jurisdiction} jurisdiction.",
            "flagged_clauses": [],
            "compliance_issues": []
        }

    async def calculate_risk_score(self, analysis_response: ContractAnalysisResponse) -> ComplianceRiskScore:
            """
            Calculate comprehensive risk scoring with proper weighting for severity and violations.
            Uses IBM Granite AI for enhanced risk assessment when available.
            """
            violation_categories = set()
            jurisdiction_risks = {}
            financial_risk = 0.0
            
            # Enhanced risk calculation with proper severity weighting
            base_risk_score = 100
            risk_deductions = 0
            
            # Analyze compliance issues with proper weighting
            for issue in analysis_response.compliance_issues or []:
                violation_categories.add(issue.law)
                
                # Calculate law-specific risk
                law_risk = self._get_risk_from_law(issue.law, len(issue.missing_requirements))
                financial_risk += law_risk
                
                # Deduct points based on number of missing requirements
                missing_count = len(issue.missing_requirements)
                
                # CCPA violations are scored more severely due to strict liability
                if issue.law == "CCPA_US":
                    if missing_count >= 5:
                        risk_deductions += 40  # Critical CCPA violations
                    elif missing_count >= 3:
                        risk_deductions += 30  # Severe CCPA violations
                    elif missing_count >= 2:
                        risk_deductions += 20  # Moderate CCPA violations
                    else:
                        risk_deductions += 12  # Minor CCPA violations
                else:
                    # Standard scoring for other laws
                    if missing_count >= 4:
                        risk_deductions += 25  # Severe compliance gaps
                    elif missing_count >= 2:
                        risk_deductions += 15  # Moderate compliance gaps
                    else:
                        risk_deductions += 8   # Minor compliance gaps
                
                jurisdiction = analysis_response.jurisdiction or "MY"
                jurisdiction_risks[jurisdiction] = jurisdiction_risks.get(jurisdiction, 0) + law_risk
            
            # Analyze flagged clauses with severity weighting
            for clause in analysis_response.flagged_clauses or []:
                severity = getattr(clause, 'severity', 'medium')
                
                if severity == 'high':
                    risk_deductions += 20
                    financial_risk += 15000
                elif severity == 'medium':
                    risk_deductions += 12
                    financial_risk += 8000
                else:  # low severity
                    risk_deductions += 5
                    financial_risk += 3000
            
            # Calculate final risk score (capped at 0-100)
            final_score = max(0, min(100, base_risk_score - risk_deductions))
            
            # Determine risk level
            if final_score >= 80:
                risk_level = "Low"
            elif final_score >= 60:
                risk_level = "Medium"
            elif final_score >= 40:
                risk_level = "High"
            else:
                risk_level = "Critical"
            
            return ComplianceRiskScore(
                overall_score=final_score,
                financial_risk_estimate=financial_risk,
                violation_categories=list(violation_categories),
                jurisdiction_risks=jurisdiction_risks
            )
    
    def _get_risk_from_law(self, law_id: str, violation_count: int) -> float:
        """Calculate financial risk based on law type and violation severity."""
        base_risks = {
            "EMPLOYMENT_ACT_MY": 12000,
            "PDPA_MY": 20000,
            "PDPA_SG": 25000,
            "GDPR_EU": 50000,
            "CCPA_US": 75000  # Increased base risk for CCPA due to $7,500 per violation penalty
        }
        
        base_risk = base_risks.get(law_id, 10000)
        
        # CCPA has particularly harsh penalties - scale more aggressively
        if law_id == "CCPA_US":
            return base_risk * (1 + (violation_count * 0.5))  # More aggressive scaling for CCPA
        
        return base_risk * (1 + (violation_count * 0.3))  # Scale by violation count
    
    def _preprocess_contract_text(self, contract_text: str) -> str:
        """
        Enhanced preprocessing to remove formatting artifacts and focus ONLY on actual contract content.
        This is the key fix to prevent markdown headers from being analyzed as contract clauses.
        """
        # Step 1: Remove markdown headers completely (these are NOT contract content)
        text = re.sub(r'^#{1,6}\s+.*$', '', contract_text, flags=re.MULTILINE)
        
        # Step 2: Remove markdown formatting but keep the content
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Code spans
        
        # Step 3: Remove markdown list markers that aren't part of contract content
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+(?=[A-Z])', '', text, flags=re.MULTILINE)
        
        # Step 4: Remove HTML tags if present
        text = re.sub(r'<[^>]+>', '', text)
        
        # Step 5: Remove common document metadata and non-contract content
        non_contract_patterns = [
            r'(?i)^(contract analysis|legal review|summary|overview|analysis|review):.*$',
            r'(?i)^(note|disclaimer|warning|important):.*$',
            r'(?i)^(created by|generated by|analyzed by|document|title):.*$',
            r'(?i)^(version|date|status|author):.*$',
            r'(?i)^(page \d+|header|footer):.*$',
            r'(?i)^\s*(summary|conclusion|recommendations?):\s*$',  # Section headers only
            r'(?i)^\s*-{3,}\s*$',  # Markdown dividers
            r'(?i)^\s*={3,}\s*$',  # Underlines
        ]
        
        for pattern in non_contract_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        
        # Step 6: Clean up whitespace but preserve paragraph structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'[ \t]+', ' ', text)              # Multiple spaces to single
        text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)  # Leading whitespace
        
        # Step 7: Remove lines that are clearly formatting artifacts or too short to be meaningful
        lines = text.split('\n')
        meaningful_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                meaningful_lines.append('')  # Preserve paragraph breaks
                continue
                
            # Skip lines that are clearly not contract content
            if (len(line) < 10 or  # Too short
                line.isupper() and len(line.split()) < 5 or  # Short ALL CAPS (likely headers)
                re.match(r'^[^\w]*$', line) or  # Only special characters
                re.match(r'^(page|section|article|chapter)\s*\d+\s*$', line, re.IGNORECASE) or  # Page/section numbers
                line.count('_') > len(line) // 3 or  # Too many underscores (formatting)
                line.count('-') > len(line) // 3):   # Too many dashes (formatting)
                continue
            
            meaningful_lines.append(line)
        
        text = '\n'.join(meaningful_lines)
        
        # Step 8: Final cleanup
        text = text.strip()
        text = re.sub(r'\n{3,}', '\n\n', text)  # No more than double newlines
        
        # Step 9: Validation - if text is too short after cleaning, it might not be a real contract
        if len(text.strip()) < 100:
            logger.warning(f"Contract text appears to be very short after preprocessing: {len(text.strip())} characters")
        
        logger.info(f"Text preprocessing complete. Removed formatting artifacts. Clean text length: {len(text)}")
        return text
    
    def _analyze_contract_metadata(self, contract_text: str) -> Dict[str, Any]:
        """
        Enhanced contract analysis that focuses ONLY on substantive contract content.
        Ignores formatting artifacts and document metadata.
        """
        text_lower = contract_text.lower()
        
        # Enhanced contract type detection with more sophisticated analysis
        contract_type = "General"
        type_scores = {}
        
        type_indicators = {
            "Employment": {
                "strong": ["employee", "employer", "employment", "position", "job duties", "workplace", "termination of employment"],
                "moderate": ["salary", "wage", "work schedule", "benefits", "leave", "resignation"],
                "weak": ["work", "duties", "responsibilities"]
            },
            "Service": {
                "strong": ["service provider", "client", "deliverables", "scope of work", "statement of work"],
                "moderate": ["services", "performance", "completion", "milestone"],
                "weak": ["provide", "deliver", "complete"]
            },
            "Privacy": {
                "strong": ["privacy policy", "privacy notice", "california consumer privacy act", "ccpa", "personal information collection", "privacy rights"],
                "moderate": ["personal information", "data collection", "consumer rights", "privacy", "california resident"],
                "weak": ["collect", "information", "data"]
            },
            "NDA": {
                "strong": ["non-disclosure", "confidentiality agreement", "trade secret", "proprietary information"],
                "moderate": ["confidential", "proprietary", "confidentiality"],
                "weak": ["information", "disclosure"]
            },
            "Rental": {
                "strong": ["landlord", "tenant", "lease agreement", "rental agreement", "premises"],
                "moderate": ["rent", "lease", "property", "occupancy"],
                "weak": ["monthly", "deposit"]
            },
            "Sales": {
                "strong": ["purchase agreement", "sale agreement", "buyer", "seller", "transfer of ownership"],
                "moderate": ["purchase", "sale", "goods", "products", "delivery"],
                "weak": ["buy", "sell", "payment"]
            },
            "Partnership": {
                "strong": ["partnership agreement", "joint venture", "business partnership", "profit sharing"],
                "moderate": ["partner", "partnership", "collaboration", "venture"],
                "weak": ["together", "joint", "share"]
            }
        }
        
        # Calculate weighted scores for each contract type
        for contract_type_candidate, indicators in type_indicators.items():
            score = 0
            
            # Strong indicators (weight: 3)
            for indicator in indicators["strong"]:
                if indicator in text_lower:
                    score += 3
            
            # Moderate indicators (weight: 2)
            for indicator in indicators["moderate"]:
                if indicator in text_lower:
                    score += 2
            
            # Weak indicators (weight: 1)
            for indicator in indicators["weak"]:
                if indicator in text_lower:
                    score += 1
            
            type_scores[contract_type_candidate] = score
        
        # Select the type with the highest score (minimum threshold of 3)
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] >= 3:
                contract_type = best_type
        
        logger.info(f"Contract type analysis: {type_scores} -> Selected: {contract_type}")
        
        # Enhanced content analysis with better detection including California/CCPA specific content
        has_data_processing = any(phrase in text_lower for phrase in [
            "personal data", "data processing", "data protection", "privacy policy",
            "data subject", "gdpr", "pdpa", "collect information", "process data",
            "personal information", "california", "ccpa", "consumer rights", "privacy rights",
            "collect personal", "share data", "sell data", "sale of personal information"
        ])
        
        has_termination_clauses = any(phrase in text_lower for phrase in [
            "termination", "terminate this", "end of contract", "contract expiry",
            "cancellation", "breach of contract", "dissolution"
        ])
        
        has_payment_terms = any(phrase in text_lower for phrase in [
            "payment terms", "payment schedule", "compensation", "remuneration",
            "salary", "wage", "fee", "amount due", "invoice"
        ])
        
        has_liability_clauses = any(phrase in text_lower for phrase in [
            "liability", "liable for", "damages", "indemnify", "indemnification",
            "limitation of liability", "hold harmless", "responsibility for"
        ])
        
        has_ip_clauses = any(phrase in text_lower for phrase in [
            "intellectual property", "copyright", "patent", "trademark",
            "work product", "invention", "proprietary rights", "trade secret"
        ])
        
        # Extract meaningful sections with improved filtering
        sections = self._extract_contract_sections_only(contract_text)
        
        # Enhanced jurisdiction detection with CCPA-specific indicators
        jurisdiction_indicators = {
            "MY": ["malaysia", "malaysian", "kuala lumpur", "ringgit", "rm ", "employment act 1955", "companies act 2016"],
            "SG": ["singapore", "singaporean", "sgd", "singapore dollar", "companies act singapore"],
            "US": ["united states", "usd", "us dollar", "state of california", "state of new york", "delaware", "california", "ccpa", "california consumer privacy act", "california resident"],
            "EU": ["european union", "gdpr", "euro", "eur", "brussels", "directive 95/46/ec"]
        }
        
        detected_jurisdictions = []
        for jurisdiction, indicators in jurisdiction_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                detected_jurisdictions.append(jurisdiction)
        
        # Calculate actual contract substance metrics
        word_count = len([word for word in contract_text.split() if len(word) > 2])  # Exclude short words
        sentence_count = len(re.findall(r'[.!?]+', contract_text))
        
        # Determine if this is a substantial contract worth analyzing
        is_substantial = (
            len(contract_text.strip()) > 500 and
            word_count > 100 and
            sentence_count > 5 and
            len(sections) > 0
        )
        
        metadata = {
            "type": contract_type,
            "type_confidence": type_scores.get(contract_type, 0),
            "has_data_processing": has_data_processing,
            "has_termination_clauses": has_termination_clauses,
            "has_payment_terms": has_payment_terms,
            "has_liability_clauses": has_liability_clauses,
            "has_ip_clauses": has_ip_clauses,
            "sections": sections,
            "detected_jurisdictions": detected_jurisdictions,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "is_substantial": is_substantial
        }
        
        logger.info(f"Contract metadata analysis complete: {contract_type} contract with {len(sections)} substantive sections")
        return metadata
    
    def _extract_contract_sections_only(self, contract_text: str) -> List[Dict[str, str]]:
        """
        Extract ONLY meaningful contract sections, completely ignoring formatting artifacts.
        This is crucial for preventing analysis of document headers and formatting.
        """
        sections = []
        
        # Multiple patterns to detect genuine contract sections vs formatting
        section_patterns = [
            # Pattern 1: Numbered contract sections (e.g., "1. Definitions", "2.1 Scope")
            r'\n\s*(\d+(?:\.\d+)*)\.\s+([A-Z][^.\n]{5,50}?)\s*\n((?:(?!\n\s*\d+(?:\.\d+)*\.)(?:[^\n]+\n?))*)',
            
            # Pattern 2: Lettered sections (e.g., "A. Terms", "B. Conditions")
            r'\n\s*([A-Z])\.\s+([A-Z][^.\n]{5,50}?)\s*\n((?:(?!\n\s*[A-Z]\.)(?:[^\n]+\n?))*)',
            
            # Pattern 3: Named sections in contracts (e.g., "WHEREAS", "NOW THEREFORE")
            r'\n\s*(WHEREAS|NOW THEREFORE|WITNESSETH|RECITALS?)\s*[,:]\s*\n((?:[^\n]+\n?)*?)(?=\n\s*(?:WHEREAS|NOW THEREFORE|WITNESSETH|\d+\.|[A-Z]{3,})|$)',
            
            # Pattern 4: Title case sections with substantial content
            r'\n\s*([A-Z][a-z][^.\n]{10,80}?)\s*[:.]?\s*\n((?:[^\n]+\n?){3,}?)(?=\n\s*[A-Z][a-z][^.\n]{10,80}?[:.]?\s*\n|$)'
        ]
        
        for pattern_idx, pattern in enumerate(section_patterns):
            matches = list(re.finditer(pattern, contract_text, re.MULTILINE | re.DOTALL))
            
            for match in matches:
                groups = match.groups()
                
                if len(groups) >= 2:
                    if len(groups) == 3:
                        section_id, title, content = groups
                    else:
                        title, content = groups
                        section_id = f"Section {len(sections) + 1}"
                    
                    title = title.strip()
                    content = content.strip()
                    
                    # Strict filtering for genuine contract content
                    if self._is_genuine_contract_section(title, content):
                        sections.append({
                            "id": section_id,
                            "title": title,
                            "content": content,
                            "word_count": len(content.split()),
                            "pattern_used": pattern_idx + 1
                        })
            
            # If we found good sections with one pattern, prioritize those
            if len(sections) >= 3:
                break
        
        # Fallback for contracts without clear section headers
        if len(sections) < 2:
            paragraphs = self._extract_meaningful_paragraphs(contract_text)
            sections.extend(paragraphs)
        
        # Sort by appearance order and limit to most substantial sections
        sections = sorted(sections, key=lambda x: x.get('word_count', 0), reverse=True)[:10]
        
        logger.info(f"Extracted {len(sections)} genuine contract sections for analysis")
        return sections
    
    def _is_genuine_contract_section(self, title: str, content: str) -> bool:
        """
        Determine if a section is genuine contract content vs formatting artifact.
        This is the key method to prevent analysis of non-contractual content.
        """
        title_lower = title.lower()
        content_lower = content.lower()
        
        # Immediately reject if title indicates non-contract content
        non_contract_titles = [
            "summary", "analysis", "review", "note", "disclaimer", "generated",
            "created", "overview", "introduction", "conclusion", "appendix",
            "table of contents", "index", "header", "footer", "document",
            "title", "subject", "re:", "from:", "to:", "date:", "version",
            "page", "confidential", "draft", "final", "approved"
        ]
        
        if any(nc_title in title_lower for nc_title in non_contract_titles):
            logger.debug(f"Rejected section '{title}' - contains non-contract title indicator")
            return False
        
        # Reject very short content (likely formatting)
        if len(content.strip()) < 50:
            logger.debug(f"Rejected section '{title}' - content too short ({len(content.strip())} chars)")
            return False
        
        # Reject content with too many special characters (formatting artifacts)
        special_char_ratio = sum(1 for c in content if not c.isalnum() and not c.isspace()) / max(len(content), 1)
        if special_char_ratio > 0.4:
            logger.debug(f"Rejected section '{title}' - too many special characters ({special_char_ratio:.2f})")
            return False
        
        # Reject if content is mostly uppercase (likely headers/formatting)
        upper_ratio = sum(1 for c in content if c.isupper()) / max(len([c for c in content if c.isalpha()]), 1)
        if upper_ratio > 0.7:
            logger.debug(f"Rejected section '{title}' - mostly uppercase ({upper_ratio:.2f})")
            return False
        
        # Require minimum word count and sentence structure
        word_count = len(content.split())
        sentence_count = len(re.findall(r'[.!?]+', content))
        
        if word_count < 15 or sentence_count < 1:
            logger.debug(f"Rejected section '{title}' - insufficient content (words: {word_count}, sentences: {sentence_count})")
            return False
        
        # Positive indicators of contract content
        contract_indicators = [
            "party", "parties", "agreement", "contract", "shall", "will",
            "hereby", "whereas", "therefore", "obligations", "rights",
            "terms", "conditions", "provision", "clause", "section"
        ]
        
        indicator_count = sum(1 for indicator in contract_indicators if indicator in content_lower)
        if indicator_count < 2:
            logger.debug(f"Rejected section '{title}' - insufficient contract indicators ({indicator_count})")
            return False
        
        logger.debug(f"Accepted section '{title}' - genuine contract content (words: {word_count}, indicators: {indicator_count})")
        return True
    
    def _extract_meaningful_paragraphs(self, contract_text: str) -> List[Dict[str, str]]:
        """
        Extract meaningful paragraphs when no clear sections are found.
        """
        paragraphs = re.split(r'\n\s*\n\s*', contract_text)
        meaningful_paragraphs = []
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            
            if self._is_genuine_contract_section(f"Paragraph {i+1}", paragraph):
                meaningful_paragraphs.append({
                    "id": f"P{i+1}",
                    "title": f"Paragraph {i+1}",
                    "content": paragraph,
                    "word_count": len(paragraph.split()),
                    "pattern_used": 0  # Fallback pattern
                })
        
        return meaningful_paragraphs
    
    def _build_enhanced_granite_prompt(self, contract_text: str, metadata: Dict[str, Any], jurisdiction: str) -> str:
        """
        Build an enhanced prompt that prevents analysis of formatting artifacts.
        """
        jurisdiction_name = {
            "MY": "Malaysia", "SG": "Singapore", "EU": "European Union", "US": "United States"
        }.get(jurisdiction, jurisdiction)
        
        # Create content focus guidance
        content_focus = f"""CRITICAL INSTRUCTION: ONLY analyze substantive contractual provisions. 

IGNORE completely:
- Document headers, titles, or section names
- Markdown formatting (###, **, *, etc.)
- Document metadata or analysis summaries
- Any content that appears to be document formatting rather than contract terms

ANALYZE only:
- Actual contractual obligations and rights
- Specific terms and conditions
- Substantive legal provisions
- Binding commitments between parties

CONTRACT CONTEXT:
- Type: {metadata['type']} contract (confidence: {metadata.get('type_confidence', 0)})
- Substantive sections: {len([s for s in metadata['sections'] if s['word_count'] > 20])}
- Contains data processing: {'Yes' if metadata['has_data_processing'] else 'No'}
- Contains termination clauses: {'Yes' if metadata['has_termination_clauses'] else 'No'}
- Contains payment terms: {'Yes' if metadata['has_payment_terms'] else 'No'}
- Word count: {metadata['word_count']} meaningful words

ANALYSIS REQUIREMENTS FOR {jurisdiction_name}:
1. Extract ONLY the exact contractual clause text that has legal issues
2. Focus on binding obligations, not descriptive text
3. Identify specific legal non-compliance, not formatting issues
4. Provide actionable legal recommendations
5. Consider jurisdiction-specific requirements for {jurisdiction_name}

OUTPUT FORMAT: Valid JSON with summary, flagged_clauses, and compliance_issues arrays."""
        
        return content_focus

    def _is_ai_response_minimal(self, response_text: str) -> bool:
        """
        Enhanced detection of minimal AI responses that need augmentation.
        Works for both Gemini and IBM Granite responses.
        """
        try:
            response_json = json.loads(response_text)
            
            # Check for truly minimal responses
            flagged_count = len(response_json.get("flagged_clauses", []))
            compliance_count = len(response_json.get("compliance_issues", []))
            summary_length = len(response_json.get("summary", ""))
            
            # Consider minimal if very few issues found AND short summary
            is_minimal = (
                (flagged_count + compliance_count) < 2 and
                summary_length < 100
            )
            
            logger.info(f"Response assessment: {flagged_count} flagged, {compliance_count} compliance issues, {summary_length} char summary -> {'Minimal' if is_minimal else 'Comprehensive'}")
            return is_minimal
            
        except json.JSONDecodeError:
            logger.warning("Could not parse AI response as JSON - treating as minimal")
            return True
    
    # Keep the old method name for backward compatibility
    def _is_granite_response_minimal(self, response_text: str) -> bool:
        """Backward compatibility wrapper for _is_ai_response_minimal"""
        return self._is_ai_response_minimal(response_text)
    
    def _clean_ai_response(self, ai_json: Dict[str, Any], jurisdiction: str, contract_text: str) -> Dict[str, Any]:
        """
        Enhanced cleaning that removes any analysis of formatting artifacts and fixes malformed law fields.
        """
        cleaned_flagged = []
        cleaned_compliance = []
        
        # Clean flagged clauses - remove any that are clearly formatting artifacts
        for flag in ai_json.get("flagged_clauses", []):
            clause_text = flag.get("clause_text", "")
            
            # Skip if clause is clearly a formatting artifact
            if not self._is_substantive_clause(clause_text):
                logger.debug(f"Removing flagged clause - not substantive: {clause_text[:50]}...")
                continue
            
            cleaned_flagged.append(flag)
        
        # Clean compliance issues and fix malformed law fields
        valid_laws = ["EMPLOYMENT_ACT_MY", "PDPA_MY", "PDPA_SG", "GDPR_EU", "CCPA_US"]
        
        for issue in ai_json.get("compliance_issues", []):
            # Fix malformed law field (contains multiple laws separated by |)
            law_field = issue.get("law", "")
            
            # CRITICAL: Block Malaysian Employment Act from being applied to US contracts
            if jurisdiction == "US" and law_field == "EMPLOYMENT_ACT_MY":
                logger.error(f"BLOCKED in AI cleaning: Malaysian Employment Act cannot be applied to US jurisdiction contract - removing issue")
                continue
            
            # CRITICAL: Block any foreign employment law application
            if law_field == "EMPLOYMENT_ACT_MY" and jurisdiction != "MY":
                logger.error(f"BLOCKED in AI cleaning: Malaysian Employment Act cannot be applied to {jurisdiction} jurisdiction contract - removing issue")
                continue
            
            if "|" in law_field:
                # Split and take the first valid law based on jurisdiction and contract type
                law_options = law_field.split("|")
                fixed_law = self._select_appropriate_law(law_options, jurisdiction, ai_json)
                issue["law"] = fixed_law
                logger.warning(f"Fixed malformed law field: '{law_field}' -> '{fixed_law}'")
            elif law_field not in valid_laws:
                # Set appropriate law based on jurisdiction
                issue["law"] = self._get_default_law_for_jurisdiction(jurisdiction)
                logger.warning(f"Invalid law field '{law_field}' replaced with '{issue['law']}'")
            
            # Additional validation - ensure the final law is appropriate for jurisdiction
            final_law = issue.get("law", "")
            jurisdiction_laws = {
                "MY": ["EMPLOYMENT_ACT_MY", "PDPA_MY"],
                "SG": ["PDPA_SG"],
                "EU": ["GDPR_EU"],
                "US": ["CCPA_US"]
            }
            applicable_laws = jurisdiction_laws.get(jurisdiction, [])
            
            if final_law not in applicable_laws:
                logger.error(f"BLOCKED in AI cleaning: Law '{final_law}' not applicable for jurisdiction '{jurisdiction}' - removing issue")
                continue
            # Clean up generic placeholder requirements
            requirements = issue.get("missing_requirements", [])
            cleaned_requirements = []
            
            for req in requirements:
                if req and not self._is_generic_placeholder(req):
                    cleaned_requirements.append(req)
            
            # If all requirements were generic, generate specific ones
            if not cleaned_requirements:
                cleaned_requirements = self._generate_specific_requirements(issue["law"], jurisdiction)
            
            issue["missing_requirements"] = cleaned_requirements
            
            # Clean up generic placeholder recommendations
            recommendations = issue.get("recommendations", [])
            cleaned_recommendations = []
            
            for rec in recommendations:
                if rec and not self._is_generic_placeholder(rec):
                    cleaned_recommendations.append(rec)
            
            # If all recommendations were generic, generate specific ones
            if not cleaned_recommendations:
                cleaned_recommendations = self._generate_specific_recommendations(issue["law"], jurisdiction)
            
            issue["recommendations"] = cleaned_recommendations
            
            cleaned_compliance.append(issue)
        
        ai_json["flagged_clauses"] = cleaned_flagged
        ai_json["compliance_issues"] = cleaned_compliance
        
        logger.info(f"Response cleaning complete: {len(cleaned_flagged)} flagged clauses, {len(cleaned_compliance)} compliance issues retained")
        return ai_json
    
    def _select_appropriate_law(self, law_options: List[str], jurisdiction: str, ai_json: Dict[str, Any]) -> str:
        """
        Select the most appropriate law from a list of options based on jurisdiction and contract context.
        """
        # Priority mapping based on jurisdiction - avoid defaulting to employment law for non-employment contracts
        jurisdiction_priority = {
            "MY": ["PDPA_MY", "EMPLOYMENT_ACT_MY"],  # PDPA first, employment law only if specifically needed
            "SG": ["PDPA_SG"],
            "EU": ["GDPR_EU"],
            "US": ["CCPA_US"]
        }
        
        # Get priority laws for jurisdiction
        priority_laws = jurisdiction_priority.get(jurisdiction, [])
        
        # Find the first matching priority law
        for priority_law in priority_laws:
            if priority_law in law_options:
                return priority_law
        
        # Fallback: return first valid law option
        valid_laws = ["EMPLOYMENT_ACT_MY", "PDPA_MY", "PDPA_SG", "GDPR_EU", "CCPA_US"]
        for law in law_options:
            if law.strip() in valid_laws:
                return law.strip()
        
        # Last resort: default for jurisdiction
        return self._get_default_law_for_jurisdiction(jurisdiction)
    
    def _get_default_law_for_jurisdiction(self, jurisdiction: str) -> str:
        """
        Get the default law for a given jurisdiction.
        NEVER defaults to employment law for non-employment contracts.
        """
        # Data protection laws are generally applicable to most contracts
        # Employment laws should only be used for employment contracts specifically
        defaults = {
            "MY": "PDPA_MY",     # Changed from EMPLOYMENT_ACT_MY to prevent cross-application
            "SG": "PDPA_SG", 
            "EU": "GDPR_EU",
            "US": "CCPA_US"
        }
        
        default_law = defaults.get(jurisdiction)
        
        if not default_law:
            logger.error(f"No default law found for jurisdiction '{jurisdiction}', this indicates a configuration issue")
            # Return None to indicate no applicable law rather than defaulting to wrong jurisdiction
            return None
        
        return default_law
    
    def _is_generic_placeholder(self, text: str) -> bool:
        """
        Check if text is a generic placeholder that should be replaced.
        """
        generic_phrases = [
            "specific statutory requirements missing",
            "specific actionable legal compliance steps",
            "general compliance concern identified",
            "review with legal counsel",
            "statutory requirements missing",
            "actionable legal compliance steps"
        ]
        
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in generic_phrases)
    
    def _generate_specific_requirements(self, law: str, jurisdiction: str) -> List[str]:
        """
        Generate specific missing requirements based on the law.
        Enhanced for comprehensive employment law compliance.
        """
        requirements_map = {
            "EMPLOYMENT_ACT_MY": [
                "Termination notice provisions do not meet Employment Act 1955 Section 12 minimum requirements (2 weeks for <2 years service, 4 weeks for >2 years service)",
                "Missing overtime compensation violates Employment Act 1955 Section 60A (minimum 1.5x normal hourly rate required)",
                "Working hours may exceed Employment Act 1955 Section 60A maximum (8 hours/day, 48 hours/week)",
                "Annual leave entitlement below Employment Act 1955 Section 60E minimum (8-16 days based on service length)",
                "Probation period may exceed Employment Act 1955 Section 11 maximum of 6 months",
                "Missing rest day and public holiday provisions required under Employment Act 1955 Sections 60C, 60D",
                "Missing EPF contribution provisions required under EPF Act 1991 (11% employee, 12-13% employer)",
                "Missing SOCSO contribution provisions required under SOCSO Act 1969",
                "Salary may be below minimum wage requirement of RM1,500 under Minimum Wages Order 2022"
            ],
            "PDPA_MY": [
                "Missing explicit consent mechanisms required under Personal Data Protection Act 2010",
                "Lacks data subject rights provisions (access, correction, withdrawal) as mandated by PDPA 2010",
                "Missing purpose limitation clauses required under PDPA 2010 Section 6",
                "Insufficient data security safeguards as required under PDPA 2010 Section 7"
            ],
            "PDPA_SG": [
                "Missing consent notification requirements under Singapore PDPA 2012",
                "Lacks data protection officer designation as required by PDPA",
                "Missing purpose specification and limitation under PDPA 2012"
            ],
            "GDPR_EU": [
                "Missing lawful basis for processing personal data under GDPR Article 6",
                "Lacks data subject rights implementation as required by GDPR Articles 15-22",
                "Missing data protection impact assessment requirements under GDPR Article 35",
                "Insufficient cross-border data transfer safeguards under GDPR Chapter V"
            ],
            "CCPA_US": [
                "Missing Right to Correct personal information under CCPA § 1798.106",
                "Missing Right to Limit Use of Sensitive Personal Information under CCPA § 1798.121", 
                "Missing Right of Non-Discrimination under CCPA § 1798.125",
                "Discriminatory practices for opt-out requests violate CCPA § 1798.125(a)",
                "Inadequate contact methods - CCPA § 1798.130(a)(1) requires at least 2 methods including toll-free number",
                "Response time violations - CCPA § 1798.130(a)(2) requires initial response within 45 days",
                "Prohibited fee structure - CCPA § 1798.130(a)(2) prohibits charging consumers for exercising rights",
                "Service provider violations - CCPA § 1798.140(ag) restricts service provider data use",
                "Incomplete Notice at Collection missing CCPA-specific PI categories under § 1798.100(b)",
                "Missing data sale disclosure and opt-out requirements under CCPA § 1798.115",
                "Missing consumer privacy rights disclosure under CCPA Section 1798.100",
                "Lacks opt-out mechanisms as required by California Consumer Privacy Act",
                "Missing data sale disclosure requirements under CCPA Section 1798.115"
            ]
        }
        
        return requirements_map.get(law, ["Contract requires legal review for compliance"])
    
    def _generate_specific_recommendations(self, law: str, jurisdiction: str) -> List[str]:
        """
        Generate specific recommendations based on the law.
        Enhanced for comprehensive employment law compliance.
        """
        recommendations_map = {
            "EMPLOYMENT_ACT_MY": [
                "Add termination clause specifying minimum notice periods: 2 weeks for employees with <2 years service, 4 weeks for >2 years service as per Section 12",
                "Include overtime payment clause requiring minimum 1.5x normal hourly rate as mandated by Section 60A",
                "Specify working hours limits: maximum 8 hours per day and 48 hours per week as per Section 60A",
                "Include annual leave entitlement: 8 days (<2 years), 12 days (2-5 years), 16 days (>5 years) as per Section 60E",
                "Limit probation period to maximum 6 months as required by Section 11",
                "Add rest day provisions (1 day per week) and gazetted public holiday entitlements as per Sections 60C, 60D",
                "Include EPF contribution clause: 11% employee contribution, 12-13% employer contribution as per EPF Act 1991",
                "Add SOCSO contribution clause for employment injury and invalidity coverage as per SOCSO Act 1969",
                "Ensure monthly salary meets minimum wage of RM1,500 as per Minimum Wages Order 2022"
            ],
            "PDPA_MY": [
                "Implement clear consent procedures with opt-in mechanisms before collecting personal data",
                "Add comprehensive data subject rights clauses covering access, correction, and withdrawal of consent",
                "Include purpose limitation clause specifying exact purposes for data collection and processing",
                "Implement robust data security measures including encryption and access controls"
            ],
            "PDPA_SG": [
                "Include notification requirements before collecting personal data with clear purpose statements",
                "Designate data protection officer and include contact details",
                "Implement consent withdrawal mechanisms and data portability procedures"
            ],
            "GDPR_EU": [
                "Establish clear lawful basis for each type of data processing under Article 6",
                "Implement comprehensive data subject rights response procedures for Articles 15-22",
                "Conduct data protection impact assessments for high-risk processing",
                "Implement appropriate safeguards for international data transfers"
            ],
            "CCPA_US": [
                "Implement all required CCPA consumer rights: Right to Know, Delete, Correct, Limit Use of Sensitive PI, and Non-Discrimination",
                "Provide at least 2 contact methods including toll-free number as required by CCPA § 1798.130(a)(1)",
                "Ensure initial response within 45 days and total fulfillment within 90 days per CCPA § 1798.130(a)(2)",
                "Remove all fees for consumer rights requests - CCPA prohibits charging consumers",
                "Restrict service providers to specific business purposes only - prohibit use for their own purposes",
                "Include comprehensive Notice at Collection with all CCPA-required PI categories and disclosures",
                "Implement proper opt-out mechanisms for data selling under CCPA § 1798.115",
                "Remove discriminatory practices - cannot charge fees or limit services for exercising rights",
                "Add consumer privacy notice with clear disclosure of data practices under Section 1798.100",
                "Implement opt-out mechanisms for data selling and sharing under Section 1798.120",
                "Establish procedures for consumer rights requests under CCPA"
            ]
        }
        
        return recommendations_map.get(law, ["Consult legal counsel for jurisdiction-specific compliance"])
    
    def _is_substantive_clause(self, clause_text: str) -> bool:
        """
        Determine if a clause is substantive contract content vs formatting artifact.
        """
        if not clause_text or len(clause_text.strip()) < 20:
            return False
        
        clause_lower = clause_text.lower()
        
        # Reject formatting artifacts
        formatting_indicators = [
            "###", "##", "#", "**", "*", "```", "---", "===",
            "summary", "analysis", "review", "note", "generated",
            "created by", "document", "title", "header", "footer"
        ]
        
        if any(indicator in clause_lower for indicator in formatting_indicators):
            return False
        
        # Require substantive legal language
        legal_indicators = [
            "shall", "will", "agree", "party", "parties", "contract",
            "obligation", "right", "term", "condition", "provision",
            "whereas", "therefore", "hereby", "subject to"
        ]
        
        return any(indicator in clause_lower for indicator in legal_indicators)
    
    def _perform_comprehensive_contract_analysis(self, contract_text: str, metadata: Dict[str, Any], 
                                               jurisdiction: str) -> Dict[str, Any]:
        """
        Perform comprehensive contract analysis using IBM Granite-inspired legal reasoning.
        Enhanced for rigorous employment contract analysis with specific statutory violations.
        """
        flagged_clauses = []
        text_lower = contract_text.lower()
        
        # ENHANCED EMPLOYMENT CONTRACT ANALYSIS (MY jurisdiction specific)
        if metadata['type'] == 'Employment' and jurisdiction == 'MY':
            
            # 1. Comprehensive Termination Analysis (Employment Act 1955, Section 12)
            self._analyze_termination_provisions(contract_text, text_lower, flagged_clauses)
            
            # 2. Working Hours and Overtime Analysis (Employment Act 1955, Section 60A)
            self._analyze_working_hours_and_overtime(contract_text, text_lower, flagged_clauses)
            
            # 3. Annual Leave Analysis (Employment Act 1955, Section 60E)
            self._analyze_annual_leave_provisions(contract_text, text_lower, flagged_clauses)
            
            # 4. Salary and Benefits Analysis (Employment Act 1955, Various Sections)
            self._analyze_salary_and_benefits(contract_text, text_lower, flagged_clauses)
            
            # 5. Probation Period Analysis (Employment Act 1955, Section 11)
            self._analyze_probation_period(contract_text, text_lower, flagged_clauses)
            
            # 6. Rest Day and Public Holiday Analysis (Employment Act 1955, Sections 60C, 60D)
            self._analyze_rest_days_and_holidays(contract_text, text_lower, flagged_clauses)
            
            # 7. EPF and SOCSO Compliance Analysis
            self._analyze_statutory_contributions(contract_text, text_lower, flagged_clauses)
        
        # Data protection analysis (only if contract actually processes personal data)
        if metadata['has_data_processing']:
            law_name = f"PDPA_{jurisdiction}" if jurisdiction in ['MY', 'SG'] else f"GDPR_{jurisdiction}" if jurisdiction == 'EU' else f"CCPA_{jurisdiction}"
            
            # Enhanced CCPA-specific flagged clause detection
            if jurisdiction == 'US':
                self._analyze_ccpa_clause_violations(contract_text, text_lower, flagged_clauses)
            
            # 1. Consent mechanisms
            if not re.search(r'consent.*(?:explicit|written|informed)', text_lower):
                data_clause = re.search(r'(?:personal.*data|information.*collect).*?(?:\.|$)', contract_text, re.IGNORECASE | re.DOTALL)
                if data_clause:
                    context = data_clause.group(0)[:200] + ("..." if len(data_clause.group(0)) > 200 else "")
                    flagged_clauses.append({
                        "clause_text": context,
                        "issue": f"Missing explicit consent mechanisms required under {law_name} for personal data processing",
                        "severity": "high"
                    })
            
            # 2. Data subject rights
            required_rights = ['access', 'rectification', 'erasure', 'portability'] if jurisdiction == 'EU' else ['access', 'correction', 'withdrawal']
            missing_rights = [right for right in required_rights if right not in text_lower]
            
            if len(missing_rights) > 2:
                flagged_clauses.append({
                    "clause_text": "Data processing provisions",
                    "issue": f"Missing data subject rights ({', '.join(missing_rights)}) required under {law_name}",
                    "severity": "medium"
                })
        
        # General contract law issues
        self._analyze_general_contract_issues(contract_text, text_lower, flagged_clauses)
        
        return {"flagged_clauses": flagged_clauses}
    
    def _analyze_termination_provisions(self, contract_text: str, text_lower: str, flagged_clauses: list):
        """Detailed analysis of termination provisions under Employment Act 1955 Section 12"""
        
        # Check for immediate termination without notice (except for misconduct)
        termination_patterns = [
            r'terminate.*without.*notice',
            r'dismiss.*immediately',
            r'termination.*effective.*immediately',
            r'end.*employment.*without.*notice'
        ]
        
        for pattern in termination_patterns:
            matches = re.finditer(pattern, contract_text, re.IGNORECASE)
            for match in matches:
                context = self._extract_clause_context(contract_text, match.start(), match.end())
                if 'misconduct' not in context.lower() and 'gross negligence' not in context.lower():
                    flagged_clauses.append({
                        "clause_text": context,
                        "issue": "Immediate termination without notice violates Employment Act 1955 Section 12 minimum notice requirements (4 weeks for employees with >2 years service, 2 weeks for <2 years)",
                        "severity": "high"
                    })
                    break  # Only flag once per contract
        
        # Check for insufficient notice periods
        notice_patterns = [
            r'(?:notice|termination).*(?:\d+.*(?:week|month|day))',
            r'(\d+).*?(day|week|month).*?(?:notice|termination)'
        ]
        
        for pattern in notice_patterns:
            matches = re.finditer(pattern, contract_text, re.IGNORECASE)
            for match in matches:
                notice_num = int(match.group(1))
                notice_period = match.group(2).lower()
                
                # Convert to days for comparison
                notice_days = notice_num
                if notice_period == 'week':
                    notice_days = notice_num * 7
                elif notice_period == 'month':
                    notice_days = notice_num * 30
                
                # Employment Act requires minimum 4 weeks (28 days) for >2 years service
                if notice_days < 14:  # Less than 2 weeks is clearly insufficient
                    context = self._extract_clause_context(contract_text, match.start(), match.end())
                    flagged_clauses.append({
                        "clause_text": context,
                        "issue": f"Notice period of {notice_num} {notice_period}{'s' if notice_num > 1 else ''} may be insufficient under Employment Act 1955 Section 12 (minimum 2-4 weeks required)",
                        "severity": "medium"
                    })
                    break
    
    def _analyze_working_hours_and_overtime(self, contract_text: str, text_lower: str, flagged_clauses: list):
        """Detailed analysis of working hours and overtime under Employment Act 1955 Section 60A"""
        
        # Check for excessive working hours
        hours_patterns = [
            r'(\d+).*hours?.*(?:per|each).*(?:day|daily)',
            r'(\d+).*hours?.*(?:per|each).*(?:week|weekly)',
            r'working.*hours?.*(\d+).*(?:per|each).*(?:day|week)'
        ]
        
        for pattern in hours_patterns:
            matches = re.finditer(pattern, contract_text, re.IGNORECASE)
            for match in matches:
                hours = int(match.group(1))
                period_text = match.group(0).lower()
                
                context = self._extract_clause_context(contract_text, match.start(), match.end())
                
                if ('day' in period_text or 'daily' in period_text) and hours > 8:
                    flagged_clauses.append({
                        "clause_text": context,
                        "issue": f"Working hours of {hours} per day exceeds Employment Act 1955 Section 60A maximum of 8 hours per day",
                        "severity": "high"
                    })
                elif ('week' in period_text or 'weekly' in period_text) and hours > 48:
                    flagged_clauses.append({
                        "clause_text": context,
                        "issue": f"Working hours of {hours} per week exceeds Employment Act 1955 Section 60A maximum of 48 hours per week",
                        "severity": "high"
                    })
        
        # Check for missing overtime compensation
        if not re.search(r'overtime.*(?:compensation|payment|rate|1\.5|time.*half)', text_lower):
            # Look for salary/wage sections to attach this issue to
            wage_section = re.search(r'(?:salary|wage|compensation|remuneration).*?(?:\.|;|$)', contract_text, re.IGNORECASE | re.DOTALL)
            if wage_section:
                context = wage_section.group(0)[:200] + ("..." if len(wage_section.group(0)) > 200 else "")
                flagged_clauses.append({
                    "clause_text": context,
                    "issue": "Missing overtime compensation provisions violates Employment Act 1955 Section 60A (minimum 1.5x normal hourly rate required)",
                    "severity": "high"
                })
    
    def _analyze_annual_leave_provisions(self, contract_text: str, text_lower: str, flagged_clauses: list):
        """Detailed analysis of annual leave under Employment Act 1955 Section 60E"""
        
        if not re.search(r'annual.*leave|vacation.*day|paid.*leave', text_lower):
            flagged_clauses.append({
                "clause_text": "Employment terms and benefits",
                "issue": "Missing annual leave entitlement violates Employment Act 1955 Section 60E (minimum 8 days for <2 years service, 12 days for 2-5 years, 16 days for >5 years)",
                "severity": "medium"
            })
        else:
            # Check if annual leave is insufficient
            leave_patterns = [
                r'annual.*leave.*(\d+).*day',
                r'vacation.*(\d+).*day',
                r'(\d+).*day.*annual.*leave'
            ]
            
            for pattern in leave_patterns:
                matches = re.finditer(pattern, contract_text, re.IGNORECASE)
                for match in matches:
                    leave_days = int(match.group(1))
                    if leave_days < 8:
                        context = self._extract_clause_context(contract_text, match.start(), match.end())
                        flagged_clauses.append({
                            "clause_text": context,
                            "issue": f"Annual leave of {leave_days} days is below Employment Act 1955 Section 60E minimum of 8 days",
                            "severity": "medium"
                        })
                        break
    
    def _analyze_salary_and_benefits(self, contract_text: str, text_lower: str, flagged_clauses: list):
        """Analysis of salary and benefits compliance"""
        
        # Check for below minimum wage (RM1,500 as of 2022)
        salary_patterns = [
            r'salary.*rm\s*(\d+(?:,\d+)*)',
            r'wage.*rm\s*(\d+(?:,\d+)*)',
            r'rm\s*(\d+(?:,\d+)*).*(?:salary|wage|month)'
        ]
        
        for pattern in salary_patterns:
            matches = re.finditer(pattern, contract_text, re.IGNORECASE)
            for match in matches:
                salary_str = match.group(1).replace(',', '')
                salary_amount = int(salary_str)
                
                if salary_amount < 1500:
                    context = self._extract_clause_context(contract_text, match.start(), match.end())
                    flagged_clauses.append({
                        "clause_text": context,
                        "issue": f"Monthly salary of RM{salary_amount} is below the minimum wage of RM1,500 under the Minimum Wages Order 2022",
                        "severity": "high"
                    })
                    break
    
    def _analyze_probation_period(self, contract_text: str, text_lower: str, flagged_clauses: list):
        """Analysis of probation period under Employment Act 1955 Section 11"""
        
        probation_patterns = [
            r'probation.*period.*(\d+).*month',
            r'probationary.*(\d+).*month',
            r'(\d+).*month.*probation'
        ]
        
        for pattern in probation_patterns:
            matches = re.finditer(pattern, contract_text, re.IGNORECASE)
            for match in matches:
                probation_months = int(match.group(1))
                if probation_months > 6:
                    context = self._extract_clause_context(contract_text, match.start(), match.end())
                    flagged_clauses.append({
                        "clause_text": context,
                        "issue": f"Probation period of {probation_months} months exceeds Employment Act 1955 Section 11 maximum of 6 months",
                        "severity": "medium"
                    })
                    break
    
    def _analyze_rest_days_and_holidays(self, contract_text: str, text_lower: str, flagged_clauses: list):
        """Analysis of rest days and public holidays under Employment Act 1955 Sections 60C, 60D"""
        
        if not re.search(r'rest.*day|public.*holiday|gazetted.*holiday', text_lower):
            flagged_clauses.append({
                "clause_text": "Employment terms and working conditions",
                "issue": "Missing rest day and public holiday provisions required under Employment Act 1955 Sections 60C and 60D",
                "severity": "medium"
            })
    
    def _analyze_statutory_contributions(self, contract_text: str, text_lower: str, flagged_clauses: list):
        """Analysis of EPF and SOCSO contributions"""
        
        if not re.search(r'epf|employees.*provident.*fund', text_lower):
            flagged_clauses.append({
                "clause_text": "Employee benefits and contributions",
                "issue": "Missing EPF (Employees Provident Fund) contribution provisions as required under EPF Act 1991",
                "severity": "medium"
            })
        
        if not re.search(r'socso|social.*security|employment.*injury', text_lower):
            flagged_clauses.append({
                "clause_text": "Employee benefits and contributions",
                "issue": "Missing SOCSO (Social Security Organisation) contribution provisions as required under SOCSO Act 1969",
                "severity": "medium"
            })
    
    def _analyze_general_contract_issues(self, contract_text: str, text_lower: str, flagged_clauses: list):
        """Analysis of general contract law issues"""
        
        # 1. Unconscionable liability limitations
        liability_match = re.search(r'liability.*limited.*to.*(?:rm\s*)?(\d+(?:,\d+)*)', text_lower)
        if liability_match:
            amount_str = liability_match.group(1).replace(',', '')
            amount = int(amount_str)
            if amount < 1000:  # Unusually low liability cap
                context = self._extract_clause_context(contract_text, liability_match.start(), liability_match.end())
                flagged_clauses.append({
                    "clause_text": context,
                    "issue": f"Liability limitation of RM{amount} may be unconscionably low and unenforceable under contract law",
                    "severity": "medium"
                })
        
        # 2. Unilateral modification rights
        if re.search(r'(?:company|employer|party).*may.*(?:modify|change|alter).*(?:unilaterally|without.*consent)', text_lower):
            modification_clause = re.search(r'(?:company|employer|party).*may.*(?:modify|change|alter).*?(?:\.|$)', contract_text, re.IGNORECASE | re.DOTALL)
            if modification_clause:
                context = modification_clause.group(0)[:200] + ("..." if len(modification_clause.group(0)) > 200 else "")
                flagged_clauses.append({
                    "clause_text": context,
                    "issue": "Unilateral modification rights without consideration may be unenforceable under contract law",
                    "severity": "medium"
                })
        
        # 3. Missing essential contract elements
        if not re.search(r'consideration|payment|compensation|remuneration', text_lower):
            flagged_clauses.append({
                "clause_text": "Contract terms and conditions",
                "issue": "Missing consideration or payment terms may affect contract enforceability under contract law",
                "severity": "medium"
            })
    
    def _extract_clause_context(self, contract_text: str, start_pos: int, end_pos: int, context_chars: int = 150) -> str:
        """
        Extract contextual clause text around a specific match position.
        """
        # Find sentence boundaries around the match
        start_context = max(0, start_pos - context_chars)
        end_context = min(len(contract_text), end_pos + context_chars)
        
        # Try to find sentence boundaries
        context = contract_text[start_context:end_context]
        
        # Clean up the context
        context = re.sub(r'\s+', ' ', context).strip()
        
        if len(context) > 200:
            context = context[:200] + "..."
        
        return context
    
    def _detect_ccpa_violations(self, contract_text: str, text_lower: str) -> tuple[List[str], List[str]]:
        """
        Enhanced CCPA violation detection to catch specific compliance issues.
        Returns tuple of (requirements, recommendations).
        """
        requirements = []
        recommendations = []
        
        logger.info("Starting comprehensive CCPA violation detection")
        
        # 1. CRITICAL: Missing Required Consumer Rights (§ 1798.100 et seq.)
        missing_rights = []
        
        # Right to Correct (§ 1798.106)
        if not re.search(r'right\s+to\s+correct|correct.*personal.*information|rectif', text_lower):
            missing_rights.append("Right to Correct personal information")
        
        # Right to Limit Use of Sensitive Personal Information (§ 1798.121)
        if not re.search(r'limit.*use.*sensitive|sensitive.*personal.*information.*limit|opt.*out.*sensitive', text_lower):
            missing_rights.append("Right to Limit Use of Sensitive Personal Information")
        
        # Right of Non-Discrimination (§ 1798.125)
        if not re.search(r'non.*discrimination|right.*not.*discriminate|equal.*treatment', text_lower):
            missing_rights.append("Right of Non-Discrimination")
        
        if missing_rights:
            requirements.append(f"Missing critical CCPA consumer rights: {', '.join(missing_rights)} under CCPA §§ 1798.100-1798.125")
            recommendations.append(f"Implement all required consumer rights: {', '.join(missing_rights)} as mandated by CCPA")
        
        # 2. CRITICAL: Explicit Discrimination Violation (§ 1798.125)
        discrimination_violations = []
        
        # Check for prohibited fee structures when opting out
        if re.search(r'opt.*out.*(?:may|will|result).*(?:additional|extra).*fee|fee.*opt.*out|charge.*opt.*out', text_lower):
            discrimination_violations.append("Charging fees for opt-out requests (§ 1798.125(a)(1))")
        
        # Check for service limitations tied to opt-out
        if re.search(r'opt.*out.*(?:may|will).*limit.*service|service.*limited.*opt.*out', text_lower):
            discrimination_violations.append("Limiting services for opt-out requests (§ 1798.125(a)(2))")
        
        if discrimination_violations:
            requirements.append(f"CRITICAL DISCRIMINATION VIOLATIONS: {', '.join(discrimination_violations)}")
            recommendations.append("Remove all discriminatory practices - CCPA § 1798.125 strictly prohibits charging fees or limiting services for exercising consumer rights")
        
        # 3. HIGH PRIORITY: Inadequate Contact Methods (§ 1798.130(a)(1))
        contact_methods = []
        
        # Check for toll-free number
        if re.search(r'toll.*free|1.*800.*|1.*888.*|1.*877.*|1.*866.*', text_lower):
            contact_methods.append("toll-free")
        
        # Check for website
        if re.search(r'website|online.*form|web.*form|www\.|http', text_lower):
            contact_methods.append("website")
        
        # Check for email
        if re.search(r'email|@.*\.com|contact.*email', text_lower):
            contact_methods.append("email")
        
        # Check for postal address
        if re.search(r'mail.*address|postal.*address|mailing.*address|street.*address', text_lower):
            contact_methods.append("postal")
        
        if len(contact_methods) < 2:
            # Extract a reasonable portion of the contact section
            contact_excerpt = text_lower[:300] + ("..." if len(text_lower) > 300 else "")
            requirements.append(f"Inadequate contact methods for consumer requests - only {len(contact_methods)} method(s) provided, CCPA § 1798.130(a)(1) requires at least 2 methods including toll-free number")
            recommendations.append("Provide at least 2 contact methods including toll-free number: phone, website, email, or postal address")
        
        # 4. MEDIUM PRIORITY: Response time violations (§ 1798.130(a)(2))
        response_time_violations = []
        
        # Check for response times over 45 days initial response
        response_matches = re.finditer(r'respond.*within.*(\d+).*days?|(\d+).*days?.*respond', text_lower)
        for match in response_matches:
            days = int(match.group(1) or match.group(2))
            if days > 45:
                response_time_violations.append(f"Initial response time of {days} days exceeds CCPA § 1798.130(a)(2) maximum of 45 days")
        
        # Check for total fulfillment time over 90 days
        fulfillment_matches = re.finditer(r'fulfill.*within.*(\d+).*days?|complete.*within.*(\d+).*days?', text_lower)
        for match in fulfillment_matches:
            days = int(match.group(1) or match.group(2))
            if days > 90:
                response_time_violations.append(f"Total fulfillment time of {days} days exceeds CCPA maximum of 90 days (45 + 45 extension)")
        
        if response_time_violations:
            requirements.append(f"Response time violations: {', '.join(response_time_violations)}")
            recommendations.append("Comply with CCPA § 1798.130(a)(2): initial response within 45 days, total fulfillment within 90 days maximum")
        
        # 5. HIGH PRIORITY: Prohibited fee structures (§ 1798.130(a)(2))
        fee_violations = []
        
        # Check for verification fees
        if re.search(r'verification.*fee|fee.*verification|charge.*verify|verification.*cost', text_lower):
            fee_violations.append("Charging verification fees")
        
        # Check for processing fees
        if re.search(r'processing.*fee|fee.*processing|charge.*process.*request', text_lower):
            fee_violations.append("Charging processing fees")
        
        # Check for any consumer request fees
        if re.search(r'fee.*consumer.*request|charge.*consumer.*right|cost.*exercise.*right', text_lower):
            fee_violations.append("Charging fees for exercising consumer rights")
        
        if fee_violations:
            requirements.append(f"PROHIBITED FEE STRUCTURE: {', '.join(fee_violations)} - CCPA § 1798.130(a)(2) prohibits charging consumers for exercising their rights")
            recommendations.append("Remove all fees associated with consumer rights requests - CCPA prohibits charging consumers")
        
        # 6. CRITICAL: Service Provider Contract Violations (§ 1798.140(ag))
        service_provider_violations = []
        
        # Check if service providers can use data for own purposes
        if re.search(r'service.*provider.*(?:can|may|use).*(?:own|their).*purpose|vendor.*use.*own.*business', text_lower):
            service_provider_violations.append("Allowing service providers to use data for their own business purposes")
        
        # Check if service providers can sell/share data
        if re.search(r'service.*provider.*(?:sell|share).*data|vendor.*sell.*data|third.*party.*sell', text_lower):
            service_provider_violations.append("Allowing service providers to sell or share personal information")
        
        # Check for lack of service provider restrictions
        if re.search(r'service.*provider|vendor|third.*party', text_lower) and not re.search(r'service.*provider.*(?:shall|must|limited|restrict)', text_lower):
            service_provider_violations.append("Missing service provider restrictions and oversight requirements")
        
        if service_provider_violations:
            requirements.append(f"SERVICE PROVIDER VIOLATIONS: {', '.join(service_provider_violations)} - violates CCPA § 1798.140(ag) service provider definition")
            recommendations.append("Restrict service providers to specific business purposes only - prohibit use for their own purposes and selling/sharing data")
        
        # 7. HIGH PRIORITY: Incomplete Notice at Collection (§ 1798.100(b))
        notice_violations = []
        
        # Check for specific CCPA PI categories
        ccpa_categories = [
            'identifiers', 'commercial information', 'biometric information', 
            'internet activity', 'geolocation data', 'sensory data', 
            'professional information', 'education information', 'inferences'
        ]
        categories_mentioned = sum(1 for cat in ccpa_categories if cat.replace(' ', '.*') in text_lower)
        
        if categories_mentioned < 3:
            notice_violations.append(f"Missing CCPA-specific personal information categories (only {categories_mentioned}/9 mentioned)")
        
        # Check for retention periods
        if not re.search(r'retention.*period|retain.*for|keep.*for.*year|delete.*after', text_lower):
            notice_violations.append("Missing retention period disclosure")
        
        # Check for source disclosure
        if not re.search(r'source.*information|collect.*from|obtain.*from|gather.*from', text_lower):
            notice_violations.append("Missing source of information disclosure")
        
        # Check for third party categories
        if not re.search(r'third.*part.*categor|share.*with.*type|disclose.*to.*categor', text_lower):
            notice_violations.append("Missing third party categories disclosure")
        
        if notice_violations:
            requirements.append(f"Incomplete Notice at Collection under CCPA § 1798.100(b): {', '.join(notice_violations)}")
            recommendations.append("Complete Notice at Collection with all CCPA-required elements: specific PI categories, retention periods, sources, and third party categories")
        
        # 8. Additional CCPA-specific checks
        additional_violations = []
        
        # Check for proper data sale disclosure (§ 1798.115)
        if re.search(r'sell.*data|sale.*personal.*information|sell.*personal', text_lower):
            if not re.search(r'opt.*out|do.*not.*sell', text_lower):
                additional_violations.append("Data sale disclosed but missing required opt-out notice under § 1798.115")
        
        # Check for sensitive PI handling (§ 1798.121)
        sensitive_terms = ['health', 'biometric', 'genetic', 'precise geolocation', 'racial', 'religious', 'sexual orientation']
        if any(term in text_lower for term in sensitive_terms):
            if not re.search(r'sensitive.*personal.*information|limit.*use.*sensitive', text_lower):
                additional_violations.append("Handling sensitive personal information without proper CCPA § 1798.121 disclosures")
        
        if additional_violations:
            requirements.extend(additional_violations)
            recommendations.append("Address all CCPA-specific disclosure and handling requirements for data sales and sensitive personal information")
        
        logger.info(f"CCPA violation detection complete: {len(requirements)} violations found")
        return requirements, recommendations
    
    def _analyze_ccpa_clause_violations(self, contract_text: str, text_lower: str, flagged_clauses: list):
        """
        Analyze specific clauses that violate CCPA requirements.
        Extract exact clause text that contains violations.
        """
        # 1. CRITICAL: Detect discrimination clauses (§ 1798.125)
        discrimination_patterns = [
            r'opt.*out.*(?:may|will|result).*(?:additional|extra).*fee[^.]*\.',
            r'opt.*out.*(?:may|will).*limit.*service[^.]*\.',
            r'(?:additional|extra).*fee.*opt.*out[^.]*\.',
            r'service.*limited.*opt.*out[^.]*\.'
        ]
        
        for pattern in discrimination_patterns:
            matches = re.finditer(pattern, contract_text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                clause_text = match.group(0).strip()
                if len(clause_text) > 20:  # Ensure substantial content
                    flagged_clauses.append({
                        "clause_text": clause_text,
                        "issue": "CRITICAL CCPA VIOLATION: Discriminatory practice for exercising opt-out rights violates CCPA § 1798.125",
                        "severity": "high"
                    })
        
        # 2. HIGH PRIORITY: Service provider violations (§ 1798.140(ag))
        service_provider_patterns = [
            r'service.*provider.*(?:can|may|use).*(?:own|their).*purpose[^.]*\.',
            r'vendor.*use.*own.*business[^.]*\.',
            r'service.*provider.*(?:sell|share).*data[^.]*\.',
            r'third.*party.*sell.*data[^.]*\.'
        ]
        
        for pattern in service_provider_patterns:
            matches = re.finditer(pattern, contract_text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                clause_text = match.group(0).strip()
                if len(clause_text) > 20:
                    flagged_clauses.append({
                        "clause_text": clause_text,
                        "issue": "SERVICE PROVIDER VIOLATION: Allowing service providers to use data for own purposes violates CCPA § 1798.140(ag)",
                        "severity": "high"
                    })
        
        # 3. HIGH PRIORITY: Prohibited fee structures (§ 1798.130(a)(2))
        fee_patterns = [
            r'verification.*fee[^.]*\.',
            r'fee.*verification[^.]*\.',
            r'processing.*fee.*request[^.]*\.',
            r'charge.*verify[^.]*\.',
            r'cost.*exercise.*right[^.]*\.'
        ]
        
        for pattern in fee_patterns:
            matches = re.finditer(pattern, contract_text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                clause_text = match.group(0).strip()
                if len(clause_text) > 20:
                    flagged_clauses.append({
                        "clause_text": clause_text,
                        "issue": "PROHIBITED FEE STRUCTURE: Charging fees for consumer rights requests violates CCPA § 1798.130(a)(2)",
                        "severity": "high"
                    })
        
        # 4. MEDIUM PRIORITY: Response time violations (§ 1798.130(a)(2))
        response_time_patterns = [
            r'respond.*within.*(\d+).*days?[^.]*\.',
            r'(\d+).*days?.*respond.*request[^.]*\.'
        ]
        
        for pattern in response_time_patterns:
            matches = re.finditer(pattern, contract_text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                clause_text = match.group(0).strip()
                days_match = re.search(r'(\d+)', clause_text)
                if days_match and int(days_match.group(1)) > 45:
                    flagged_clauses.append({
                        "clause_text": clause_text,
                        "issue": f"RESPONSE TIME VIOLATION: {days_match.group(1)} days exceeds CCPA § 1798.130(a)(2) maximum of 45 days",
                        "severity": "medium"
                    })
        
        # 5. MEDIUM PRIORITY: Contact method violations (§ 1798.130)
        # Check entire contact section for inadequate methods
        contact_section_pattern = r'(?:how.*to.*exercise|contact.*us|exercise.*rights)[\s\S]*?(?=\n\s*#{1,6}|\n\s*\*\*|$)'
        contact_match = re.search(contact_section_pattern, contract_text, re.IGNORECASE)
        
        if contact_match:
            contact_section = contact_match.group(0)
            contact_methods = 0
            
            if re.search(r'toll.*free|1.*800.*|1.*888.*|1.*877.*|1.*866.*', contact_section, re.IGNORECASE):
                contact_methods += 1
            if re.search(r'website|online.*form|web.*form|www\.|http', contact_section, re.IGNORECASE):
                contact_methods += 1
            if re.search(r'email|@.*\.com', contact_section, re.IGNORECASE):
                contact_methods += 1
            if re.search(r'mail.*address|postal.*address|mailing.*address|street.*address', contact_section, re.IGNORECASE):
                contact_methods += 1
            
            if contact_methods < 2:
                # Extract a reasonable portion of the contact section
                contact_excerpt = contact_section[:300] + ("..." if len(contact_section) > 300 else "")
                flagged_clauses.append({
                    "clause_text": contact_excerpt,
                    "issue": f"INADEQUATE CONTACT METHODS: Only {contact_methods} method(s) provided, CCPA § 1798.130(a)(1) requires at least 2 methods",
                    "severity": "medium"
                })
        
        # 6. CRITICAL: Data sale without proper opt-out (§ 1798.115)
        data_sale_pattern = r'(?:sell.*personal.*information|sale.*personal.*data)(?:(?!opt.*out|do.*not.*sell).)*[^.]*\.'
        matches = re.finditer(data_sale_pattern, contract_text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            clause_text = match.group(0).strip()
            if len(clause_text) > 20 and not re.search(r'opt.*out|do.*not.*sell', clause_text, re.IGNORECASE):
                flagged_clauses.append({
                    "clause_text": clause_text,
                    "issue": "DATA SALE VIOLATION: Personal information sale disclosed without required opt-out notice under CCPA § 1798.115",
                    "severity": "high"
                })
        
        logger.info(f"CCPA clause analysis complete: {len([c for c in flagged_clauses if 'CCPA' in c.get('issue', '')])} CCPA-specific violations flagged")
    
    def _validate_compliance_issues(self, compliance_issues: List[Dict[str, Any]], jurisdiction: str) -> List[Dict[str, Any]]:
        """
        Validate and filter compliance issues to ensure they are appropriate for the jurisdiction.
        """
        validated_issues = []
        
        jurisdiction_laws = {
            "MY": ["EMPLOYMENT_ACT_MY", "PDPA_MY"],
            "SG": ["PDPA_SG"],
            "EU": ["GDPR_EU"],
            "US": ["CCPA_US"]
        }
        
        applicable_laws = jurisdiction_laws.get(jurisdiction, [])
        
        for issue in compliance_issues:
            law = issue.get('law', '')
            
            # Ensure law is applicable to jurisdiction
            if law not in applicable_laws:
                logger.warning(f"Filtering out inapplicable law '{law}' for jurisdiction '{jurisdiction}'")
                continue
            
            # Ensure issue has required fields
            if not issue.get('missing_requirements'):
                logger.warning(f"Filtering out compliance issue with no missing requirements: {law}")
                continue
            
            if not issue.get('recommendations'):
                logger.warning(f"Filtering out compliance issue with no recommendations: {law}")
                continue
            
            # Filter out empty or placeholder requirements
            valid_requirements = [
                req for req in issue.get('missing_requirements', [])
                if req and len(req.strip()) > 10 and not self._is_generic_placeholder(req)
            ]
            
            valid_recommendations = [
                rec for rec in issue.get('recommendations', [])
                if rec and len(rec.strip()) > 10 and not self._is_generic_placeholder(rec)
            ]
            
            if valid_requirements and valid_recommendations:
                issue['missing_requirements'] = valid_requirements
                issue['recommendations'] = valid_recommendations
                validated_issues.append(issue)
            else:
                logger.warning(f"Filtering out compliance issue with insufficient detail: {law}")
        
        logger.info(f"Validated {len(validated_issues)}/{len(compliance_issues)} compliance issues")
        return validated_issues
    
    def _apply_critical_legal_analysis(self, flagged_clauses: List[Dict[str, Any]], 
                                     metadata: Dict[str, Any], jurisdiction: str) -> List[Dict[str, Any]]:
        """
        Apply critical legal analysis to prioritize only the most serious violations.
        Enhanced for jurisdiction-specific filtering.
        """
        critical_clauses = []
        
        # Priority scoring based on severity and legal significance
        for clause in flagged_clauses:
            severity = clause.get('severity', 'low')
            issue = clause.get('issue', '')
            
            # Calculate priority score
            priority_score = 0
            
            # Severity weighting
            if severity == 'high':
                priority_score += 10
            elif severity == 'medium':
                priority_score += 5
            else:
                priority_score += 1
            
            # Legal significance indicators
            critical_indicators = [
                'violation', 'violates', 'prohibited', 'critical', 'mandatory',
                'statutory', 'criminal', 'penalty', 'fine', 'breach', 'illegal',
                'non-compliance', 'contravenes', 'discrimination'
            ]
            
            issue_lower = issue.lower()
            for indicator in critical_indicators:
                if indicator in issue_lower:
                    priority_score += 3
            
            # Jurisdiction-specific critical issues
            if jurisdiction == 'US':
                ccpa_critical = ['discrimination', 'fee', 'service provider', 'response time']
                for critical_term in ccpa_critical:
                    if critical_term in issue_lower:
                        priority_score += 5
            elif jurisdiction == 'MY':
                employment_critical = ['termination', 'overtime', 'working hours', 'minimum wage']
                for critical_term in employment_critical:
                    if critical_term in issue_lower:
                        priority_score += 5
            
            # Only include clauses that meet minimum priority threshold
            if priority_score >= 8:  # High priority threshold
                clause['priority_score'] = priority_score
                critical_clauses.append(clause)
            else:
                logger.debug(f"Filtered out low-priority clause (score: {priority_score}): {issue[:50]}...")
        
        # Sort by priority score and limit to most critical issues
        critical_clauses.sort(key=lambda x: x.get('priority_score', 0), reverse=True)
        
        # Limit to top 10 most critical issues to avoid overwhelming the user
        critical_clauses = critical_clauses[:10]
        
        logger.info(f"Critical legal analysis: {len(critical_clauses)}/{len(flagged_clauses)} clauses passed priority filter")
        return critical_clauses
    
    def _enhance_ai_response(self, ai_response: str, contract_text: str, 
                           metadata: Dict[str, Any], jurisdiction: str) -> str:
        """
        Enhance a minimal AI response (Gemini or Granite) by combining it with domain expertise.
        """
        try:
            # Parse existing AI response
            ai_json = json.loads(ai_response)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse {self.ai_provider} response, creating new analysis")
            ai_json = {"summary": "", "flagged_clauses": [], "compliance_issues": []}
        
        # Get our intelligent analysis to supplement AI
        intelligent_analysis = self._get_intelligent_mock_analysis(
            contract_text, metadata, {}, jurisdiction
        )
        
        try:
            intelligent_json = json.loads(intelligent_analysis)
        except json.JSONDecodeError:
            logger.error("Could not parse intelligent analysis")
            return ai_response
        
        # Merge the analyses - prefer AI's results but supplement with ours
        merged_flagged = list(ai_json.get("flagged_clauses", []))
        merged_compliance = list(ai_json.get("compliance_issues", []))
        
        # Add our flagged clauses if AI didn't find enough
        if len(merged_flagged) < 3:
            for clause in intelligent_json.get("flagged_clauses", []):
                # Avoid duplicates
                clause_text = clause.get("clause_text", "")
                if not any(clause_text in existing.get("clause_text", "") for existing in merged_flagged):
                    merged_flagged.append(clause)
                    if len(merged_flagged) >= 5:  # Limit total flagged clauses
                        break
        
        # Add our compliance issues if AI didn't find enough
        if len(merged_compliance) < 2:
            for issue in intelligent_json.get("compliance_issues", []):
                # Avoid duplicates by law type
                law = issue.get("law", "")
                if not any(law == existing.get("law", "") for existing in merged_compliance):
                    merged_compliance.append(issue)
        
        # Create enhanced summary
        enhanced_summary = self._create_enhanced_summary(
            ai_json.get("summary", ""), intelligent_json.get("summary", ""),
            len(merged_flagged), len(merged_compliance), metadata, jurisdiction
        )
        
        enhanced_response = {
            "summary": enhanced_summary,
            "flagged_clauses": merged_flagged[:10],  # Limit to top 10
            "compliance_issues": merged_compliance[:5]  # Limit to top 5
        }
        
        logger.info(f"Enhanced {self.ai_provider} response: {len(merged_flagged)} flagged clauses, {len(merged_compliance)} compliance issues")
        return json.dumps(enhanced_response)
    
    # Keep the old method name for backward compatibility
    def _enhance_granite_response(self, granite_response: str, contract_text: str, 
                                 metadata: Dict[str, Any], jurisdiction: str) -> str:
        """Backward compatibility wrapper for _enhance_ai_response"""
        return self._enhance_ai_response(granite_response, contract_text, metadata, jurisdiction)
    
    def _create_enhanced_summary(self, granite_summary: str, intelligent_summary: str,
                                flagged_count: int, compliance_count: int,
                                metadata: Dict[str, Any], jurisdiction: str) -> str:
        """
        Create an enhanced summary combining Granite and intelligent analysis.
        """
        jurisdiction_name = {
            "MY": "Malaysia", "SG": "Singapore", "EU": "European Union", "US": "United States"
        }.get(jurisdiction, jurisdiction)
        
        # Use Granite summary if substantial, otherwise use intelligent summary
        base_summary = granite_summary if len(granite_summary) > 50 else intelligent_summary
        
        # If both are minimal, create our own
        if len(base_summary) < 50:
            total_issues = flagged_count + compliance_count
            contract_type = metadata.get('type', 'contract')
            
            if total_issues == 0:
                base_summary = f"Analysis of this {contract_type.lower()} for {jurisdiction_name} compliance found no critical issues requiring immediate attention."
            elif total_issues >= 5:
                base_summary = f"This {contract_type.lower()} contains {total_issues} compliance concerns for {jurisdiction_name} jurisdiction requiring legal review."
            else:
                base_summary = f"This {contract_type.lower()} has {total_issues} moderate compliance issues for {jurisdiction_name} jurisdiction."
        
        return base_summary
    
    def _is_substantive_legal_issue(self, issue: Dict[str, Any], contract_text: str) -> bool:
        """
        Determine if a flagged issue represents a substantive legal concern.
        """
        issue_text = issue.get('issue', '')
        clause_text = issue.get('clause_text', '')
        severity = issue.get('severity', 'low')
        
        # Always include high severity issues
        if severity == 'high':
            return True
        
        # Filter out generic or non-specific issues
        generic_indicators = [
            'review recommended', 'consider adding', 'may want to include',
            'formatting issue', 'style concern', 'minor adjustment',
            'cosmetic change', 'presentation'
        ]
        
        issue_lower = issue_text.lower()
        if any(indicator in issue_lower for indicator in generic_indicators):
            return False
        
        # Require legal significance indicators
        legal_significance = [
            'violates', 'violation', 'non-compliance', 'prohibited', 'illegal',
            'breach', 'contravenes', 'mandatory', 'required by law', 'statutory',
            'regulation', 'act', 'section', 'article', 'code', 'ordinance'
        ]
        
        if any(indicator in issue_lower for indicator in legal_significance):
            return True
        
        # Check if clause text appears to be substantive contract content
        if len(clause_text) > 30 and self._is_substantive_clause(clause_text):
            return True
        
        # Medium severity issues need additional validation
        if severity == 'medium':
            # Must reference specific legal concepts
            legal_concepts = [
                'liability', 'damages', 'termination', 'breach', 'warranty',
                'indemnification', 'jurisdiction', 'governing law', 'arbitration',
                'intellectual property', 'confidentiality', 'non-disclosure'
            ]
            
            combined_text = (issue_text + ' ' + clause_text).lower()
            return any(concept in combined_text for concept in legal_concepts)
        
        # Low severity issues are generally filtered out unless very specific
        return False