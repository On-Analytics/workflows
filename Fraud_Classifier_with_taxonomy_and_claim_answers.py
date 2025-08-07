from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from pathlib import Path
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from typing import Literal
from datetime import datetime
import json, csv, glob
# =============================
# Fraud Taxonomy Schema
# =============================
class FraudTaxonomy(BaseModel):
    category: str
    subcategory: List[str]
    channels: List[str]

FRAUD_TAXONOMY: List[FraudTaxonomy] = [
    FraudTaxonomy(
        category="Digital Fraud",
        subcategory=["Phishing", "SMiShing", "Vishing", "Malware", "Remote Access Tools (RAT)", "Fake Mobile Apps"],
        channels=["Email", "SMS", "Phone Call", "Mobile Banking App", "Web Banking", "Social Media"]
    ),
    FraudTaxonomy(
        category="Payment Fraud",
        subcategory=["Authorized Push Payment (APP) Fraud", "Fake Invoices", "Interception of Payment Instructions", "Stolen Credentials"],
        channels=["Email", "Web Banking", "Mobile Banking App", "P2P Payment App", "Call Center"]
    ),
    FraudTaxonomy(
        category="Identity Fraud",
        subcategory=["Synthetic Identity", "Stolen Identity", "Forged Documents", "Deepfake ID"],
        channels=["Account Opening Portal", "Loan Application", "Mobile App", "In-Branch"]
    ),
    FraudTaxonomy(
        category="Account Takeover",
        subcategory=["Credential Stuffing", "SIM Swap", "Device Cloning", "Social Engineering"],
        channels=["Mobile Banking App", "Web Banking", "Call Center", "Email", "Social Media", "Phone Call"]
    ),
    FraudTaxonomy(
        category="Card Fraud",
        subcategory=["Skimming", "Counterfeit Card", "Lost/Stolen Card Use", "Card Not Present (CNP) Fraud"],
        channels=["POS Terminal", "ATM", "Online Store / Merchant", "Mobile Wallet"]
    ),
    FraudTaxonomy(
        category="Social Engineering",
        subcategory=["Romance Scam", "Tech Support Scam", "CEO Fraud", "Impersonation"],
        channels=["Email", "Phone Call", "Messaging Apps", "Social Media"]
    ),
    FraudTaxonomy(
        category="Internal Fraud",
        subcategory=["Unauthorized Access", "Data Theft", "Transaction Manipulation", "Bribery or Kickbacks"],
        channels=["Internal Systems", "Branch Operations", "Call Center"]
    )
]

# =============================
# Unified Schema
# =============================
class FraudAnalysisState(BaseModel):
    """Single comprehensive state model for fraud analysis"""

    # File and document info
    source_path: Optional[str] = None  
    file_path: Optional[str] = None    
    raw_text: Optional[str] = None
    
    # LLM extracted fields
    summary: Optional[str] = Field(default=None, description="Brief summary of the fraud inquiry")
    fraud_types: List[FraudTaxonomy] = Field(default_factory=list, description="List of extracted fraud types with category, subcategory, and channels")
    monetary_impact: Optional[float] = Field(default=None, description="Estimated monetary impact")
    urgency: bool = Field(default=False, description="Whether case requires urgent attention base on the timestamp sensitivity")
    extraction_reason: Optional[str] = Field(default=None, description="Explanation for classification")
    suggested_answer: Optional[str] = Field(default=None, description="Suggested answer for the customer claim")

    # Computed fields
    risk_score: Literal["Low", "Medium", "High"] = "Low"
    needs_human_review: bool = False
    processing_timestamp: str = ""

# =============================
# Setup
# =============================
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# =============================
# Node Functions
# =============================
def document_ingestion(state: FraudAnalysisState) -> FraudAnalysisState:
    """Load document content from file"""
    path = Path(state.file_path)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            state.raw_text = content
    except Exception as e:
        print(f"  → ERROR loading document: {e}")
        state.raw_text = ""
    return state

def generate_taxonomy_representation() -> str:
    """Generate a clear, hierarchical representation of the fraud taxonomy"""
    taxonomy_str = ""
    
    for i, tax in enumerate(FRAUD_TAXONOMY, 1):
        taxonomy_str += f"🔹 {tax.category.upper()}\n"
        
        subcategories = " | ".join(tax.subcategory)
        taxonomy_str += f"   Subcategories: {subcategories}\n"
        
        channels = " | ".join(tax.channels)
        taxonomy_str += f"   Channels: {channels}\n"
        
        if i < len(FRAUD_TAXONOMY):
            taxonomy_str += "\n"
    
    taxonomy_str += "\nCRITICAL: Each fraud_types object must only contain subcategories and channels that belong to the same category. Cross-category mixing is INVALID."
    
    return taxonomy_str

def build_fraud_analysis_prompt(state: FraudAnalysisState) -> str:
    """
    Build the full prompt for the fraud analysis LLM call, including taxonomy, rules, and examples.
    """
    # Generate taxonomy section
    taxonomy_section = generate_taxonomy_representation()

    # Validation Rules and examples
    prompt = f'''
You are an expert fraud analyst. Analyze the following fraud inquiry document and extract ALL the requested information related to categories, subcategories and channels. 

{taxonomy_section}

STRICT VALIDATION RULES:

1. CATEGORY VALIDATION:
   - ONLY use categories that exist exactly in the taxonomy above
   - Valid categories are: "Digital Fraud", "Payment Fraud", "Identity Fraud", "Account Takeover", "Card Fraud", "Social Engineering", "Internal Fraud"
   - If no category matches, use "Unknown"
   - NEVER create new categories or modify existing names

2. SUBCATEGORY VALIDATION:
   - ONLY use subcategories that belong to the selected category
   - Each subcategory has a specific parent category - respect this relationship
   - If fraud matches a category but no subcategory fits, use "Unknown"
   - NEVER mix subcategories from different categories in a single fraud_types object

3. CHANNEL VALIDATION:
   - ONLY use channels that are valid for the selected category
   - Each channel list is specific to its parent category
   - If channel isn't listed for that category, use "Unknown"
   - NEVER mix channels from different categories in a single fraud_types object

4. MULTIPLE FRAUD TYPES:
   - If incident involves multiple fraud categories, create SEPARATE fraud_types objects
   - Each fraud_types object must maintain category-subcategory-channel consistency

5. UNKNOWN/AMBIGUOUS VALUES:
   - Use "Unknown" when category/subcategory/channel cannot be determined
   - Use "Ambiguous" when multiple options seem equally likely within same category

6. ERROR PREVENTION:
   - Do NOT assume relationships - always check the taxonomy
   - Do NOT use partial matches (e.g., "Phish" instead of "Phishing")
   - Do NOT create hybrid categories or subcategories
   - Do NOT default to the most common fraud type if it doesn't match the evidence

7. URGENCY
   - True if client language indicates immediate need (words like "immediately", "ASAP", "help now", etc.) 
   - True when high-risk situations are detected early, to avoid costly complaints, regulatory issues, and customer churn. 

8. MONETARY IMPACT:
   - For monetary_impact: Extract numeric value if mentioned, otherwise null

9 EXTRACTION REASON: 
   - Keep extraction_reason concise (1-2 sentences max)

IMPORTANT: Return your response as a valid JSON object with these exact fields:

{{
  "summary": "Brief 1-2 sentence summary of the inquiry",
  "fraud_types": [
    {{
      "category": "Category from taxonomy",
      "subcategory": ["List of subcategories for this category"],
      "channels": ["List of channels for this category"]
    }}
    // ... (one object per fraud type detected)
  ],
  "monetary_impact": numeric_value_or_null,
  "urgency": true_or_false,
  "risk_score": "Low" or "Medium" or "High",
  "needs_human_review": true_or_false,
  "extraction_reason": "Brief 1-2 sentence explanation for your classifications"
  processing_timestamp: "Current timestamp in ISO format"
}}

VALIDATION CHECKLIST (Internal Check). Before finalizing your response, verify:
- Each category used exists in the taxonomy
- Each subcategory belongs to its paired category
- Each channel belongs to its paired category

EXAMPLES:

Example 1: Single Category - Digital Fraud
Document: "I received a suspicious email claiming to be from my bank asking me to click a link and enter my login details. I'm worried it might be a scam."
Response:
{{
  "summary": "Customer received suspicious email impersonating bank requesting login credentials, potential phishing attempt.",
  "fraud_types": [
    {{
      "category": "Digital Fraud",
      "subcategory": ["Phishing"],
      "channels": ["Email"]
    }}
  ],
  "monetary_impact": null,
  "urgency": false,
  "risk_score": "Low",
  "needs_human_review": false,
  "extraction_reason": "Classic phishing attempt via email, no monetary loss reported, no urgency indicators.",
  "processing_timestamp": "Current timestamp in ISO format"
}}

Example 2: Multiple Categories - Complex Fraud Chain
Document: "Help! Someone called pretending to be tech support, made me download remote access software, then they logged into my online banking and transferred $5,000 to another account. This happened this morning and I need help immediately!"
Response:
{{
  "summary": "Customer fell victim to tech support scam involving remote access software and unauthorized online banking transfer of $5,000.",
  "fraud_types": [
    {{
      "category": "Social Engineering",
      "subcategory": ["Tech Support Scam"],
      "channels": ["Phone Call"]
    }},
    {{
      "category": "Digital Fraud",
      "subcategory": ["Remote Access Tools (RAT)"],
      "channels": ["Web Banking"]
    }},
    {{
      "category": "Payment Fraud",
      "subcategory": ["Stolen Credentials"],
      "channels": ["Web Banking"]
    }}
  ],
  "monetary_impact": 5000.0,
  "urgency": true,
  "risk_score": "Medium",
  "needs_human_review": true,
  "extraction_reason": "Multi-stage fraud: social engineering led to RAT installation, enabling credential theft and unauthorized transfer. Urgency indicated by 'immediately' and 'help'."
  "processing_timestamp": "Current timestamp in ISO format"
}}

Example 3: Using "Unknown" Values
Document: "Someone used my card details to make purchases online, but I'm not sure how they got my information. The transactions were for $300 total."
Response:
{{
  "summary": "Customer reports unauthorized online card transactions totaling $300, source of compromise unknown.",
  "fraud_types": [
    {{
      "category": "Card Fraud",
      "subcategory": ["Unknown"],
      "channels": ["Online Store / Merchant"]
    }}
  ],
  "monetary_impact": 300.0,
  "urgency": false,
  "risk_score": "Low",
  "needs_human_review": false,
  "extraction_reason": "Card fraud confirmed via online transactions, but method of compromise cannot be determined from available information.",
  "processing_timestamp": "Current timestamp in ISO format"
}}


Example 4: Account Takeover - Ambiguous Method
Document: "My account was accessed without my permission and money was transferred out. I think someone might have my password."
Response:
{{
  "summary": "Unauthorized account access and money transfer, suspected credential compromise.",
  "fraud_types": [
    {{
      "category": "Account Takeover",
      "subcategory": ["Ambiguous"],
      "channels": ["Web Banking"]
    }}
  ],
  "monetary_impact": null,
  "urgency": false,
  "risk_score": "Low",
  "needs_human_review": false,
  "extraction_reason": "Account takeover evident from unauthorized access, but specific method unclear - could be credential stuffing, social engineering, or other means.",
  "processing_timestamp": "Current timestamp in ISO format"
}}

Now analyze this document:
{state.raw_text}
'''
    return prompt

def comprehensive_analysis(state: FraudAnalysisState) -> FraudAnalysisState:
    """Single LLM call to extract all information at once, using structured output and explicit taxonomy guidance."""
    prompt = build_fraud_analysis_prompt(state)
    response = llm.with_structured_output(FraudAnalysisState).invoke(prompt)
    
    # If response is valid, use it directly and set timestamp
    if isinstance(response, FraudAnalysisState):
        response.processing_timestamp = datetime.now().isoformat()
        return response
    else:
        # Fallback: mark state for human review and set error reason
        state.extraction_reason = "Failed to parse LLM response as structured output"
        state.needs_human_review = True
        state.processing_timestamp = datetime.now().isoformat()
        return state

def apply_business_rules(state: FraudAnalysisState) -> FraudAnalysisState:
    """Apply business logic for risk scoring and human review flags"""

    # Risk scoring logic
    urgency = bool(state.urgency)
    monetary_impact = state.monetary_impact if state.monetary_impact is not None else 0

    if urgency and monetary_impact > 10000:
        state.risk_score = "High"
    elif urgency or monetary_impact > 10000:
        state.risk_score = "Medium"
    else:
        state.risk_score = "Low"
    
    # Human review detection
    has_flagged = False
    for fraud in state.fraud_types:
        if (
            fraud.category in ["Unknown", "Ambiguous"] or
            any(sc in ["Unknown", "Ambiguous"] for sc in fraud.subcategory) or
            any(ch in ["Unknown", "Ambiguous"] for ch in fraud.channels)
        ):
            has_flagged = True
            break
    # Human review flagging logic
    state.needs_human_review = (
        has_flagged or 
        state.risk_score == "High" or
        not state.extraction_reason  # If extraction failed
    )
    return state

def generate_suggested_answer(state: FraudAnalysisState) -> FraudAnalysisState:
    """Generate a polite, personalized draft response for every claim."""
   
    fraud_context = ""
    if state.fraud_types:
        fraud_categories = [ft.category for ft in state.fraud_types]
        fraud_context += f"\nFraud Analysis Results:\n- Identified categories: {', '.join(fraud_categories)}"
    
    if state.risk_score:
        fraud_context += f"\n- Risk level: {state.risk_score}"
    
    if state.urgency:
        fraud_context += f"\n- Urgency flag: {'Yes' if state.urgency else 'No'}"
    
    if state.monetary_impact:
        fraud_context += f"\n- Monetary impact: ${state.monetary_impact:,.2f}"
    
    if state.needs_human_review:
        fraud_context += f"\n- Requires human review: {'Yes' if state.needs_human_review else 'No'}"

    customer_response_prompt = f'''You are a professional customer support assistant specialized in fraud-related banking claims. 
    Your task is to write a **brief**, **clear**, and **reassuring** response to the customer below.
    
    GUIDELINES:
    - Keep the answer short: 1-2 sentences maximum.
    - Use a calm, empathetic, and confident tone.
    - Personalize the response using the content of the claim and analysis results.
    - Avoid vague language—be direct and helpful.
    - Your goal is to reassure the customer their concern is being handled.
    - Write the answer in the **same language** as the original claim.
    - Use the fraud analysis context to provide more informed responses, but DO NOT reveal internal risk scores or technical details.
    - If high risk or needs human review, mention that the case will receive priority attention.
    - If urgent, acknowledge the time-sensitive nature appropriately.

    Customer Claim: {state.raw_text}
    {fraud_context}
'''
    try:
        response = llm.invoke(customer_response_prompt)

        state.suggested_answer = response.content.strip()
    except Exception as e:
        # Optionally log or handle the error here
        state.suggested_answer = "[Draft unavailable due to system error.]"
    return state

def generate_report(state: FraudAnalysisState) -> Dict[str, Any]:
    """Generate final report, ensuring source_path and file_path are present, and removing raw_text."""
    output = state.model_dump()
    # Always include source_path and file_path
    output['source_path'] = state.source_path
    output['file_path'] = state.file_path
    # Remove raw_text from the output if present
    output.pop('raw_text', None)
    return output

# =============================
# LangGraph Setup
# =============================
from langgraph.graph import StateGraph, START, END

# Define the simplified graph
fraud_graph = StateGraph(FraudAnalysisState)

# Add nodes
fraud_graph.add_node("ingest", document_ingestion)
fraud_graph.add_node("analyze", comprehensive_analysis)
fraud_graph.add_node("apply_rules", apply_business_rules)
fraud_graph.add_node("draft_suggested_answer", generate_suggested_answer)
fraud_graph.add_node("report", generate_report)

# Define flow
fraud_graph.add_edge(START, "ingest")
fraud_graph.add_edge("ingest", "analyze")
fraud_graph.add_edge("analyze", "apply_rules")
fraud_graph.add_edge("apply_rules", "draft_suggested_answer")
fraud_graph.add_edge("draft_suggested_answer", "report")
fraud_graph.add_edge("report", END)

fraud_graph = fraud_graph.compile()

# =============================
# Individual File Processing Functions
# =============================

def run_fraud_analysis(file_path: str) -> Dict[str, Any]:
    """Process a single fraud inquiry file using the graph"""
    source_path = os.path.basename(file_path)
    state = FraudAnalysisState(
        source_path=source_path,
        file_path=file_path
    )
    result = fraud_graph.invoke(state)  # One graph call per file
    # Ensure these fields are always set in the final state
    if isinstance(result, FraudAnalysisState):
        result.source_path = source_path
        result.file_path = file_path
    elif isinstance(result, dict):
        result['source_path'] = source_path
        result['file_path'] = file_path
    return result

def process_files_individually(claims_folder: str) -> List[Dict[str, Any]]:
    """Process files one by one (individual processing with batch appearance)"""
    txt_files = glob.glob(os.path.join(claims_folder, '*.txt'))
    results = []
    
    print(f"Found {len(txt_files)} files to process")
    print("=" * 50)
    
    # Batch processing loop - but each file processed individually
    for i, txt_file in enumerate(txt_files, 1):
        print(f"Processing ({i}/{len(txt_files)}): {os.path.basename(txt_file)}")
        
        try:
            # One graph execution per file
            result = run_fraud_analysis(txt_file)
            # Remove 'raw_text' if present before saving
            if isinstance(result, dict):
                result.pop('raw_text', None)
            results.append(result)
   
        except Exception as e:
            print(f"  → ❌ ERROR: {e}")
            # Add error result to maintain consistency
            error_result = {
                "source_path": os.path.basename(txt_file),
                "error": str(e),
                "needs_human_review": True,
                "risk_score": "High",
                "processing_timestamp": datetime.now().isoformat()
            }
            results.append(error_result)
        
        print()  # Empty line for readability
    
    return results

def export_results(results: List[Dict[str, Any]], output_dir: str):
    """Export results to JSON and CSV formats"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Export JSON
    json_path = os.path.join(output_dir, f'fraud_results_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Results saved as JSON: {json_path}")
    
    # Export CSV
    if results:
        csv_path = os.path.join(output_dir, f'fraud_results_{timestamp}.csv')
        # Get all possible keys from all results
        all_keys = sorted(set().union(*(r.keys() for r in results)))
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"✅ Results saved as CSV: {csv_path}")

def print_summary(results: List[Dict[str, Any]]):
    """Print processing summary"""
    total = len(results)
    review_needed = sum(1 for r in results if r.get('needs_human_review', False))
    high_risk = sum(1 for r in results if r.get('risk_score') == 'High')
    errors = sum(1 for r in results if 'error' in r)
    
    print("=" * 50)
    print(f"📊 PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Total files processed: {total}")
    print(f"Files needing human review: {review_needed}")
    print(f"High risk cases: {high_risk}")
    print(f"Processing errors: {errors}")
    print(f"Auto-processed successfully: {total - review_needed}")

# =============================
# Main Execution
# =============================

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    claims_folder = os.path.join(script_dir, 'Output_Routed/Fraud')
    output_dir = os.path.join(script_dir, 'Output')
    
    print(f"🔍 Looking for files in: {claims_folder}")
    
    # Individual file processing with batch appearance
    results = process_files_individually(claims_folder)
    
    # Print summary
    print_summary(results)
    
    # Export results
    export_results(results, output_dir)
    
    print("✅ Individual file processing complete!")