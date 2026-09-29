"""Dispute Investigation Agent using LangChain + Ollama

Reuses the same ReAct prompt-based pattern from investigator.py but tailored
for customer dispute resolution. Analyzes the customer's claim against
transaction evidence and makes a fair determination.
"""

import logging
import re
import json
from datetime import datetime
from typing import Callable, Awaitable, Optional

from backend.agent.tools import (
    check_account_history,
    verify_merchant,
)
from backend.agent.dispute_tools import (
    get_dispute_context,
    check_customer_dispute_history,
    find_duplicate_transactions,
    resolve_dispute,
)

logger = logging.getLogger(__name__)

DISPUTE_TOOL_MAP = {
    "check_account_history": check_account_history,
    "verify_merchant": verify_merchant,
    "get_dispute_context": get_dispute_context,
    "check_customer_dispute_history": check_customer_dispute_history,
    "find_duplicate_transactions": find_duplicate_transactions,
    "resolve_dispute": resolve_dispute,
}

DISPUTE_SYSTEM_PROMPT = """You are a customer dispute analyst AI in a banking security system. Your role is to fairly evaluate customer disputes against transaction evidence.

You must balance customer protection with fraud prevention. Analyze the evidence objectively:
- Was the transaction authorized?
- Does the customer's claim match the transaction evidence?
- Is there a history of dispute abuse?
- Was this transaction already flagged as fraudulent?

Investigation process:
1. Review the dispute details and customer's claim
2. Get the original transaction context (check if it was flagged as fraud)
3. Check the account's transaction history for patterns
4. Verify the merchant's legitimacy
5. Check the customer's dispute history
6. Make your resolution using ONLY ONE of these three actions:

APPROVE (refund the customer) - Use ONLY when evidence clearly supports the customer's claim:
  - Transaction was already flagged as fraud by the system
  - Merchant is unregistered or suspicious
  - Duplicate charge confirmed by find_duplicate_transactions tool
  - Customer has clean dispute history and claim is straightforward

DENY (reject the dispute) - Use when evidence clearly contradicts the claim:
  - Transaction matches customer's normal spending patterns
  - Customer has a history of false disputes
  - Evidence shows the customer authorized the transaction

ESCALATE (send to human review) - Use when you are uncertain or evidence is mixed:
  - You need more verification or information
  - Evidence is inconclusive or contradictory
  - Identity theft claims without clear fraud indicators
  - High-value disputes with ambiguous evidence

IMPORTANT: If you say "further verification needed" or "more investigation required", you MUST choose ESCALATE, not APPROVE.
IMPORTANT: For DUPLICATE disputes, always call find_duplicate_transactions to check for matching charges before making your decision. If a duplicate is confirmed, APPROVE. If no duplicate is found, DENY or ESCALATE.
IMPORTANT: Write all summaries in plain text only. Do NOT use markdown formatting (no **, *, #, -, or bullet points). Use simple sentences."""

DISPUTE_TOOL_DESCRIPTIONS = """You have access to the following tools:

1. get_dispute_context(transaction_id: str) - Get original transaction details and any fraud alert/verdict
2. check_account_history(account_id: str) - Check recent transaction history for patterns
3. verify_merchant(merchant_id: str) - Verify merchant legitimacy and business details
4. check_customer_dispute_history(account_id: str) - Check past dispute count, approval rate, patterns
5. find_duplicate_transactions(transaction_id: str, account_id: str, amount: float, merchant_name: str) - Search for duplicate charges with same merchant and amount on the account. USE THIS for duplicate/double-charge dispute claims.
6. resolve_dispute(action: str, summary: str) - Submit final resolution: APPROVE, DENY, or ESCALATE

To use a tool, output EXACTLY this format:
Action: tool_name
Action Input: {"param1": "value1", "param2": "value2"}

After seeing the tool result, continue your analysis. When done, use the resolve_dispute tool with your final decision.

IMPORTANT: You MUST use the resolve_dispute tool as your final step.
IMPORTANT: For DUPLICATE dispute claims, you MUST call find_duplicate_transactions before resolving."""

DISPUTE_REACT_PROMPT = """{system_prompt}

{tool_descriptions}

Investigate the following customer dispute step by step. Think carefully, use tools to gather evidence, then make your resolution.

DISPUTE DETAILS:
  Dispute ID: {dispute_id}
  Transaction ID: {transaction_id}
  Account: {account_id}
  Merchant: {merchant_name}
  Amount: ${amount:.2f}
  Dispute Reason: {reason}
  Customer Statement: "{customer_statement}"
  Filed At: {timestamp}

Begin your investigation. Start with a Thought, then take an Action."""


class DisputeInvestigatorAgent:
    """AI agent that investigates customer disputes using ReAct prompting"""

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        broadcast_fn: Optional[Callable[[str, dict], Awaitable[None]]] = None,
    ):
        self.ollama_base_url = ollama_base_url
        self.model = model
        self.broadcast_fn = broadcast_fn
        self._llm = None

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        from langchain_ollama import ChatOllama
        self._llm = ChatOllama(
            base_url=self.ollama_base_url,
            model=self.model,
            temperature=0.1,
            num_predict=2048,
        )
        return self._llm

    async def _emit_trace(self, dispute_id: str, step_type: str, content: str):
        if self.broadcast_fn:
            await self.broadcast_fn("dispute_trace", {
                "dispute_id": dispute_id,
                "step_type": step_type,
                "content": content,
                "timestamp": datetime.utcnow().isoformat(),
            })

    def _parse_action(self, text: str):
        """Parse Action and Action Input from LLM output"""
        action_match = re.search(r'Action:\s*(\w+)', text)
        if not action_match:
            return None, None

        tool_name = action_match.group(1).strip()

        # Try JSON format
        input_match = re.search(r'Action Input:\s*({[^}]+})', text)
        if input_match:
            try:
                tool_input = json.loads(input_match.group(1))
                return tool_name, tool_input
            except json.JSONDecodeError:
                pass

        # Special handling for resolve_dispute
        if tool_name == "resolve_dispute":
            action_val = self._detect_action_from_text(text)
            # Extract summary text, handling formats like:
            # summary: "text", summary="text", summary: text, "summary": "text"
            summary_match = re.search(
                r'(?:"?summary"?)\s*[:=]\s*"?([^"}\n]+)"?',
                text, re.IGNORECASE
            )
            if summary_match:
                summary = summary_match.group(1).strip()
            else:
                # Fallback: try reason/because/conclusion
                fallback = re.search(
                    r'(?:reason|because|conclusion)[:\s]+([^\n]+)',
                    text, re.IGNORECASE
                )
                summary = fallback.group(1).strip() if fallback else "Based on investigation findings."
            return tool_name, {"action": action_val, "summary": summary[:300]}

        # Simpler format: Action Input: value
        input_match = re.search(r'Action Input:\s*"?([^"\n]+)"?', text)
        if input_match:
            raw_input = input_match.group(1).strip()
            if tool_name in ("check_account_history", "check_customer_dispute_history"):
                return tool_name, {"account_id": raw_input}
            elif tool_name == "verify_merchant":
                return tool_name, {"merchant_id": raw_input}
            elif tool_name == "get_dispute_context":
                return tool_name, {"transaction_id": raw_input}
            else:
                return tool_name, {"input": raw_input}

        return None, None

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        tool_fn = DISPUTE_TOOL_MAP.get(tool_name)
        if not tool_fn:
            return f"Error: Unknown tool '{tool_name}'. Available: {list(DISPUTE_TOOL_MAP.keys())}"
        try:
            result = tool_fn.invoke(tool_input)
            return str(result)
        except Exception as e:
            logger.error(f"Dispute tool {tool_name} failed: {e}")
            return f"Error executing {tool_name}: {str(e)}"

    def _detect_action_from_text(self, text: str) -> str:
        """Detect the intended dispute action from raw LLM text.

        Handles cases where the LLM says conflicting things like
        'I would not approve this... escalate for further review'.
        Escalation signals take priority over bare keyword matches.
        """
        text_upper = text.upper()
        text_lower = text.lower()

        # Escalation signals — if the LLM expresses uncertainty, always escalate
        escalation_phrases = [
            "further verification", "further review", "more investigation",
            "more information", "needs review", "need to verify",
            "uncertain", "inconclusive", "ambiguous", "mixed evidence",
            "cannot determine", "insufficient evidence", "human review",
            "manual review", "additional review",
        ]
        has_escalation_signal = any(p in text_lower for p in escalation_phrases)

        if has_escalation_signal:
            return "ESCALATE"

        # Clean keyword detection — check for explicit action declarations
        # Look for patterns like "Action: APPROVE" or "resolution: DENY"
        explicit_match = re.search(
            r'(?:action|resolution|decision|verdict)[:\s]+\b(APPROVE|DENY|ESCALATE)\b',
            text, re.IGNORECASE
        )
        if explicit_match:
            return explicit_match.group(1).upper()

        # Fallback to keyword presence (ESCALATE > DENY > APPROVE priority)
        if "ESCALATE" in text_upper:
            return "ESCALATE"
        if "DENY" in text_upper or "DENIED" in text_upper:
            return "DENY"
        if "APPROVE" in text_upper:
            return "APPROVE"

        return "ESCALATE"  # Default to escalate when unclear

    def _extract_llm_conclusion(self, llm_text: str, action: str) -> str:
        """Extract a clean conclusion from raw LLM thought text."""
        # Try to find explicit conclusion/recommendation
        for pattern in [
            r'(?:conclusion|recommend|decision|therefore|resolution|result|finding)[:\s]+(.+?)(?:\n|$)',
            r'(?:I\s+(?:recommend|suggest|conclude|decide))\s+(?:to\s+)?(.+?)(?:\n|$)',
            r'(?:Based on|After reviewing|Given the)[^,]*,\s*(.+?)(?:\n|$)',
        ]:
            m = re.search(pattern, llm_text, re.IGNORECASE)
            if m:
                return m.group(1).strip()[:300]

        # Fallback: take last substantive line that isn't a thought prefix
        lines = [l.strip() for l in llm_text.split('\n') if l.strip() and len(l.strip()) > 15]
        conclusion_lines = [
            l for l in lines
            if not re.match(r'^(Thought|Action|Action Input|Observation|I\'ll start|Let me|Next|First)[:\s]', l, re.IGNORECASE)
        ]
        if conclusion_lines:
            return conclusion_lines[-1][:300]
        if lines:
            return lines[-1][:300]

        return f"Dispute {action.lower()}d based on investigation findings."

    def _auto_resolve(self, dispute_data: dict) -> str:
        """Auto-resolve when LLM fails to call resolve_dispute"""
        reason = dispute_data.get("reason", "")
        amount = dispute_data.get("amount", 0)

        # Simple heuristic fallback
        if reason in ("unauthorized_charge", "fraud_claim"):
            action = "APPROVE"
            summary = f"Auto-approved: Customer reported {reason} for ${amount:.2f}. Refund recommended."
        elif reason == "duplicate":
            action = "APPROVE"
            summary = f"Auto-approved: Duplicate charge of ${amount:.2f} identified."
        elif amount > 5000:
            action = "ESCALATE"
            summary = f"High-value dispute (${amount:.2f}) escalated for manual review."
        else:
            action = "ESCALATE"
            summary = f"Dispute for ${amount:.2f} escalated - insufficient evidence for auto-resolution."

        return f"DISPUTE RESOLUTION: {action}\nSummary: {summary}"

    async def investigate(self, dispute_data: dict) -> dict:
        """Run a full dispute investigation using ReAct loop"""
        dispute_id = dispute_data.get("id", "unknown")

        full_prompt = DISPUTE_REACT_PROMPT.format(
            system_prompt=DISPUTE_SYSTEM_PROMPT,
            tool_descriptions=DISPUTE_TOOL_DESCRIPTIONS,
            dispute_id=dispute_id,
            transaction_id=dispute_data.get("transaction_id", "unknown"),
            account_id=dispute_data.get("account_id", "unknown"),
            merchant_name=dispute_data.get("merchant_name", "Unknown"),
            amount=dispute_data.get("amount", 0),
            reason=dispute_data.get("reason", "unknown"),
            customer_statement=dispute_data.get("customer_statement", "No statement provided"),
            timestamp=dispute_data.get("timestamp", ""),
        )

        try:
            llm = self._get_llm()
            conversation = full_prompt
            max_steps = 6
            final_output = ""

            await self._emit_trace(dispute_id, "thinking", "Starting dispute investigation...")

            for step in range(max_steps):
                response = await llm.ainvoke(conversation)
                llm_text = response.content if hasattr(response, 'content') else str(response)

                # Extract thought
                thought_match = re.search(r'Thought:?\s*(.*?)(?=Action:|$)', llm_text, re.DOTALL)
                if thought_match:
                    thought = thought_match.group(1).strip()
                    if thought:
                        await self._emit_trace(dispute_id, "thinking", thought[:300])

                # Parse action
                tool_name, tool_input = self._parse_action(llm_text)

                if tool_name and tool_name in DISPUTE_TOOL_MAP:
                    await self._emit_trace(
                        dispute_id, "tool_call",
                        f"Calling: {tool_name}({json.dumps(tool_input)[:200]})"
                    )

                    tool_result = self._execute_tool(tool_name, tool_input)
                    await self._emit_trace(dispute_id, "tool_result", tool_result[:400])

                    if tool_name == "resolve_dispute":
                        final_output = tool_result
                        break

                    truncated_result = tool_result[:300]
                    conversation += f"\n{llm_text}\nObservation: {truncated_result}\n\nContinue. Use your next Action."
                else:
                    if any(kw in llm_text.upper() for kw in ["APPROVE", "DENY", "ESCALATE"]):
                        # LLM gave a decision without calling resolve_dispute
                        action_val = self._detect_action_from_text(llm_text)
                        tool_result = self._execute_tool("resolve_dispute", {
                            "action": action_val,
                            "summary": self._extract_llm_conclusion(llm_text, action_val),
                        })
                        final_output = tool_result
                    break

            # Nudge if no resolution
            if not final_output:
                nudge = (
                    "\n\nYou have gathered enough evidence. Now you MUST call the resolve_dispute tool. "
                    "Output exactly:\n"
                    "Action: resolve_dispute\n"
                    'Action Input: {"action": "APPROVE", "summary": "your summary here"}\n\n'
                    "Choose APPROVE, DENY, or ESCALATE based on your findings."
                )
                conversation += nudge
                try:
                    response = await llm.ainvoke(conversation)
                    llm_text = response.content if hasattr(response, 'content') else str(response)
                    tool_name, tool_input = self._parse_action(llm_text)
                    if tool_name == "resolve_dispute" and tool_input:
                        tool_result = self._execute_tool(tool_name, tool_input)
                        await self._emit_trace(dispute_id, "tool_result", tool_result[:400])
                        final_output = tool_result
                    else:
                        # LLM still didn't call tool — extract decision from raw text
                        action_val = self._detect_action_from_text(llm_text)
                        tool_result = self._execute_tool("resolve_dispute", {
                            "action": action_val,
                            "summary": self._extract_llm_conclusion(llm_text, action_val),
                        })
                        final_output = tool_result
                except Exception as e:
                    logger.warning(f"Dispute nudge failed: {e}")

            # Final fallback
            if not final_output:
                final_output = self._auto_resolve(dispute_data)

            await self._emit_trace(dispute_id, "verdict", final_output[:500])

            # Parse action from structured output (e.g. "DISPUTE RESOLUTION: APPROVE")
            # Check ESCALATE first — escalation signals always take priority
            action = "escalated"
            detected = self._detect_action_from_text(final_output)
            if detected == "APPROVE":
                action = "approved"
            elif detected == "DENY":
                action = "denied"
            else:
                action = "escalated"

            # Extract clean summary text from raw output
            # Possible formats:
            # 1. Tool output: "DISPUTE RESOLUTION: APPROVE\nSummary: the actual summary"
            # 2. Raw LLM text: "Thought: I'll start by reviewing... Based on my analysis, I recommend APPROVE because..."
            # 3. Summary ="text" or Summary: text
            clean_summary = final_output

            # First try: structured "Summary:" line
            summary_match = re.search(r'Summary:?\s*=?\s*"?([^"\n]+)', final_output)
            if summary_match:
                clean_summary = summary_match.group(1).strip().strip('"\')')
            else:
                # Strip "DISPUTE RESOLUTION: ACTION" header if present
                clean_summary = re.sub(
                    r'^DISPUTE RESOLUTION:\s*\w+\s*\n?\s*', '', clean_summary, flags=re.IGNORECASE
                ).strip()

                # If it's raw LLM thought text, extract just the conclusion
                # Look for conclusion/recommendation/decision patterns
                conclusion_match = re.search(
                    r'(?:conclusion|recommend|decision|therefore|resolution|result|finding)[:\s]+(.+?)(?:\n|$)',
                    clean_summary, re.IGNORECASE
                )
                if conclusion_match:
                    clean_summary = conclusion_match.group(1).strip()
                elif '\n' in clean_summary:
                    # Multi-line LLM output — take the last substantive line
                    # (the LLM typically puts its conclusion last)
                    lines = [l.strip() for l in clean_summary.split('\n') if l.strip() and len(l.strip()) > 10]
                    if lines:
                        # Skip lines that start with thought-process prefixes
                        conclusion_lines = [
                            l for l in lines
                            if not re.match(r'^(Thought|Action|Action Input|Observation|I\'ll start|Let me|Next|First)[:\s]', l, re.IGNORECASE)
                        ]
                        if conclusion_lines:
                            clean_summary = conclusion_lines[-1]
                        else:
                            clean_summary = lines[-1]

            # Strip any remaining thought-process prefixes
            clean_summary = re.sub(
                r'^(Thought|Action|Observation)[:\s]+', '', clean_summary, flags=re.IGNORECASE
            ).strip()

            if not clean_summary or len(clean_summary) < 5:
                clean_summary = f"Dispute {action} based on investigation findings."

            return {
                "dispute_id": dispute_id,
                "action": action,
                "summary": clean_summary[:500],
            }

        except Exception as e:
            logger.error(f"Dispute investigation failed for {dispute_id}: {e}")
            await self._emit_trace(
                dispute_id, "verdict",
                f"Investigation error. Escalating for manual review. Error: {str(e)[:200]}"
            )
            return {
                "dispute_id": dispute_id,
                "action": "escalated",
                "summary": f"Investigation error: {str(e)[:200]}",
            }
