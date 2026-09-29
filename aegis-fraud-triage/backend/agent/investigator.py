"""Fraud Investigation Agent using LangChain + Ollama/Llama3

Uses a prompt-based ReAct pattern since llama3:8b doesn't support
native tool-calling via Ollama's API. The LLM outputs Thought/Action/Action Input
text which we parse and execute manually.
"""

import asyncio
import logging
import re
import json
from datetime import datetime
from typing import Callable, Awaitable, Optional

from backend.agent.prompts import SYSTEM_PROMPT, INVESTIGATION_TEMPLATE
from backend.agent.tools import (
    check_account_history,
    verify_merchant,
    check_travel_feasibility,
    get_account_risk_profile,
    run_kyc_check,
    recommend_action,
    find_similar_transactions,
    find_similar_investigations,
    detect_fraud_ring,
)

logger = logging.getLogger(__name__)

# Map of tool name -> callable
TOOL_MAP = {
    "check_account_history": check_account_history,
    "verify_merchant": verify_merchant,
    "check_travel_feasibility": check_travel_feasibility,
    "get_account_risk_profile": get_account_risk_profile,
    "run_kyc_check": run_kyc_check,
    "find_similar_transactions": find_similar_transactions,
    "find_similar_investigations": find_similar_investigations,
    "detect_fraud_ring": detect_fraud_ring,
    "recommend_action": recommend_action,
}

TOOL_DESCRIPTIONS = """TOOLS (use EXACTLY this format — one Action per turn, then STOP and wait):

Thought: your reasoning
Action: tool_name
Action Input: {"parameter_name": "parameter_value"}

IMPORTANT: The JSON keys must be the actual parameter names (account_id, merchant_id, etc.), NOT "param" or "value".

Available tools:
- check_account_history(account_id) — get recent transactions for an account
- verify_merchant(merchant_id) — check merchant legitimacy
- check_travel_feasibility(lat1, lon1, lat2, lon2, time_gap_minutes) — check if travel is possible
- get_account_risk_profile(account_id) — get risk stats for an account
- run_kyc_check(account_id) — identity fraud check for an account
- find_similar_transactions(account_id) — search vector DB for similar past transactions and fraud patterns
- find_similar_investigations(account_id) — find similar past investigations and their verdicts
- detect_fraud_ring(account_id) — analyze graph database for fraud ring connections
- recommend_action(action, summary) — FINAL step. action must be BLOCK, FLAG_FOR_REVIEW, or CLEAR

You MUST call recommend_action as your last step. Do NOT skip it."""

REACT_PROMPT = """{system_prompt}

{tool_descriptions}

{investigation_input}

Begin now. Output ONE Thought, then ONE Action. Wait for the Observation."""


class FraudInvestigatorAgent:
    """Autonomous AI agent that investigates fraud alerts using ReAct prompting"""

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model: str = "llama3:8b",
        broadcast_fn: Optional[Callable[[str, dict], Awaitable[None]]] = None,
    ):
        self.ollama_base_url = ollama_base_url
        self.model = model
        self.broadcast_fn = broadcast_fn
        self._llm = None

    def _get_llm(self):
        """Lazy-initialize the ChatOllama LLM"""
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

    async def _emit_trace(self, alert_id: str, step_type: str, content: str):
        """Broadcast an agent trace step via WebSocket"""
        if self.broadcast_fn:
            await self.broadcast_fn("agent_trace", {
                "alert_id": alert_id,
                "step_type": step_type,
                "content": content,
                "timestamp": datetime.utcnow().isoformat(),
            })

    def _parse_action(self, text: str, alert_context: dict = None):
        """Parse Action and Action Input from LLM output.

        alert_context should contain 'account_id' and 'merchant_id' from the
        original alert, used to fix placeholder values like '{}' that small
        models sometimes emit.
        """
        action_match = re.search(r'Action:\s*(\w+)', text)
        if not action_match:
            return None, None

        tool_name = action_match.group(1).strip()

        # Try JSON format - use greedy match to capture full JSON object
        input_match = re.search(r'Action Input:\s*({[^}]+})', text)
        if input_match:
            try:
                tool_input = json.loads(input_match.group(1))
                # Fix {"param": "x", "value": "y"} → {"x": "y"} format
                tool_input = self._fix_param_value_format(tool_name, tool_input)
                # Fix placeholder IDs that small models emit
                tool_input = self._fix_placeholder_ids(tool_input, alert_context)
                return tool_name, tool_input
            except json.JSONDecodeError:
                pass

        # Special handling for check_travel_feasibility - extract numbers
        if tool_name == "check_travel_feasibility":
            # Try to find coordinates in the text
            numbers = re.findall(r'[-+]?\d+\.?\d*', text[text.find('Action Input'):] if 'Action Input' in text else text)
            if len(numbers) >= 5:
                lat1, lon1 = float(numbers[0]), float(numbers[1])
                lat2, lon2 = float(numbers[2]), float(numbers[3])
                # Fix lat/lon confusion: latitude must be [-90, 90]
                if abs(lat1) > 90:
                    lat1, lon1 = lon1, lat1
                if abs(lat2) > 90:
                    lat2, lon2 = lon2, lat2
                return tool_name, {
                    "lat1": lat1, "lon1": lon1,
                    "lat2": lat2, "lon2": lon2,
                    "time_gap_minutes": float(numbers[4]),
                }

        # Special handling for recommend_action
        if tool_name == "recommend_action":
            action_val = "BLOCK"
            text_upper = text.upper()
            if "CLEAR" in text_upper:
                action_val = "CLEAR"
            elif "FLAG" in text_upper or "REVIEW" in text_upper:
                action_val = "FLAG_FOR_REVIEW"
            summary_match = re.search(r'(?:summary|reason|because|conclusion)[:\s]*([^\n]+)', text, re.IGNORECASE)
            summary = summary_match.group(1).strip().lstrip('="\'').strip() if summary_match else "Based on investigation findings."
            return tool_name, {"action": action_val, "summary": summary[:300]}

        # Try simpler format: Action Input: value
        input_match = re.search(r'Action Input:\s*"?([^"\n]+)"?', text)
        if input_match:
            raw_input = input_match.group(1).strip()
            # Fix placeholder values
            if raw_input in ("{}", "{{}}", "''", '""', "unknown") and alert_context:
                if tool_name in ("check_account_history", "get_account_risk_profile", "run_kyc_check"):
                    raw_input = alert_context.get("account_id", raw_input)
                elif tool_name == "verify_merchant":
                    raw_input = alert_context.get("merchant_id", raw_input)
            if tool_name in ("check_account_history", "get_account_risk_profile", "run_kyc_check",
                            "find_similar_transactions", "find_similar_investigations",
                            "detect_fraud_ring"):
                return tool_name, {"account_id": raw_input}
            elif tool_name == "verify_merchant":
                return tool_name, {"merchant_id": raw_input}
            else:
                return tool_name, {"input": raw_input}

        # Last resort for account/merchant tools: infer the ID from alert context
        if alert_context:
            if tool_name in ("check_account_history", "get_account_risk_profile", "run_kyc_check",
                            "find_similar_transactions", "find_similar_investigations",
                            "detect_fraud_ring"):
                return tool_name, {"account_id": alert_context.get("account_id", "unknown")}
            elif tool_name == "verify_merchant":
                return tool_name, {"merchant_id": alert_context.get("merchant_id", "unknown")}

        return None, None

    def _fix_param_value_format(self, tool_name: str, tool_input: dict) -> dict:
        """Fix LLM outputting {"param": "account_id", "value": "abc"} instead of {"account_id": "abc"}"""
        if "param" in tool_input and "value" in tool_input and len(tool_input) == 2:
            # The LLM used the wrong format — reconstruct the correct dict
            param_name = str(tool_input["param"]).strip()
            param_value = tool_input["value"]
            return {param_name: param_value}
        return tool_input

    def _fix_placeholder_ids(self, tool_input: dict, alert_context: dict = None) -> dict:
        """Replace placeholder IDs like '{}' with real values from alert context"""
        if not alert_context:
            return tool_input
        placeholders = ("{}", "{{}}", "", "unknown", "placeholder", "{account_id}", "{merchant_id}")
        for key in ("account_id",):
            if key in tool_input and str(tool_input[key]).strip() in placeholders:
                tool_input[key] = alert_context.get("account_id", tool_input[key])
        for key in ("merchant_id",):
            if key in tool_input and str(tool_input[key]).strip() in placeholders:
                tool_input[key] = alert_context.get("merchant_id", tool_input[key])
        return tool_input

    def _compute_confidence(self, alert_data: dict, evidence: list, action: str, tool_count: int) -> float:
        """Compute a confidence score (0-100) for the verdict based on evidence strength."""
        txn = alert_data.get("transaction", alert_data)
        risk_score = txn.get("risk_score", 0)
        evidence_text = " ".join(evidence).upper()

        # Base: scale risk_score into [0, 40]
        base = min(risk_score * 50, 40)

        # Agreeing / contradicting signals
        block_signals = [
            risk_score > 0.7,
            "IMPOSSIBLE" in evidence_text,
            "NOT REGISTERED" in evidence_text,
            "FRAUD REPORTS" in evidence_text,
            txn.get("amount", 0) > 5000,
        ]
        clear_signals = [
            risk_score < 0.3,
            "FEASIBLE" in evidence_text,
            "VERIFIED" in evidence_text and "NOT" not in evidence_text.split("VERIFIED")[0][-10:],
        ]

        if action == "blocked":
            base += sum(15 for s in block_signals if s)
            base -= sum(10 for s in clear_signals if s)
        elif action == "cleared":
            base += sum(15 for s in clear_signals if s)
            base -= sum(10 for s in block_signals if s)
        else:
            # flagged = uncertain by definition
            pass

        # Bonus for tool calls (more evidence = more confidence), capped at 25
        base += min(tool_count * 5, 25)

        # Flagged caps at 55 (uncertainty)
        if action == "flagged":
            base = min(base, 55)

        return max(5.0, min(98.0, round(base, 1)))

    def _auto_verdict(self, alert_data: dict, evidence: list) -> str:
        """Generate a verdict automatically when the LLM fails to call recommend_action"""
        txn = alert_data.get("transaction", alert_data)
        risk_score = txn.get("risk_score", 0)
        risk_factors = alert_data.get("risk_factors", [])
        amount = txn.get("amount", 0)
        evidence_text = " ".join(evidence).upper()

        # Determine action based on hard evidence
        block_signals = [
            risk_score > 0.7,
            amount > 5000,
            "IMPOSSIBLE" in evidence_text,
            "NOT REGISTERED" in evidence_text,
            "FRAUD REPORTS" in evidence_text and "0" not in evidence_text.split("FRAUD REPORTS")[1][:5],
        ]
        clear_signals = [
            risk_score < 0.3,
            "FEASIBLE" in evidence_text,
            "VERIFIED" in evidence_text and "NOT" not in evidence_text.split("VERIFIED")[0][-10:],
        ]

        block_count = sum(1 for s in block_signals if s)
        clear_count = sum(1 for s in clear_signals if s)

        if block_count >= 2:
            action = "BLOCK"
            summary = (
                f"BLOCK - ${amount:.2f} transaction blocked due to high risk indicators. "
                f"Evidence: risk score {risk_score:.2f}, {', '.join(risk_factors[:3])}."
            )
        elif clear_count >= 2 and block_count == 0:
            action = "CLEAR"
            summary = (
                f"CLEAR - ${amount:.2f} transaction cleared after investigation. "
                f"Evidence: risk score {risk_score:.2f}, merchant verified, travel feasible."
            )
        else:
            action = "FLAG_FOR_REVIEW"
            summary = (
                f"FLAG_FOR_REVIEW - ${amount:.2f} transaction flagged for manual review. "
                f"Evidence: risk score {risk_score:.2f}, inconclusive findings."
            )

        return f"RECOMMENDATION SUBMITTED: {action}\nSummary: {summary}"

    async def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool in a thread pool to avoid blocking the event loop"""
        tool_fn = TOOL_MAP.get(tool_name)
        if not tool_fn:
            return f"Error: Unknown tool '{tool_name}'. Available: {list(TOOL_MAP.keys())}"

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, tool_fn.invoke, tool_input)
            return str(result)
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return f"Error executing {tool_name}: {str(e)}"

    async def investigate(self, alert_data: dict) -> dict:
        """Run a full fraud investigation on an alert using ReAct loop"""
        alert_id = alert_data.get("id", "unknown")
        txn = alert_data.get("transaction", alert_data)

        # Build the investigation prompt
        risk_factors_str = "\n".join(
            f"  - {f}" for f in alert_data.get("risk_factors", ["Statistical anomaly detected"])
        )
        country = txn.get("country", "US")
        state = txn.get("state", "")
        is_intl = "YES — non-US country" if country != "US" else "NO — domestic US transaction"
        investigation_input = INVESTIGATION_TEMPLATE.format(
            alert_id=alert_id,
            account_id=txn.get("account_id", "unknown"),
            merchant_name=txn.get("merchant_name", "Unknown"),
            merchant_id=txn.get("merchant_id", "unknown"),
            amount=txn.get("amount", 0),
            currency=txn.get("currency", "USD"),
            city=txn.get("city", "Unknown"),
            state=state if state else "",
            country=country,
            is_international=is_intl,
            latitude=txn.get("latitude", 0),
            longitude=txn.get("longitude", 0),
            category=txn.get("category", "Unknown"),
            risk_score=txn.get("risk_score", 0),
            risk_factors=risk_factors_str,
        )

        full_prompt = REACT_PROMPT.format(
            system_prompt=SYSTEM_PROMPT,
            tool_descriptions=TOOL_DESCRIPTIONS,
            investigation_input=investigation_input,
        )

        # Context dict for fixing placeholder IDs in tool calls
        alert_context = {
            "account_id": txn.get("account_id", "unknown"),
            "merchant_id": txn.get("merchant_id", "unknown"),
        }

        try:
            llm = self._get_llm()

            # ReAct loop: Thought -> Action -> Observation -> repeat
            conversation = full_prompt
            max_steps = 6
            final_output = ""
            evidence_gathered = []
            tools_called: set = set()  # Track (tool_name, input_json) to prevent duplicates
            successful_tool_calls = 0
            parse_failures = 0

            await self._emit_trace(alert_id, "thinking", "[1/6] Starting fraud investigation...")
            step_number = 0

            for step in range(max_steps):
                step_number += 1
                response = await llm.ainvoke(conversation)
                llm_text = response.content if hasattr(response, 'content') else str(response)

                # Extract thought
                thought_match = re.search(r'Thought:?\s*(.*?)(?=Action:|$)', llm_text, re.DOTALL)
                if thought_match:
                    thought = thought_match.group(1).strip()
                    if thought:
                        await self._emit_trace(alert_id, "thinking", f"[{step_number}/6] {thought[:500]}" + ("..." if len(thought) > 500 else ""))

                # Try to parse action (with alert context for ID fixup)
                tool_name, tool_input = self._parse_action(llm_text, alert_context)

                if tool_name and tool_name in TOOL_MAP:
                    parse_failures = 0  # reset on success

                    # Deduplication: skip if we already called this exact tool with these inputs
                    if tool_name != "recommend_action":
                        call_key = (tool_name, json.dumps(tool_input, sort_keys=True))
                        if call_key in tools_called:
                            conversation += f"\n{llm_text}\n\nObservation: Already called {tool_name} with these inputs — use a DIFFERENT tool next."
                            continue
                        tools_called.add(call_key)

                    await self._emit_trace(
                        alert_id, "tool_call",
                        f"[{step_number}/6] {tool_name}({json.dumps(tool_input)[:300]})"
                    )

                    tool_result = await self._execute_tool(tool_name, tool_input)
                    await self._emit_trace(alert_id, "tool_result", tool_result[:600] + ("..." if len(tool_result) > 600 else ""))

                    if tool_name == "recommend_action":
                        final_output = tool_result
                        break

                    # Track evidence for fallback verdict
                    evidence_gathered.append(f"[{tool_name}] {tool_result[:200]}")
                    successful_tool_calls += 1

                    # Truncate tool result in conversation to prevent context bloat
                    truncated_result = tool_result[:400]

                    # Early nudge: after 4 tools, remind agent to wrap up
                    if successful_tool_calls >= 4:
                        conversation += (
                            f"\n{llm_text}\nObservation: {truncated_result}\n\n"
                            f"You have used {successful_tool_calls} of 6 steps. Call recommend_action NEXT to submit your verdict."
                        )
                    else:
                        conversation += f"\n{llm_text}\nObservation: {truncated_result}\n\nContinue. Output ONE Thought and ONE Action."

                else:
                    parse_failures += 1
                    # If we already have evidence and LLM mentions a verdict keyword, use auto_verdict
                    if evidence_gathered and any(kw in llm_text.upper() for kw in ["BLOCK", "CLEAR", "FLAG", "REVIEW"]):
                        final_output = self._auto_verdict(alert_data, evidence_gathered)
                        break
                    # Give the model one retry with a corrective prompt
                    if parse_failures >= 2:
                        break
                    conversation += (
                        f"\n{llm_text}\n\n"
                        "You did not output a valid Action. Use EXACTLY this format:\n"
                        "Thought: your reasoning\n"
                        "Action: tool_name\n"
                        'Action Input: {"account_id": "the_id_here"}\n\n'
                        "The JSON key must be the parameter name like account_id or merchant_id, NOT 'param' or 'value'.\n"
                        "Try again."
                    )

            # If the LLM never called recommend_action, nudge it once
            if not final_output and evidence_gathered:
                nudge = (
                    "\n\nYou have gathered enough evidence. Now you MUST call the recommend_action tool. "
                    "Output exactly:\n"
                    "Action: recommend_action\n"
                    'Action Input: {"action": "BLOCK", "summary": "your summary here"}\n\n'
                    "Choose BLOCK, FLAG_FOR_REVIEW, or CLEAR based on your findings."
                )
                conversation += nudge
                try:
                    response = await llm.ainvoke(conversation)
                    llm_text = response.content if hasattr(response, 'content') else str(response)
                    tool_name, tool_input = self._parse_action(llm_text, alert_context)
                    if tool_name == "recommend_action" and tool_input:
                        tool_result = await self._execute_tool(tool_name, tool_input)
                        await self._emit_trace(alert_id, "tool_result", tool_result[:600] + ("..." if len(tool_result) > 600 else ""))
                        final_output = tool_result
                    else:
                        # Don't dump raw LLM text — use structured auto-verdict instead
                        final_output = self._auto_verdict(alert_data, evidence_gathered)
                except Exception as e:
                    logger.warning(f"Nudge LLM call failed: {e}")

            # Final fallback: auto-generate verdict from evidence
            if not final_output:
                final_output = self._auto_verdict(alert_data, evidence_gathered)

            # Emit final verdict
            await self._emit_trace(alert_id, "verdict", final_output[:800] + ("..." if len(final_output) > 800 else ""))

            # Parse action from output
            action = "flagged"
            output_upper = final_output.upper()
            if "BLOCK" in output_upper:
                action = "blocked"
            elif "CLEAR" in output_upper:
                action = "cleared"
            elif "FLAG" in output_upper or "REVIEW" in output_upper:
                action = "flagged"

            confidence = self._compute_confidence(alert_data, evidence_gathered, action, successful_tool_calls)

            return {
                "alert_id": alert_id,
                "action": action,
                "confidence": confidence,
                "summary": final_output[:800] + ("..." if len(final_output) > 800 else ""),
                "raw_output": final_output,
            }

        except Exception as e:
            logger.error(f"Investigation failed for {alert_id}: {e}")
            await self._emit_trace(
                alert_id, "verdict",
                f"Investigation encountered an error. Flagging for manual review. Error: {str(e)[:200]}"
            )
            return {
                "alert_id": alert_id,
                "action": "flagged",
                "summary": f"Investigation error: {str(e)[:200]}",
            }
