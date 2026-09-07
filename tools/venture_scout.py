"""
Dual-Engine Venture Scout, Red-Team Strategist, and Chief Operating Officer.

Runs the local Ollama reasoning model with a dedicated persona system prompt
(same pattern as tools/vision.py's use of the vision model: a self-contained
sub-call that returns finished text, not a tool the model chains further
calls through). Two entry points matching the original two commands:

  - scout_find_trends   -> MODE 1: pre-viral opportunity radar
  - scout_stress_test   -> MODE 2 (brutal critique) immediately followed by
                           MODE 3 (execution blueprint) on the upgraded concept

Both are read-only text generation - no system/file/network side effects -
so they're safe tier regardless of how harsh the critique gets.
"""
from __future__ import annotations

from typing import List

from core.llm_client import OllamaClient
from tools.base import BaseTool, ToolParameter, ToolResult

SYSTEM_PROMPT = """Act as a Dual-Engine Venture Scout, Ruthless Red-Team Strategist, and Chief Operating Officer. Your role is split into three non-negotiable functions: uncovering pre-viral market arbitrage, brutally dismantling business concepts to expose fatal flaws, and drafting bulletproof execution blueprints.

### OPERATING MODES & INSTRUCTIONS

#### MODE 1: PRE-VIRAL TREND & OPPORTUNITY RADAR
When asked to scout or surface business ideas:
1. Ignore saturated mainstream trends (dropshipping, basic AI wrappers, generic agencies).
2. Look at "Weak Signals": regulatory shifts, newly open-sourced infrastructure, cross-industry supply chain gaps, and emerging consumer friction points.
3. Deliver each idea using the Arbitrage Matrix:
   - The Wedge: The non-obvious entry point or pain point.
   - The Catalyst: Why this works right now (and why it wasn't possible 12-18 months ago).
   - Monetization Engine: High-margin revenue model (B2B SaaS, usage-based, productized service, high-ticket niche).
   - The Viral Tipping Point: The specific trigger that will push this from niche to mainstream in 6-12 months.
Deliver exactly 3 ideas, each as a level-3 heading "### Idea 1: <Name>" etc., followed by the four Arbitrage Matrix bullets as **bold** labels.

#### MODE 2: BRUTAL RED-TEAM CRITIQUE & IDEA STRESS-TEST
When analyzing any user-submitted idea, drop all polite validation. Act as an adversarial investor looking for every reason this will fail.
1. The Post-Mortem Premortem: List the top 3-5 structural reasons this business will run out of cash, get commoditized, or fail to gain traction (e.g. CAC-to-LTV death spirals, platform dependency risk, lack of pricing power, distribution bottlenecks, high switching costs for buyers).
2. Hidden Assumptions Exposed: Identify the dangerous, unproven assumptions the founder is making about user behavior, unit economics, or market dynamics.
3. Moat & Defensibility Audit: Rank the competitive moat from 1 to 10. Include a standalone line formatted EXACTLY as "Moat Score: X/10" (X = your rating) before your explanation. Explain why a competitor with $5M or an incumbent could copy this overnight.
4. The Upgrade / Strategic Pivot: Reconstruct the concept to eliminate the primary failure point. Turn a commodity idea into a defensible platform, high-retention system, or proprietary distribution flywheel.

#### MODE 3: ZERO-FRICTION EXECUTION BLUEPRINT
Once the idea is refined (in Mode 2's Upgrade/Strategic Pivot), provide an execution roadmap optimized for speed, minimum capital burn, and maximum de-risking:
1. Phase 0: 48-Hour Validation Smoke Test - confirm paying demand before writing a line of code or signing a contract (presales, concierge MVP, landing page test, cold outbound campaign).
2. Phase 1: Lean MVP Build - the absolute minimum viable stack/mechanism required to deliver core value.
3. Phase 2: Proprietary Distribution Engine - the exact, low-cost channel to acquire the first 100 paying customers without relying on expensive ad spend.
4. Risk-Mitigation Checkpoints: clear kill/pivot thresholds and contingency fallbacks if milestones are missed at each phase.

### TONE & BEHAVIOR RULES
- No Sycophancy: never open with "Great idea!" or "That's very innovative!". Be objective, analytical, and ruthlessly practical.
- Data & Economics First: quantify claims with pricing ranges, estimated conversion metrics, and margins.
- Action-Oriented: every criticism must be coupled with an architectural fix.
- Be dense: short, punchy bullets over long paragraphs. No padding, no throat-clearing."""


def _build_user_message(command: str, target_audience: str, resource_constraints: str) -> str:
    return (
        f"Command: {command}\n"
        f"Target Audience / Market: {target_audience or '(not specified)'}\n"
        f"Resource Constraints: {resource_constraints or '(not specified)'}"
    )


class ScoutFindTrendsTool(BaseTool):
    name = "scout_find_trends"
    description = (
        "Run the Venture Scout to surface 3 pre-viral, non-obvious business opportunities in a "
        "given niche using weak-signal analysis (regulatory shifts, newly open-sourced "
        "infrastructure, supply chain gaps) plus live social signals (real current Reddit "
        "discussion, YouTube videos outperforming their channel's normal reach) where available "
        "- not saturated mainstream trends. Use when the user asks to find/scout business ideas "
        "or opportunities in a space."
    )
    parameters: List[ToolParameter] = [
        ToolParameter(name="niche", type="string", description="The niche/space to scout, e.g. 'regenerative agriculture'."),
        ToolParameter(name="target_audience", type="string", required=False, description="Target audience/market, e.g. 'SMBs', 'Enterprise FinTech'."),
        ToolParameter(name="resource_constraints", type="string", required=False, description="e.g. 'Solo founder, $5,000 budget, 15 hrs/week'."),
    ]

    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client

    async def run(self, niche: str, target_audience: str = "", resource_constraints: str = "", **kwargs) -> ToolResult:
        try:
            user_msg = _build_user_message(f"FIND TRENDS in {niche}", target_audience, resource_constraints)

            try:
                from tools.social_media import gather_social_signals
                signals = await gather_social_signals(niche)
                if signals:
                    user_msg += (
                        f"\n\n{signals}\n\nGround at least one of your three ideas in one of these "
                        f"live signals where genuinely relevant, rather than relying purely on general knowledge."
                    )
            except Exception:
                pass  # live signals are a bonus, never a hard dependency for this tool to work

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            response = await self.llm_client.chat(messages)
            text = response.get("message", {}).get("content", "").strip()
            if not text:
                return ToolResult(success=False, error="Empty response from the reasoning model.")
            return ToolResult(success=True, output={"report": text})
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class ScoutStressTestTool(BaseTool):
    name = "scout_stress_test"
    description = (
        "Run the Red-Team Strategist + COO on a business idea: a brutal adversarial critique "
        "(failure modes, hidden assumptions, a 1-10 moat/defensibility score) followed immediately "
        "by a zero-friction execution blueprint for the upgraded/pivoted version of the idea. Use "
        "when the user wants their business idea stress-tested, critiqued, or pressure-tested."
    )
    parameters: List[ToolParameter] = [
        ToolParameter(name="idea_description", type="string", description="The business idea/concept as you'd pitch it to an investor."),
        ToolParameter(name="target_audience", type="string", required=False, description="Target audience/market, e.g. 'Freelancers'."),
        ToolParameter(name="resource_constraints", type="string", required=False, description="e.g. 'Solo founder, 3-month runway'."),
    ]

    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client

    async def run(self, idea_description: str, target_audience: str = "", resource_constraints: str = "", **kwargs) -> ToolResult:
        try:
            user_msg = _build_user_message(f"STRESS TEST: {idea_description}", target_audience, resource_constraints)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            response = await self.llm_client.chat(messages)
            text = response.get("message", {}).get("content", "").strip()
            if not text:
                return ToolResult(success=False, error="Empty response from the reasoning model.")

            moat_score = None
            for line in text.splitlines():
                if "moat score" in line.lower():
                    digits = "".join(ch for ch in line.split(":")[-1] if ch.isdigit())
                    if digits:
                        moat_score = int(digits[:2]) if len(digits) >= 2 and int(digits[:2]) <= 10 else int(digits[0])
                    break

            return ToolResult(success=True, output={"report": text, "moat_score": moat_score})
        except Exception as e:
            return ToolResult(success=False, error=str(e))
