"""
Academic Advisor Agent  —  CMP-N206-0 Coursework 2
A rule-based conversational AI agent that assesses student academic risk.
"""

import re
import matplotlib.pyplot as plt
import numpy as np


class AcademicAdvisorAgent:

    def __init__(self):
        # ── Knowledge Base ────────────────────────────────────────────────
        self.KB = {
            "weights": {
                "attendance": 0.3,
                "coursework": 0.4,
                "deadline":   0.2,
                "difficulty": 0.1
            },

            # Prior (assumed) probabilities before seeing evidence
            "priors": {
                "high_risk":   0.3,
                "medium_risk": 0.4,
                "low_risk":    0.3
            },

            # Structured rules for forward chaining
            # Each rule has: id, field, op, value, consequence, description
            "rules": [
                {
                    "id": "R1", "field": "attendance", "op": "<", "value": 50,
                    "consequence": "high_risk",
                    "description": "low_attendance (<50%) -> high_risk"
                },
                {
                    "id": "R2", "field": "attendance", "op": "<", "value": 70,
                    "consequence": "medium_risk",
                    "description": "moderate_attendance (50-70%) -> medium_risk"
                },
                {
                    "id": "R3", "field": "attendance", "op": ">=", "value": 75,
                    "consequence": "low_risk",
                    "description": "good_attendance (>=75%) -> lower_risk"
                },
                {
                    "id": "R4", "field": "coursework", "op": "<", "value": 50,
                    "consequence": "high_risk",
                    "description": "low_coursework (<50) -> high_risk"
                },
                {
                    "id": "R5", "field": "coursework", "op": "<", "value": 65,
                    "consequence": "medium_risk",
                    "description": "average_coursework (50-65) -> medium_risk"
                },
                {
                    "id": "R6", "field": "coursework", "op": ">=", "value": 65,
                    "consequence": "low_risk",
                    "description": "strong_coursework (>=65) -> lower_risk"
                },
                {
                    "id": "R7", "field": "deadline", "op": "<=", "value": 3,
                    "consequence": "high_risk",
                    "description": "deadline_urgent (<=3 days) -> high_risk"
                },
                {
                    "id": "R8", "field": "deadline", "op": "<=", "value": 7,
                    "consequence": "medium_risk",
                    "description": "deadline_approaching (<=7 days) -> medium_risk"
                },
                {
                    "id": "R9", "field": "difficulty", "op": "==", "value": "hard",
                    "consequence": "medium_risk",
                    "description": "high_difficulty -> increases_risk"
                },
                {
                    "id": "R10", "field": "difficulty", "op": "==", "value": "medium",
                    "consequence": "low_risk",
                    "description": "moderate_difficulty -> minor_risk_increase"
                },
            ],

            # Recommended actions per risk level
            "resources": {
                "high_risk": [
                    "Book an urgent appointment with your academic tutor",
                    "Visit the university learning support centre",
                    "Review past exam papers immediately",
                    "Create a daily revision timetable",
                    "Contact your module lecturer about weak areas"
                ],
                "medium_risk": [
                    "Schedule a weekly study review session",
                    "Form or join a study group",
                    "Use the library's online resources and past papers",
                    "Speak to your personal tutor for guidance"
                ],
                "low_risk": [
                    "Continue your current study strategy",
                    "Challenge yourself with past exam questions",
                    "Consider helping peers — teaching reinforces understanding"
                ]
            },

            # Thresholds used in inference (centralised for consistency)
            "thresholds": {
                "attendance": {"low": 50, "moderate": 70, "good": 75},
                "coursework":  {"low": 50, "moderate": 65},
                "deadline":    {"urgent": 3, "approaching": 7}
            }
        }

        # ── Multi-turn Memory ────────────────────────────────────────────
        self.memory = {
            "conversation_history": [],   # list of {turn, user, agent, risk}
            "student_data": {},           # persists student data across turns
            "risk_history": [],           # track risk level over time
            "turn_count": 0
        }

    # ══════════════════════════════════════════════════════════════════════
    # MEMORY
    # ══════════════════════════════════════════════════════════════════════

    def reset_memory(self):
        """Reset conversation memory (call between independent test scenarios)."""
        self.memory = {
            "conversation_history": [],
            "student_data": {},
            "risk_history": [],
            "turn_count": 0
        }

    def update_memory(self, user_msg, agent_response, risk):
        """Store each turn in conversation history."""
        self.memory["turn_count"] += 1
        self.memory["conversation_history"].append({
            "turn":  self.memory["turn_count"],
            "user":  user_msg,
            "agent": agent_response,
            "risk":  risk
        })
        if risk not in ("unknown", None):
            self.memory["risk_history"].append(risk)

    # ══════════════════════════════════════════════════════════════════════
    # PERCEPTION — interpret user input
    # ══════════════════════════════════════════════════════════════════════

    def interpret_input(self, student_data):
        """
        Merge new data with persistent memory so previous answers
        are retained across conversation turns.
        """
        self.memory["student_data"].update(student_data)
        return self.memory["student_data"]

    def interpret_user_message(self, message):
        """
        Parse free-text input using regex and keyword rules.

        Improvements over v1:
        - Regex patterns handle flexible word order (e.g. '72 in coursework')
        - No silent key-collision bugs
        - Richer synonym/phrase coverage

        Returns: (intent, extracted_data_dict)
        """
        msg = message.lower()
        data = {}

        # ── Intent detection ───────────────────────────────────────────
        # Triggers on either distress language OR explicit academic metric keywords
        risk_keywords = [
            "struggle", "struggling", "help", "risk", "fail", "failing",
            "worried", "behind", "stressed", "panic", "bad", "low"
        ]
        metric_keywords = [
            "attendance", "coursework", "deadline", "score", "exam", "cw"
        ]
        intent = (
            "risk_assessment"
            if any(w in msg for w in risk_keywords) or any(w in msg for w in metric_keywords)
            else "general"
        )

        # ── Attendance ─────────────────────────────────────────────────
        # Matches: "attendance is 45", "45% attendance", "45 attendance"
        m = re.search(r'attendance\D{0,10}?(\d{1,3})', msg)
        if not m:
            m = re.search(r'(\d{1,3})\s*%?\s*attendance', msg)
        if m:
            data["attendance"] = int(m.group(1))
        elif "missed" in msg or "low attendance" in msg or "skipped" in msg:
            data["attendance"] = 40   # conservative assumption

        # ── Coursework ─────────────────────────────────────────────────
        # Matches: "coursework 55", "scored 72 in coursework", "cw is 60"
        m = re.search(r'(?:coursework|cw|score)\D{0,10}?(\d{1,3})', msg)
        if not m:
            m = re.search(r'(\d{1,3})\D{0,10}?(?:coursework|cw)', msg)
        if m:
            data["coursework"] = int(m.group(1))
        elif "failed coursework" in msg or "failed my coursework" in msg:
            data["coursework"] = 40

        # ── Deadline ───────────────────────────────────────────────────
        # Matches: "2 days", "deadline in 5 days", "3 days left"
        m = re.search(r'(\d+)\s*day', msg)
        if m:
            data["deadline"] = int(m.group(1))
        elif "tonight" in msg or "tomorrow" in msg or "urgent" in msg:
            data["deadline"] = 1
        elif "this week" in msg:
            data["deadline"] = 5
        elif "next week" in msg:
            data["deadline"] = 9

        # ── Difficulty ─────────────────────────────────────────────────
        if any(w in msg for w in ["hard", "difficult", "tough", "confusing", "lost"]):
            data["difficulty"] = "hard"
        elif any(w in msg for w in ["medium", "okay", "ok", "average", "alright"]):
            data["difficulty"] = "medium"
        elif any(w in msg for w in ["easy", "fine", "straightforward", "no problem"]):
            data["difficulty"] = "easy"

        return intent, data

    # ══════════════════════════════════════════════════════════════════════
    # INFERENCE — forward chaining + weighted scoring
    # ══════════════════════════════════════════════════════════════════════

    def forward_chain(self, data):
        """
        Apply each rule in the KB against student data.
        Rules are evaluated left-to-right; all matching rules fire.

        This is forward chaining: we start from facts (data)
        and derive consequences (risk indicators).

        Returns: list of fired rule dicts
        """
        fired = []
        for rule in self.KB["rules"]:
            field = rule["field"]
            if field not in data:
                continue
            val = data[field]
            op  = rule["op"]

            if   op == "<"  and val <  rule["value"]: fired.append(rule)
            elif op == "<=" and val <= rule["value"]: fired.append(rule)
            elif op == ">=" and val >= rule["value"]: fired.append(rule)
            elif op == ">"  and val >  rule["value"]: fired.append(rule)
            elif op == "==" and str(val) == str(rule["value"]): fired.append(rule)

        return fired

    def infer_risk(self, data):
        """
        Weighted risk scoring, informed by forward-chained rules.

        Scoring logic:
          - Each factor contributes up to its weight (0–1 scale after normalisation)
          - Partial contributions applied for borderline values
          - Final score normalised by total available weight

        Returns: (risk_level, confidence, explanation_list, fired_rules, factor_scores)
        """
        explanation  = []
        score        = 0.0
        total_weight = 0.0
        factor_scores = {}

        # Run forward chaining first (for explainability)
        fired_rules = self.forward_chain(data)

        t = self.KB["thresholds"]
        w = self.KB["weights"]

        # ── Attendance ─────────────────────────────────────────────────
        if "attendance" in data:
            att = data["attendance"]
            if att < t["attendance"]["low"]:
                contribution = w["attendance"] * 1.0
                explanation.append("Very low attendance (<50%)")
            elif att < t["attendance"]["moderate"]:
                contribution = w["attendance"] * 0.5
                explanation.append("Moderate attendance (50–70%)")
            else:
                contribution = 0.0
                explanation.append("Good attendance (>=70%)")
            score                    += contribution
            factor_scores["attendance"] = round(contribution, 3)
            total_weight             += w["attendance"]

        # ── Coursework ─────────────────────────────────────────────────
        if "coursework" in data:
            cw = data["coursework"]
            if cw < t["coursework"]["low"]:
                contribution = w["coursework"] * 1.0
                explanation.append("Low coursework score (<50)")
            elif cw < t["coursework"]["moderate"]:
                contribution = w["coursework"] * 0.5
                explanation.append("Average coursework (50–65)")
            else:
                contribution = 0.0
                explanation.append("Strong coursework (>=65)")
            score                    += contribution
            factor_scores["coursework"] = round(contribution, 3)
            total_weight             += w["coursework"]

        # ── Deadline ───────────────────────────────────────────────────
        if "deadline" in data:
            d = data["deadline"]
            if d <= t["deadline"]["urgent"]:
                contribution = w["deadline"] * 1.0
                explanation.append("Deadline very close (<=3 days)")
            elif d <= t["deadline"]["approaching"]:
                contribution = w["deadline"] * 0.5
                explanation.append("Deadline approaching (<=7 days)")
            else:
                contribution = 0.0
                explanation.append("Plenty of time before deadline")
            score                   += contribution
            factor_scores["deadline"] = round(contribution, 3)
            total_weight            += w["deadline"]

        # ── Difficulty ─────────────────────────────────────────────────
        if "difficulty" in data:
            diff = data["difficulty"]
            if diff == "hard":
                contribution = w["difficulty"] * 1.0
                explanation.append("Subject perceived as difficult")
            elif diff == "medium":
                contribution = w["difficulty"] * 0.5
                explanation.append("Moderate difficulty")
            else:
                contribution = 0.0
                explanation.append("Subject perceived as easy")
            score                    += contribution
            factor_scores["difficulty"] = round(contribution, 3)
            total_weight             += w["difficulty"]

        # ── No data edge case ───────────────────────────────────────────
        if total_weight == 0:
            return "unknown", 0.0, ["No useful data provided"], [], {}

        # Normalise to 0–1
        risk_score = score / total_weight
        factor_scores["total_risk_score"] = round(risk_score, 3)

        # Classify risk
        if risk_score > 0.7:
            risk = "high_risk"
        elif risk_score > 0.4:
            risk = "medium_risk"
        else:
            risk = "low_risk"

        # Confidence = fraction of possible weight that was available
        confidence = round(total_weight, 2)
        return risk, confidence, explanation, fired_rules, factor_scores

    # ══════════════════════════════════════════════════════════════════════
    # BAYESIAN REASONING — probabilistic inference
    # ══════════════════════════════════════════════════════════════════════

    def bayesian_risk_estimate(self, data):
        """
        Update prior probabilities with evidence using Bayesian-style reasoning.

        Each piece of evidence acts as a likelihood multiplier:
        P(risk | evidence) ∝ P(risk) × P(evidence | risk)

        The likelihood multipliers are heuristic estimates:
            1.5 = strong evidence
            1.3 = moderate evidence
            1.2 = weak evidence

        Returns: posterior probability distribution dict
        """
        p_high   = self.KB["priors"]["high_risk"]
        p_medium = self.KB["priors"]["medium_risk"]
        p_low    = self.KB["priors"]["low_risk"]

        if "attendance" in data:
            att = data["attendance"]
            if att < 50:
                p_high   *= 1.5
                p_medium *= 1.2
            elif att < 70:
                p_medium *= 1.3
            elif att >= 75:
                p_low    *= 1.5

        if "coursework" in data:
            cw = data["coursework"]
            if cw < 50:
                p_high   *= 1.5
            elif cw < 65:
                p_medium *= 1.3
            elif cw >= 65:
                p_low    *= 1.5

        if "deadline" in data:
            d = data["deadline"]
            if d <= 3:
                p_high   *= 1.4
            elif d <= 7:
                p_medium *= 1.2
            else:
                p_low    *= 1.2

        if "difficulty" in data:
            diff = data["difficulty"]
            if diff == "hard":
                p_high   *= 1.2
                p_medium *= 1.1
            elif diff == "easy":
                p_low    *= 1.2

        # Normalise so probabilities sum to 1
        total    = p_high + p_medium + p_low
        p_high   /= total
        p_medium /= total
        p_low    /= total

        return {
            "high_risk":   round(p_high,   3),
            "medium_risk": round(p_medium, 3),
            "low_risk":    round(p_low,    3)
        }

    # ══════════════════════════════════════════════════════════════════════
    # DECISION LOGIC
    # ══════════════════════════════════════════════════════════════════════

    def decide_action(self, risk_level, confidence):
        """
        Return a decision message with recommended resources.
        Handles low-confidence case (uncertainty) explicitly.
        """
        if confidence < 0.5:
            return (
                "I do not have enough information to give a confident assessment. "
                "Please provide at least two of: attendance %, coursework score, "
                "days until deadline, or subject difficulty."
            )

        resources    = self.KB["resources"].get(risk_level, [])
        resource_str = "\n  • " + "\n  • ".join(resources) if resources else ""

        if risk_level == "high_risk":
            return f"You are at HIGH risk. Immediate action is recommended:{resource_str}"
        elif risk_level == "medium_risk":
            return f"You are at MODERATE risk. Focus on consistency and weak areas:{resource_str}"
        elif risk_level == "low_risk":
            return f"You are performing well. Keep up your study strategy:{resource_str}"
        else:
            return "I need more information to assess your situation."

    # ══════════════════════════════════════════════════════════════════════
    # EXPLAINABILITY
    # ══════════════════════════════════════════════════════════════════════

    def explain(self, explanation, confidence, risk, fired_rules, factor_scores):
        """
        Produce a transparent, structured explanation of the agent's reasoning.
        Answers the question: 'Why did you give that answer?'

        Includes:
        - Risk classification and score
        - Per-factor contributions
        - Rules that fired (forward chaining trace)
        - Confidence level
        """
        lines = [
            f"Risk level   : {risk}",
            f"Risk score   : {factor_scores.get('total_risk_score', 'N/A')}  "
            f"(0 = no risk, 1 = maximum risk)",
            f"Confidence   : {confidence}  "
            f"(max 1.0 — increases as more data is provided)",
            "",
            "Factor contributions to risk score:",
        ]
        for k, v in factor_scores.items():
            if k != "total_risk_score":
                max_w = self.KB["weights"].get(k, "?")
                lines.append(f"  {k:12s}: {v}  (max possible: {max_w})")

        lines += ["", "Reasoning:"]
        for item in explanation:
            lines.append(f"  • {item}")

        if fired_rules:
            lines += ["", "Rules fired (forward chaining trace):"]
            for r in fired_rules:
                lines.append(f"  [{r['id']}] {r['description']}")
        else:
            lines += ["", "No rules fired (insufficient data)."]

        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════════════
    # UNCERTAINTY HANDLING
    # ══════════════════════════════════════════════════════════════════════

    def ask_missing_info(self, data):
        """
        Prompt only for fields that are missing.
        This implements targeted clarification — a key uncertainty-handling pattern.
        """
        prompts = {
            "attendance": "What is your current attendance percentage? (0-100): ",
            "coursework": "What is your latest coursework score? (0-100): ",
            "deadline":   "How many days until your next deadline? ",
            "difficulty": "How difficult is the subject? (easy / medium / hard): "
        }
        for field, prompt in prompts.items():
            if field not in data:
                val = input(prompt).strip()
                if val:
                    if field in ("attendance", "coursework", "deadline"):
                        try:
                            data[field] = int(val)
                        except ValueError:
                            print(f"  (Could not parse '{val}' — skipping {field})")
                    else:
                        data[field] = val.lower()
        return data

    # ══════════════════════════════════════════════════════════════════════
    # VISUALISATIONS
    # ══════════════════════════════════════════════════════════════════════

    def plot_factor_contributions(self, factor_scores, title="Risk Factor Contributions"):
        """
        Bar chart showing how each factor contributed to the overall risk score.
        Grey bars show the maximum each factor could have contributed.
        """
        factors   = [k for k in factor_scores if k != "total_risk_score"]
        values    = [factor_scores[k] for k in factors]
        max_vals  = [self.KB["weights"].get(k, 0) for k in factors]
        colors    = [
            "#e74c3c" if v >= 0.2 else "#f39c12" if v >= 0.08 else "#2ecc71"
            for v in values
        ]

        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(factors))

        ax.bar(x, max_vals, width=0.5, color="#dfe6e9", label="Max possible contribution")
        bars = ax.bar(x, values, width=0.5, color=colors, label="Actual contribution")

        total = factor_scores.get("total_risk_score", 0)
        ax.axhline(
            y=total, color="#2c3e50", linestyle="--", linewidth=1.2,
            label=f"Overall risk score: {total}"
        )

        ax.set_xticks(x)
        ax.set_xticklabels([f.capitalize() for f in factors])
        ax.set_ylabel("Weighted risk contribution")
        ax.set_title(title)
        ax.set_ylim(0, 0.55)
        ax.legend(loc="upper right", fontsize=8)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center", fontsize=9, fontweight="bold"
            )

        plt.tight_layout()
        plt.show()

    def plot_bayesian_probabilities(self, probabilities, title="Bayesian Risk Probability Distribution"):
        """
        Bar chart of posterior probabilities for each risk level.
        Dashed line marks the uniform prior (1/3 each).
        """
        labels = ["High Risk", "Medium Risk", "Low Risk"]
        keys   = ["high_risk", "medium_risk", "low_risk"]
        values = [probabilities[k] for k in keys]
        colors = ["#e74c3c", "#f39c12", "#2ecc71"]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=1.2)

        ax.axhline(y=1/3, color="#7f8c8d", linestyle="--", linewidth=1,
                   label="Uniform prior (0.33)")
        ax.set_ylabel("Posterior Probability")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center", fontsize=11, fontweight="bold"
            )

        plt.tight_layout()
        plt.show()

    def plot_risk_history(self, title="Risk Level Over Conversation Turns"):
        """
        Line chart showing how the assessed risk level changed across turns.
        Requires at least 2 assessments in memory.
        """
        if len(self.memory["risk_history"]) < 2:
            print("Not enough history to plot a trend (need at least 2 assessments).")
            return

        mapping = {"high_risk": 3, "medium_risk": 2, "low_risk": 1}
        y = [mapping.get(r, 0) for r in self.memory["risk_history"]]
        x = list(range(1, len(y) + 1))

        colors_map = {1: "#2ecc71", 2: "#f39c12", 3: "#e74c3c"}
        point_colors = [colors_map.get(v, "grey") for v in y]

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(x, y, color="#3498db", linewidth=2, zorder=1)
        ax.scatter(x, y, c=point_colors, s=80, zorder=2, edgecolors="white", linewidths=1.5)

        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(["Low", "Medium", "High"])
        ax.set_xlabel("Assessment Turn")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_ylim(0.5, 3.5)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.show()

    # ══════════════════════════════════════════════════════════════════════
    # FULL PIPELINE
    # ══════════════════════════════════════════════════════════════════════

    def run(self, student_data, user_msg=""):
        """
        Main pipeline:
          1. Merge input with memory (persistence)
          2. Forward chain rules against data
          3. Compute weighted risk score
          4. Run Bayesian probability update
          5. Generate decision and explanation
          6. Store turn in memory
        """
        data          = self.interpret_input(student_data)
        risk, confidence, explanation, fired_rules, factor_scores = self.infer_risk(data)
        probabilities = self.bayesian_risk_estimate(data)
        decision      = self.decide_action(risk, confidence)
        explanation_text = self.explain(
            explanation, confidence, risk, fired_rules, factor_scores
        )

        self.update_memory(user_msg, decision, risk)

        return {
            "decision":      decision,
            "explanation":   explanation_text,
            "probabilities": probabilities,
            "factor_scores": factor_scores,
            "fired_rules":   [f"[{r['id']}] {r['description']}" for r in fired_rules],
            "risk":          risk,
            "confidence":    confidence
        }

    # ══════════════════════════════════════════════════════════════════════
    # INTERACTIVE LOOPS
    # ══════════════════════════════════════════════════════════════════════

    def run_smart_agent(self):
        """
        Conversational loop using free-text input.
        Extracts data from natural language, fills gaps interactively,
        and maintains memory across turns.
        """
        print("=" * 60)
        print("  Academic Advisor Agent   (type 'exit' to quit)")
        print("=" * 60)
        print("Tip: describe your situation naturally, e.g.")
        print("  'My attendance is 45 and my deadline is in 2 days'\n")

        while True:
            message = input("You: ").strip()

            if message.lower() in ("exit", "quit"):
                print("Goodbye. Good luck with your studies!")
                if len(self.memory["risk_history"]) >= 2:
                    self.plot_risk_history()
                break

            if not message:
                continue

            intent, data = self.interpret_user_message(message)

            if intent != "risk_assessment":
                print(
                    "Agent: I can help you assess your academic risk. "
                    "Describe your attendance, coursework, upcoming deadlines, "
                    "or how you're finding your subject.\n"
                )
                continue

            # Only ask follow-up questions if we extracted very little
            if len(data) < 2:
                data = self.ask_missing_info(data)

            result = self.run(data, user_msg=message)

            print(f"\nAgent: {result['decision']}\n")
            print(result["explanation"])
            print(f"\nBayesian probabilities: {result['probabilities']}")
            print("-" * 60 + "\n")


if __name__ == "__main__":
    agent = AcademicAdvisorAgent()
    agent.run_smart_agent()