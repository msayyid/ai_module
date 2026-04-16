class AcademicAdvisorAgent:
    def __init__(self):
        self.rules = {
            "attendance_low": {
                "risk": "high_risk",
                "description": "Attendance is below 50%"
            },
            "attendance_mid": {
                "risk": "medium_risk",
                "description": "Attendance is between 50% and 69%"
            },
            "coursework_low": {
                "risk": "high_risk",
                "description": "Coursework score is below 50"
            },
            "coursework_mid": {
                "risk": "medium_risk",
                "description": "Coursework score is between 50 and 64"
            },
            "deadline_urgent": {
                "risk": "high_risk",
                "description": "Deadline is very close (3 days or less)"
            },
            "deadline_soon": {
                "risk": "medium_risk",
                "description": "Deadline is approaching (7 days or less)"
            },
            "difficulty_hard": {
                "risk": "medium_risk",
                "description": "The subject feels hard"
            }
        }

        self.resources = {
            "high_risk": [
                "Book an urgent meeting with your tutor",
                "Make a daily revision plan",
                "Speak to your lecturer about weak areas"
            ],
            "medium_risk": [
                "Revise every week",
                "Use past papers",
                "Join or form a study group"
            ],
            "low_risk": [
                "Keep using your current study plan",
                "Practise more exam-style questions"
            ]
        }

        self.priors = {
            "high_risk": 0.3,
            "medium_risk": 0.4,
            "low_risk": 0.3
        }

    def get_number_input(self, prompt, min_value, max_value, current_value=None):
        # If we already guessed something from the first message,
        # let the user keep it by pressing Enter
        while True:
            if current_value is not None:
                value = input(f"{prompt} [{current_value}]: ").strip().lower()
                if value == "":
                    return current_value
            else:
                value = input(prompt).strip().lower()

            if value in ["skip", "dont know", "don't know", "not sure", "i dont know", "i don't know"]:
                return None

            if value.isdigit():
                number = int(value)
                if min_value <= number <= max_value:
                    return number

            print(f"Please enter a number between {min_value} and {max_value}, press Enter to keep the suggested value, or type skip.")

    def get_difficulty_input(self, prompt, current_value=None):
        while True:
            if current_value is not None:
                value = input(f"{prompt} [{current_value}]: ").strip().lower()
                if value == "":
                    return current_value
            else:
                value = input(prompt).strip().lower()

            if value in ["skip", "dont know", "don't know", "not sure"]:
                return None

            if value in ["easy", "medium", "hard"]:
                return value

            print("Please enter easy, medium, hard, press Enter to keep the suggested value, or type skip.")

    def parse_first_message(self, message):
        # This is still simple on purpose.
        # We are not doing fancy NLP here, just checking keywords
        # and looking for obvious numbers.
        msg = message.lower()
        data = {}

        words = msg.replace("%", " ").replace(",", " ").split()
        numbers = []

        for word in words:
            if word.isdigit():
                numbers.append(int(word))

        # Try to understand some obvious clues from the user's first message

        if "attendance" in msg and numbers:
            # Take the first number as attendance if user mentioned attendance
            data["attendance"] = numbers[0]

        if ("coursework" in msg or "cw" in msg or "score" in msg) and numbers:
            # Try to take the last number if coursework is mentioned
            data["coursework"] = numbers[-1]

        if "day" in msg or "deadline" in msg:
            for n in numbers:
                if 0 <= n <= 365:
                    data["deadline"] = n
                    break

        if "tomorrow" in msg:
            data["deadline"] = 1
        elif "next week" in msg:
            data["deadline"] = 7

        if "hard" in msg or "difficult" in msg or "tough" in msg:
            data["difficulty"] = "hard"
        elif "medium" in msg or "okay" in msg or "average" in msg:
            data["difficulty"] = "medium"
        elif "easy" in msg or "fine" in msg:
            data["difficulty"] = "easy"

        # Some rough assumptions if the message gives hints but no numbers
        if "low attendance" in msg and "attendance" not in data:
            data["attendance"] = 45

        if "failed coursework" in msg and "coursework" not in data:
            data["coursework"] = 40

        return data

    def collect_student_data(self, first_message):
        # Start with whatever we could understand from the first message
        guessed_data = self.parse_first_message(first_message)

        print("\nPlease answer these questions. You can press Enter to keep a suggested value or type 'skip' if unsure.\n")

        data = {}

        attendance = self.get_number_input(
            "Attendance percentage (0-100): ",
            0, 100,
            guessed_data.get("attendance")
        )
        coursework = self.get_number_input(
            "Coursework score (0-100): ",
            0, 100,
            guessed_data.get("coursework")
        )
        deadline = self.get_number_input(
            "Days until deadline (0-365): ",
            0, 365,
            guessed_data.get("deadline")
        )
        difficulty = self.get_difficulty_input(
            "Difficulty (easy / medium / hard): ",
            guessed_data.get("difficulty")
        )

        if attendance is not None:
            data["attendance"] = attendance
        if coursework is not None:
            data["coursework"] = coursework
        if deadline is not None:
            data["deadline"] = deadline
        if difficulty is not None:
            data["difficulty"] = difficulty

        return data

    def forward_chain(self, data):
        fired_rules = []

        if "attendance" in data:
            if data["attendance"] < 50:
                fired_rules.append("attendance_low")
            elif data["attendance"] < 70:
                fired_rules.append("attendance_mid")

        if "coursework" in data:
            if data["coursework"] < 50:
                fired_rules.append("coursework_low")
            elif data["coursework"] < 65:
                fired_rules.append("coursework_mid")

        if "deadline" in data:
            if data["deadline"] <= 3:
                fired_rules.append("deadline_urgent")
            elif data["deadline"] <= 7:
                fired_rules.append("deadline_soon")

        if "difficulty" in data:
            if data["difficulty"] == "hard":
                fired_rules.append("difficulty_hard")

        return fired_rules

    def infer_risk_from_rules(self, fired_rules):
        high_count = 0
        medium_count = 0

        for rule in fired_rules:
            if self.rules[rule]["risk"] == "high_risk":
                high_count += 1
            elif self.rules[rule]["risk"] == "medium_risk":
                medium_count += 1

        if high_count >= 2:
            return "high_risk"
        elif high_count == 1 or medium_count >= 2:
            return "medium_risk"
        elif medium_count == 1:
            return "medium_risk"
        else:
            return "low_risk"

    def bayesian_risk_estimate(self, data):
        p_high = self.priors["high_risk"]
        p_medium = self.priors["medium_risk"]
        p_low = self.priors["low_risk"]

        if "attendance" in data:
            att = data["attendance"]
            if att < 50:
                p_high *= 1.6
                p_medium *= 1.2
            elif att < 70:
                p_medium *= 1.4
            else:
                p_low *= 1.4

        if "coursework" in data:
            cw = data["coursework"]
            if cw < 50:
                p_high *= 1.6
            elif cw < 65:
                p_medium *= 1.4
            else:
                p_low *= 1.4

        if "deadline" in data:
            d = data["deadline"]
            if d <= 3:
                p_high *= 1.5
            elif d <= 7:
                p_medium *= 1.3
            else:
                p_low *= 1.2

        if "difficulty" in data:
            diff = data["difficulty"]
            if diff == "hard":
                p_medium *= 1.2
                p_high *= 1.1
            elif diff == "easy":
                p_low *= 1.2

        total = p_high + p_medium + p_low
        p_high /= total
        p_medium /= total
        p_low /= total

        return {
            "high_risk": round(p_high, 3),
            "medium_risk": round(p_medium, 3),
            "low_risk": round(p_low, 3)
        }

    def combine_results(self, rule_risk, bayes_probs):
        final_risk = max(bayes_probs, key=bayes_probs.get)
        confidence = bayes_probs[final_risk]
        return final_risk, confidence

    def decide_action(self, risk):
        text = f"\nDecision: {risk}\n"
        text += "Recommendations:\n"
        for action in self.resources[risk]:
            text += f"- {action}\n"
        return text

    def explain(self, data, fired_rules, rule_risk, bayes_probs, final_risk, confidence):
        # Cleaner explanation text, less raw and easier to read
        lines = []
        lines.append("Explanation:")
        lines.append(f"- I used this information: {data}")
        lines.append(f"- The rule-based part suggests: {rule_risk}")
        lines.append(
            f"- The Bayesian part gives these probabilities: "
            f"high={bayes_probs['high_risk']}, "
            f"medium={bayes_probs['medium_risk']}, "
            f"low={bayes_probs['low_risk']}"
        )
        lines.append(
            f"- So the final answer is {final_risk} "
            f"because it has the highest probability."
        )
        lines.append(f"- Confidence in this result: {confidence}")

        if fired_rules:
            lines.append("- Rules that were triggered:")
            for rule in fired_rules:
                lines.append(f"  * {self.rules[rule]['description']}")
        else:
            lines.append("- No risk rules were triggered from the rule base.")

        lines.append("- This is only a study-support tool, not an official academic judgement.")

        return "\n".join(lines)

    def run(self):
        print("Academic Advisor Agent")
        print("Type 'exit' to stop.\n")

        while True:
            message = input("You: ").strip()

            if message.lower() == "exit":
                print("Goodbye.")
                break

            data = self.collect_student_data(message)

            if len(data) < 2:
                print("\nI do not have enough information for a useful assessment.")
                print("Please provide at least two of these: attendance, coursework, deadline, difficulty.")
                print("-" * 50)
                continue

            fired_rules = self.forward_chain(data)
            rule_risk = self.infer_risk_from_rules(fired_rules)
            bayes_probs = self.bayesian_risk_estimate(data)
            final_risk, confidence = self.combine_results(rule_risk, bayes_probs)

            print(self.decide_action(final_risk))
            print(self.explain(data, fired_rules, rule_risk, bayes_probs, final_risk, confidence))
            print("-" * 50)


if __name__ == "__main__":
    agent = AcademicAdvisorAgent()
    agent.run()