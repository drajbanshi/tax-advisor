"""YAML template generation and loading for W-2 and 1099 forms."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# W-2 YAML template
# ---------------------------------------------------------------------------

W2_TEMPLATE_YAML = """\
# W-2 Wage and Tax Statement — YAML Template
# Fill in the values below and upload with: /upload w2_template.yaml
# For dollar amounts, enter the numeric value (e.g. 52000.00). No $ signs or commas.
# Leave fields blank ("") if not applicable.

# Employee / Employer Identifiers
employee_ssn: ""          # (a) Employee's Social Security number
employer_ein: ""          # (b) Employer identification number (EIN)
employer_name: ""         # (c) Employer's name
employer_address: ""      # Employer's address (city, state, ZIP)
employee_name: ""         # (e/f) Employee's name
employee_address: ""      # Employee's address (city, state, ZIP)
control_number: ""        # (d) Control number

# Compensation (Boxes 1-11)
box1_wages: ""            # Box 1 — Wages, tips, other compensation
box2_fed_tax_withheld: "" # Box 2 — Federal income tax withheld
box3_ss_wages: ""         # Box 3 — Social Security wages
box4_ss_tax_withheld: ""  # Box 4 — Social Security tax withheld
box5_medicare_wages: ""   # Box 5 — Medicare wages and tips
box6_medicare_tax_withheld: ""  # Box 6 — Medicare tax withheld
box7_ss_tips: ""          # Box 7 — Social Security tips
box8_allocated_tips: ""   # Box 8 — Allocated tips
box9: ""                  # Box 9 — (blank / verification code)
box10_dependent_care: ""  # Box 10 — Dependent care benefits
box11_nonqualified_plans: ""  # Box 11 — Nonqualified plans

# Box 12 — Codes (up to 4 entries: 12a-12d)
# Delete unused entries. Common codes: D (401k), DD (health insurance cost), W (HSA)
box12:
  - code: ""
    amount: ""
# - code: ""
#   amount: ""

# Box 13 — Checkboxes (true or false)
box13_statutory_employee: false
box13_retirement_plan: false
box13_third_party_sick_pay: false

# Box 14 — Other
box14_other: ""

# State and Local (Boxes 15-20)
box15_state: ""               # Box 15 — State abbreviation
box15_employer_state_id: ""   # Box 15 — Employer's state ID number
box16_state_wages: ""         # Box 16 — State wages, tips, etc.
box17_state_tax: ""           # Box 17 — State income tax
box18_local_wages: ""         # Box 18 — Local wages, tips, etc.
box19_local_tax: ""           # Box 19 — Local income tax
box20_locality_name: ""       # Box 20 — Locality name
"""

# ---------------------------------------------------------------------------
# 1099 YAML template
# ---------------------------------------------------------------------------

FORM1099_TEMPLATE_YAML = """\
# Consolidated 1099 Tax Statement — YAML Template
# Fill in the values below and upload with: /upload 1099_template.yaml
# For dollar amounts, enter the numeric value (e.g. 1234.56). No $ signs or commas.
# Losses/negative numbers: use a minus sign (e.g. -1234.56).
# Delete any section (form_1099b, form_1099div, form_1099int) that does not apply.
# Leave individual fields blank ("") if not applicable.

tax_year: ""

# Payer and Recipient Information
payer_info:
  payer_name: ""
  payer_tin: ""             # Payer's TIN (Tax Identification Number)
  payer_address: ""
  recipient_name: ""
  recipient_tin: ""         # Recipient's TIN
  recipient_address: ""
  account_number: ""

# 1099-B — Proceeds from Broker Transactions (Summary)
# Delete this entire section if you have no 1099-B.
form_1099b:
  short_term_proceeds: ""
  short_term_cost_basis: ""
  short_term_wash_sale_loss: ""
  short_term_gain_loss: ""
  long_term_proceeds: ""
  long_term_cost_basis: ""
  long_term_wash_sale_loss: ""
  long_term_gain_loss: ""
  undetermined_proceeds: ""
  undetermined_cost_basis: ""
  undetermined_wash_sale_loss: ""
  undetermined_gain_loss: ""

# 1099-DIV — Dividends and Distributions
# Delete this entire section if you have no 1099-DIV.
form_1099div:
  box_1a_total_ordinary_dividends: ""
  box_1b_qualified_dividends: ""
  box_2a_total_capital_gain_dist: ""
  box_2b_unrecap_sec_1250_gain: ""
  box_2c_sec_1202_gain: ""
  box_2d_collectibles_gain: ""
  box_2e_sec_897_ordinary_dividends: ""
  box_2f_sec_897_capital_gain: ""
  box_3_nondividend_distributions: ""
  box_4_federal_tax_withheld: ""
  box_5_sec_199a_dividends: ""
  box_6_investment_expenses: ""
  box_7_foreign_tax_paid: ""
  box_8_foreign_country: ""
  box_11_exempt_interest_dividends: ""
  box_12_specified_pab_interest_dividends: ""
  box_13_state: ""
  box_14_state_tax_withheld: ""
  box_15_state_id_number: ""

# 1099-INT — Interest Income
# Delete this entire section if you have no 1099-INT.
form_1099int:
  box_1_interest_income: ""
  box_2_early_withdrawal_penalty: ""
  box_3_us_savings_bond_interest: ""
  box_4_federal_tax_withheld: ""
  box_8_tax_exempt_interest: ""
  box_10_market_discount: ""
  box_11_bond_premium: ""
  box_12_bond_premium_treasury: ""
  box_13_bond_premium_tax_exempt: ""
  box_15_state: ""
  box_16_state_tax_withheld: ""
  box_17_state_id_number: ""
"""


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def generate_w2_template(output_dir: Path | None = None) -> Path:
    """Write a blank W-2 YAML template to *output_dir* (default: cwd).

    Returns the path of the generated file.
    """
    dest = (output_dir or Path.cwd()) / "w2_template.yaml"
    dest.write_text(W2_TEMPLATE_YAML)
    return dest


def generate_1099_template(output_dir: Path | None = None) -> Path:
    """Write a blank 1099 YAML template to *output_dir* (default: cwd).

    Returns the path of the generated file.
    """
    dest = (output_dir or Path.cwd()) / "1099_template.yaml"
    dest.write_text(FORM1099_TEMPLATE_YAML)
    return dest


def classify_yaml_template(data: dict[str, Any]) -> str:
    """Classify a parsed YAML dict as ``"w2"``, ``"1099"``, or ``"unknown"``.

    Detection is based on top-level keys:
    - ``payer_info`` → 1099
    - ``employee_ssn`` or ``employer_ein`` → W-2
    """
    if "payer_info" in data:
        return "1099"
    if "employee_ssn" in data or "employer_ein" in data:
        return "w2"
    return "unknown"


def load_yaml_template(yaml_path: Path) -> tuple[str, dict[str, Any]]:
    """Read and classify a YAML template file.

    Returns ``(form_type, data)`` where *form_type* is one of
    ``"w2"``, ``"1099"``, or ``"unknown"``.

    Raises:
        FileNotFoundError: If *yaml_path* does not exist.
        ValueError: If the file is not valid YAML or is empty.
    """
    path = Path(yaml_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    text = path.read_text()
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path.name}, got {type(data).__name__}")

    form_type = classify_yaml_template(data)
    return form_type, data
