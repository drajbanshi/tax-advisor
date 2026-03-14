"""Consolidated 1099 data extraction from images using LLM vision."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tax_advisor.models import chat_completion
from tax_advisor.w2_extract import load_image_as_base64


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PayerRecipientInfo:
    """Payer and recipient identification from a consolidated 1099."""

    payer_name: str = ""
    payer_tin: str = ""
    payer_address: str = ""
    recipient_name: str = ""
    recipient_tin: str = ""
    recipient_address: str = ""
    account_number: str = ""


@dataclass
class Form1099BSummary:
    """1099-B aggregate summary totals (short-term / long-term / undetermined)."""

    # Short-term
    short_term_proceeds: str = ""
    short_term_cost_basis: str = ""
    short_term_wash_sale_loss: str = ""
    short_term_gain_loss: str = ""

    # Long-term
    long_term_proceeds: str = ""
    long_term_cost_basis: str = ""
    long_term_wash_sale_loss: str = ""
    long_term_gain_loss: str = ""

    # Undetermined
    undetermined_proceeds: str = ""
    undetermined_cost_basis: str = ""
    undetermined_wash_sale_loss: str = ""
    undetermined_gain_loss: str = ""


@dataclass
class Form1099DIV:
    """1099-DIV dividend income fields."""

    box_1a_total_ordinary_dividends: str = ""
    box_1b_qualified_dividends: str = ""
    box_2a_total_capital_gain_dist: str = ""
    box_2b_unrecap_sec_1250_gain: str = ""
    box_2c_sec_1202_gain: str = ""
    box_2d_collectibles_gain: str = ""
    box_2e_sec_897_ordinary_dividends: str = ""
    box_2f_sec_897_capital_gain: str = ""
    box_3_nondividend_distributions: str = ""
    box_4_federal_tax_withheld: str = ""
    box_5_sec_199a_dividends: str = ""
    box_6_investment_expenses: str = ""
    box_7_foreign_tax_paid: str = ""
    box_8_foreign_country: str = ""
    box_11_exempt_interest_dividends: str = ""
    box_12_specified_pab_interest_dividends: str = ""
    box_13_state: str = ""
    box_14_state_tax_withheld: str = ""
    box_15_state_id_number: str = ""


@dataclass
class Form1099INT:
    """1099-INT interest income fields."""

    box_1_interest_income: str = ""
    box_2_early_withdrawal_penalty: str = ""
    box_3_us_savings_bond_interest: str = ""
    box_4_federal_tax_withheld: str = ""
    box_8_tax_exempt_interest: str = ""
    box_10_market_discount: str = ""
    box_11_bond_premium: str = ""
    box_12_bond_premium_treasury: str = ""
    box_13_bond_premium_tax_exempt: str = ""
    box_15_state: str = ""
    box_16_state_tax_withheld: str = ""
    box_17_state_id_number: str = ""


@dataclass
class Consolidated1099Data:
    """Top-level container for all data extracted from a consolidated 1099."""

    payer_info: PayerRecipientInfo
    form_1099b: Form1099BSummary | None = None
    form_1099div: Form1099DIV | None = None
    form_1099int: Form1099INT | None = None
    tax_year: str = ""


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are a tax document data extraction assistant. You are given one or more \
pages of a consolidated 1099 brokerage statement. Extract the following \
information and return it as a single JSON object.

Use exactly this structure:
{
  "tax_year": "",
  "payer_info": {
    "payer_name": "",
    "payer_tin": "",
    "payer_address": "",
    "recipient_name": "",
    "recipient_tin": "",
    "recipient_address": "",
    "account_number": ""
  },
  "form_1099b": {
    "short_term_proceeds": "",
    "short_term_cost_basis": "",
    "short_term_wash_sale_loss": "",
    "short_term_gain_loss": "",
    "long_term_proceeds": "",
    "long_term_cost_basis": "",
    "long_term_wash_sale_loss": "",
    "long_term_gain_loss": "",
    "undetermined_proceeds": "",
    "undetermined_cost_basis": "",
    "undetermined_wash_sale_loss": "",
    "undetermined_gain_loss": ""
  },
  "form_1099div": {
    "box_1a_total_ordinary_dividends": "",
    "box_1b_qualified_dividends": "",
    "box_2a_total_capital_gain_dist": "",
    "box_2b_unrecap_sec_1250_gain": "",
    "box_2c_sec_1202_gain": "",
    "box_2d_collectibles_gain": "",
    "box_2e_sec_897_ordinary_dividends": "",
    "box_2f_sec_897_capital_gain": "",
    "box_3_nondividend_distributions": "",
    "box_4_federal_tax_withheld": "",
    "box_5_sec_199a_dividends": "",
    "box_6_investment_expenses": "",
    "box_7_foreign_tax_paid": "",
    "box_8_foreign_country": "",
    "box_11_exempt_interest_dividends": "",
    "box_12_specified_pab_interest_dividends": "",
    "box_13_state": "",
    "box_14_state_tax_withheld": "",
    "box_15_state_id_number": ""
  },
  "form_1099int": {
    "box_1_interest_income": "",
    "box_2_early_withdrawal_penalty": "",
    "box_3_us_savings_bond_interest": "",
    "box_4_federal_tax_withheld": "",
    "box_8_tax_exempt_interest": "",
    "box_10_market_discount": "",
    "box_11_bond_premium": "",
    "box_12_bond_premium_treasury": "",
    "box_13_bond_premium_tax_exempt": "",
    "box_15_state": "",
    "box_16_state_tax_withheld": "",
    "box_17_state_id_number": ""
  }
}

Rules:
- Return ONLY the JSON object, no surrounding text or markdown fences.
- For dollar amounts, include the numeric value as a string (e.g. "52000.00"). \
Do NOT include dollar signs or commas.
- For empty/blank fields or fields not present on the form, use an empty string "".
- If a sub-form section (1099-B, 1099-DIV, or 1099-INT) is not present at all \
in the document, set its value to null.
- For 1099-B, extract ONLY the summary/aggregate totals (short-term, long-term, \
undetermined). Do NOT extract individual transactions.
- Losses and negative numbers should be preceded by a minus sign (e.g. "-1234.56").
- Read all pages carefully — information may span across multiple pages.
"""


# ---------------------------------------------------------------------------
# Extraction function
# ---------------------------------------------------------------------------


def extract_1099_from_images(
    image_paths: list[Path],
    model: str,
    bedrock_profile: str | None = None,
) -> Consolidated1099Data:
    """Extract consolidated 1099 data from one or more page images.

    All images are sent together in a single multimodal message so the LLM
    can correlate information across pages.

    Args:
        image_paths: Paths to the 1099 page images.
        model: litellm model identifier.
        bedrock_profile: Optional AWS profile for Bedrock models.

    Returns:
        A populated :class:`Consolidated1099Data` instance.
    """
    content: list[dict[str, Any]] = [
        {"type": "text", "text": _EXTRACTION_PROMPT},
    ]

    for img_path in image_paths:
        b64, mime = load_image_as_base64(img_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{b64}",
                },
            }
        )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": content},
    ]

    response = chat_completion(
        messages=messages,
        model=model,
        temperature=0.0,
        stream=False,
        bedrock_profile=bedrock_profile,
    )

    raw = response.choices[0].message.content.strip()

    # Defensive: strip markdown code fences if the model wrapped the JSON
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        raw = "\n".join(lines)

    data: dict[str, Any] = json.loads(raw)
    return _parse_1099_data(data)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _parse_1099_data(data: dict[str, Any]) -> Consolidated1099Data:
    """Convert a raw JSON dict into a :class:`Consolidated1099Data` instance."""
    # Payer / Recipient info
    pi = data.get("payer_info", {}) or {}
    payer_info = PayerRecipientInfo(
        payer_name=str(pi.get("payer_name", "")),
        payer_tin=str(pi.get("payer_tin", "")),
        payer_address=str(pi.get("payer_address", "")),
        recipient_name=str(pi.get("recipient_name", "")),
        recipient_tin=str(pi.get("recipient_tin", "")),
        recipient_address=str(pi.get("recipient_address", "")),
        account_number=str(pi.get("account_number", "")),
    )

    # 1099-B Summary
    form_1099b: Form1099BSummary | None = None
    b_raw = data.get("form_1099b")
    if b_raw is not None:
        form_1099b = Form1099BSummary(
            short_term_proceeds=str(b_raw.get("short_term_proceeds", "")),
            short_term_cost_basis=str(b_raw.get("short_term_cost_basis", "")),
            short_term_wash_sale_loss=str(b_raw.get("short_term_wash_sale_loss", "")),
            short_term_gain_loss=str(b_raw.get("short_term_gain_loss", "")),
            long_term_proceeds=str(b_raw.get("long_term_proceeds", "")),
            long_term_cost_basis=str(b_raw.get("long_term_cost_basis", "")),
            long_term_wash_sale_loss=str(b_raw.get("long_term_wash_sale_loss", "")),
            long_term_gain_loss=str(b_raw.get("long_term_gain_loss", "")),
            undetermined_proceeds=str(b_raw.get("undetermined_proceeds", "")),
            undetermined_cost_basis=str(b_raw.get("undetermined_cost_basis", "")),
            undetermined_wash_sale_loss=str(b_raw.get("undetermined_wash_sale_loss", "")),
            undetermined_gain_loss=str(b_raw.get("undetermined_gain_loss", "")),
        )

    # 1099-DIV
    form_1099div: Form1099DIV | None = None
    d_raw = data.get("form_1099div")
    if d_raw is not None:
        form_1099div = Form1099DIV(
            box_1a_total_ordinary_dividends=str(d_raw.get("box_1a_total_ordinary_dividends", "")),
            box_1b_qualified_dividends=str(d_raw.get("box_1b_qualified_dividends", "")),
            box_2a_total_capital_gain_dist=str(d_raw.get("box_2a_total_capital_gain_dist", "")),
            box_2b_unrecap_sec_1250_gain=str(d_raw.get("box_2b_unrecap_sec_1250_gain", "")),
            box_2c_sec_1202_gain=str(d_raw.get("box_2c_sec_1202_gain", "")),
            box_2d_collectibles_gain=str(d_raw.get("box_2d_collectibles_gain", "")),
            box_2e_sec_897_ordinary_dividends=str(d_raw.get("box_2e_sec_897_ordinary_dividends", "")),
            box_2f_sec_897_capital_gain=str(d_raw.get("box_2f_sec_897_capital_gain", "")),
            box_3_nondividend_distributions=str(d_raw.get("box_3_nondividend_distributions", "")),
            box_4_federal_tax_withheld=str(d_raw.get("box_4_federal_tax_withheld", "")),
            box_5_sec_199a_dividends=str(d_raw.get("box_5_sec_199a_dividends", "")),
            box_6_investment_expenses=str(d_raw.get("box_6_investment_expenses", "")),
            box_7_foreign_tax_paid=str(d_raw.get("box_7_foreign_tax_paid", "")),
            box_8_foreign_country=str(d_raw.get("box_8_foreign_country", "")),
            box_11_exempt_interest_dividends=str(d_raw.get("box_11_exempt_interest_dividends", "")),
            box_12_specified_pab_interest_dividends=str(d_raw.get("box_12_specified_pab_interest_dividends", "")),
            box_13_state=str(d_raw.get("box_13_state", "")),
            box_14_state_tax_withheld=str(d_raw.get("box_14_state_tax_withheld", "")),
            box_15_state_id_number=str(d_raw.get("box_15_state_id_number", "")),
        )

    # 1099-INT
    form_1099int: Form1099INT | None = None
    i_raw = data.get("form_1099int")
    if i_raw is not None:
        form_1099int = Form1099INT(
            box_1_interest_income=str(i_raw.get("box_1_interest_income", "")),
            box_2_early_withdrawal_penalty=str(i_raw.get("box_2_early_withdrawal_penalty", "")),
            box_3_us_savings_bond_interest=str(i_raw.get("box_3_us_savings_bond_interest", "")),
            box_4_federal_tax_withheld=str(i_raw.get("box_4_federal_tax_withheld", "")),
            box_8_tax_exempt_interest=str(i_raw.get("box_8_tax_exempt_interest", "")),
            box_10_market_discount=str(i_raw.get("box_10_market_discount", "")),
            box_11_bond_premium=str(i_raw.get("box_11_bond_premium", "")),
            box_12_bond_premium_treasury=str(i_raw.get("box_12_bond_premium_treasury", "")),
            box_13_bond_premium_tax_exempt=str(i_raw.get("box_13_bond_premium_tax_exempt", "")),
            box_15_state=str(i_raw.get("box_15_state", "")),
            box_16_state_tax_withheld=str(i_raw.get("box_16_state_tax_withheld", "")),
            box_17_state_id_number=str(i_raw.get("box_17_state_id_number", "")),
        )

    return Consolidated1099Data(
        payer_info=payer_info,
        form_1099b=form_1099b,
        form_1099div=form_1099div,
        form_1099int=form_1099int,
        tax_year=str(data.get("tax_year", "")),
    )


# ---------------------------------------------------------------------------
# Markdown conversion
# ---------------------------------------------------------------------------


def form1099_to_markdown(data: Consolidated1099Data) -> str:
    """Convert a :class:`Consolidated1099Data` instance to structured markdown."""
    lines: list[str] = []

    lines.append("# Consolidated 1099 Tax Statement")
    if data.tax_year:
        lines.append(f"Tax Year: {data.tax_year}")
    lines.append("")

    # Payer info
    pi = data.payer_info
    lines.append("## Payer Information")
    lines.append(f"- **Payer Name:** {pi.payer_name}")
    lines.append(f"- **Payer TIN:** {pi.payer_tin}")
    lines.append(f"- **Payer Address:** {pi.payer_address}")
    lines.append("")

    # Recipient info
    lines.append("## Recipient Information")
    lines.append(f"- **Recipient Name:** {pi.recipient_name}")
    lines.append(f"- **Recipient TIN:** {pi.recipient_tin}")
    lines.append(f"- **Recipient Address:** {pi.recipient_address}")
    lines.append(f"- **Account Number:** {pi.account_number}")
    lines.append("")

    # 1099-B Summary
    if data.form_1099b is not None:
        b = data.form_1099b
        lines.append("## 1099-B — Proceeds from Broker Transactions (Summary)")
        lines.append("")
        lines.append("### Short-Term")
        lines.append(f"- **Proceeds:** {b.short_term_proceeds}")
        lines.append(f"- **Cost Basis:** {b.short_term_cost_basis}")
        lines.append(f"- **Wash Sale Loss Disallowed:** {b.short_term_wash_sale_loss}")
        lines.append(f"- **Gain/Loss:** {b.short_term_gain_loss}")
        lines.append("")
        lines.append("### Long-Term")
        lines.append(f"- **Proceeds:** {b.long_term_proceeds}")
        lines.append(f"- **Cost Basis:** {b.long_term_cost_basis}")
        lines.append(f"- **Wash Sale Loss Disallowed:** {b.long_term_wash_sale_loss}")
        lines.append(f"- **Gain/Loss:** {b.long_term_gain_loss}")
        lines.append("")
        if (
            b.undetermined_proceeds
            or b.undetermined_cost_basis
            or b.undetermined_wash_sale_loss
            or b.undetermined_gain_loss
        ):
            lines.append("### Undetermined Term")
            lines.append(f"- **Proceeds:** {b.undetermined_proceeds}")
            lines.append(f"- **Cost Basis:** {b.undetermined_cost_basis}")
            lines.append(f"- **Wash Sale Loss Disallowed:** {b.undetermined_wash_sale_loss}")
            lines.append(f"- **Gain/Loss:** {b.undetermined_gain_loss}")
            lines.append("")

    # 1099-DIV
    if data.form_1099div is not None:
        d = data.form_1099div
        lines.append("## 1099-DIV — Dividends and Distributions")
        lines.append(f"- **Box 1a — Total ordinary dividends:** {d.box_1a_total_ordinary_dividends}")
        lines.append(f"- **Box 1b — Qualified dividends:** {d.box_1b_qualified_dividends}")
        lines.append(f"- **Box 2a — Total capital gain distributions:** {d.box_2a_total_capital_gain_dist}")
        lines.append(f"- **Box 2b — Unrecap. Sec. 1250 gain:** {d.box_2b_unrecap_sec_1250_gain}")
        lines.append(f"- **Box 2c — Sec. 1202 gain:** {d.box_2c_sec_1202_gain}")
        lines.append(f"- **Box 2d — Collectibles (28%) gain:** {d.box_2d_collectibles_gain}")
        lines.append(f"- **Box 2e — Sec. 897 ordinary dividends:** {d.box_2e_sec_897_ordinary_dividends}")
        lines.append(f"- **Box 2f — Sec. 897 capital gain:** {d.box_2f_sec_897_capital_gain}")
        lines.append(f"- **Box 3 — Nondividend distributions:** {d.box_3_nondividend_distributions}")
        lines.append(f"- **Box 4 — Federal income tax withheld:** {d.box_4_federal_tax_withheld}")
        lines.append(f"- **Box 5 — Sec. 199A dividends:** {d.box_5_sec_199a_dividends}")
        lines.append(f"- **Box 6 — Investment expenses:** {d.box_6_investment_expenses}")
        lines.append(f"- **Box 7 — Foreign tax paid:** {d.box_7_foreign_tax_paid}")
        lines.append(f"- **Box 8 — Foreign country:** {d.box_8_foreign_country}")
        lines.append(f"- **Box 11 — Exempt-interest dividends:** {d.box_11_exempt_interest_dividends}")
        lines.append(f"- **Box 12 — Specified PAB interest dividends:** {d.box_12_specified_pab_interest_dividends}")
        lines.append(f"- **Box 13 — State:** {d.box_13_state}")
        lines.append(f"- **Box 14 — State tax withheld:** {d.box_14_state_tax_withheld}")
        lines.append(f"- **Box 15 — State ID number:** {d.box_15_state_id_number}")
        lines.append("")

    # 1099-INT
    if data.form_1099int is not None:
        i = data.form_1099int
        lines.append("## 1099-INT — Interest Income")
        lines.append(f"- **Box 1 — Interest income:** {i.box_1_interest_income}")
        lines.append(f"- **Box 2 — Early withdrawal penalty:** {i.box_2_early_withdrawal_penalty}")
        lines.append(f"- **Box 3 — Interest on U.S. savings bonds:** {i.box_3_us_savings_bond_interest}")
        lines.append(f"- **Box 4 — Federal income tax withheld:** {i.box_4_federal_tax_withheld}")
        lines.append(f"- **Box 8 — Tax-exempt interest:** {i.box_8_tax_exempt_interest}")
        lines.append(f"- **Box 10 — Market discount:** {i.box_10_market_discount}")
        lines.append(f"- **Box 11 — Bond premium:** {i.box_11_bond_premium}")
        lines.append(f"- **Box 12 — Bond premium on Treasury obligations:** {i.box_12_bond_premium_treasury}")
        lines.append(f"- **Box 13 — Bond premium on tax-exempt bond:** {i.box_13_bond_premium_tax_exempt}")
        lines.append(f"- **Box 15 — State:** {i.box_15_state}")
        lines.append(f"- **Box 16 — State tax withheld:** {i.box_16_state_tax_withheld}")
        lines.append(f"- **Box 17 — State ID number:** {i.box_17_state_id_number}")
        lines.append("")

    return "\n".join(lines)
