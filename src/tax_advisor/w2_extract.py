"""W-2 data extraction from images using LLM vision."""

from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tax_advisor.models import chat_completion


@dataclass
class W2Box12Entry:
    """A single Box 12 entry (code + amount)."""

    code: str = ""
    amount: str = ""


@dataclass
class W2Data:
    """Typed representation of all W-2 form fields."""

    # Employee / employer identifiers
    employee_ssn: str = ""
    employer_ein: str = ""
    employer_name: str = ""
    employer_address: str = ""
    employee_name: str = ""
    employee_address: str = ""
    control_number: str = ""  # Box d

    # Compensation boxes 1-11
    box1_wages: str = ""
    box2_fed_tax_withheld: str = ""
    box3_ss_wages: str = ""
    box4_ss_tax_withheld: str = ""
    box5_medicare_wages: str = ""
    box6_medicare_tax_withheld: str = ""
    box7_ss_tips: str = ""
    box8_allocated_tips: str = ""
    box9: str = ""  # (blank / verification code)
    box10_dependent_care: str = ""
    box11_nonqualified_plans: str = ""

    # Box 12a-d
    box12: list[W2Box12Entry] = field(default_factory=list)

    # Box 13 checkboxes
    box13_statutory_employee: bool = False
    box13_retirement_plan: bool = False
    box13_third_party_sick_pay: bool = False

    # Box 14 (other)
    box14_other: str = ""

    # State / local (boxes 15-20)
    box15_state: str = ""
    box15_employer_state_id: str = ""
    box16_state_wages: str = ""
    box17_state_tax: str = ""
    box18_local_wages: str = ""
    box19_local_tax: str = ""
    box20_locality_name: str = ""


_EXTRACTION_PROMPT = """\
You are a tax document data extraction assistant. Extract ALL fields from this \
W-2 (Wage and Tax Statement) image and return them as a single JSON object.

Use exactly these keys:
{
  "employee_ssn": "",
  "employer_ein": "",
  "employer_name": "",
  "employer_address": "",
  "employee_name": "",
  "employee_address": "",
  "control_number": "",
  "box1_wages": "",
  "box2_fed_tax_withheld": "",
  "box3_ss_wages": "",
  "box4_ss_tax_withheld": "",
  "box5_medicare_wages": "",
  "box6_medicare_tax_withheld": "",
  "box7_ss_tips": "",
  "box8_allocated_tips": "",
  "box9": "",
  "box10_dependent_care": "",
  "box11_nonqualified_plans": "",
  "box12": [{"code": "", "amount": ""}],
  "box13_statutory_employee": false,
  "box13_retirement_plan": false,
  "box13_third_party_sick_pay": false,
  "box14_other": "",
  "box15_state": "",
  "box15_employer_state_id": "",
  "box16_state_wages": "",
  "box17_state_tax": "",
  "box18_local_wages": "",
  "box19_local_tax": "",
  "box20_locality_name": ""
}

Rules:
- Return ONLY the JSON object, no surrounding text or markdown fences.
- For dollar amounts, include the numeric value as a string (e.g. "52000.00"). Do NOT include dollar signs or commas.
- For empty/blank fields, use an empty string "".
- For checkboxes (box 13), use true/false booleans.
- For box 12, include only entries that are present (up to 4: 12a-12d).
- Read the form carefully — labels and values may not be aligned in a simple grid.
"""


def load_image_as_base64(image_path: str | Path) -> tuple[str, str]:
    """Read an image file and return ``(base64_data, mime_type)``.

    Raises:
        FileNotFoundError: If the image does not exist.
        ValueError: If the MIME type cannot be determined or is unsupported.
    """
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None or not mime_type.startswith("image/"):
        raise ValueError(
            f"Unsupported or unrecognised image type for {path.name}. "
            "Supported formats: PNG, JPEG, GIF, WEBP."
        )

    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8"), mime_type


def extract_w2_from_image(
    image_path: str | Path,
    model: str,
    bedrock_profile: str | None = None,
) -> W2Data:
    """Send a W-2 image to the LLM via vision and return structured data.

    Args:
        image_path: Path to a W-2 image (PNG/JPEG).
        model: litellm model identifier.
        bedrock_profile: Optional AWS profile for Bedrock models.

    Returns:
        A populated :class:`W2Data` instance.
    """
    b64, mime = load_image_as_base64(image_path)

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _EXTRACTION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64}",
                    },
                },
            ],
        }
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
        # Remove first line (```json or ```) and last line (```)
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        raw = "\n".join(lines)

    data: dict[str, Any] = json.loads(raw)
    return _parse_w2_data(data)


def _parse_w2_data(data: dict[str, Any]) -> W2Data:
    """Convert a raw JSON dict into a :class:`W2Data` instance."""
    box12_raw = data.get("box12", [])
    box12_entries = [
        W2Box12Entry(code=str(e.get("code", "")), amount=str(e.get("amount", "")))
        for e in box12_raw
        if isinstance(e, dict)
    ]

    return W2Data(
        employee_ssn=str(data.get("employee_ssn", "")),
        employer_ein=str(data.get("employer_ein", "")),
        employer_name=str(data.get("employer_name", "")),
        employer_address=str(data.get("employer_address", "")),
        employee_name=str(data.get("employee_name", "")),
        employee_address=str(data.get("employee_address", "")),
        control_number=str(data.get("control_number", "")),
        box1_wages=str(data.get("box1_wages", "")),
        box2_fed_tax_withheld=str(data.get("box2_fed_tax_withheld", "")),
        box3_ss_wages=str(data.get("box3_ss_wages", "")),
        box4_ss_tax_withheld=str(data.get("box4_ss_tax_withheld", "")),
        box5_medicare_wages=str(data.get("box5_medicare_wages", "")),
        box6_medicare_tax_withheld=str(data.get("box6_medicare_tax_withheld", "")),
        box7_ss_tips=str(data.get("box7_ss_tips", "")),
        box8_allocated_tips=str(data.get("box8_allocated_tips", "")),
        box9=str(data.get("box9", "")),
        box10_dependent_care=str(data.get("box10_dependent_care", "")),
        box11_nonqualified_plans=str(data.get("box11_nonqualified_plans", "")),
        box12=box12_entries,
        box13_statutory_employee=bool(data.get("box13_statutory_employee", False)),
        box13_retirement_plan=bool(data.get("box13_retirement_plan", False)),
        box13_third_party_sick_pay=bool(data.get("box13_third_party_sick_pay", False)),
        box14_other=str(data.get("box14_other", "")),
        box15_state=str(data.get("box15_state", "")),
        box15_employer_state_id=str(data.get("box15_employer_state_id", "")),
        box16_state_wages=str(data.get("box16_state_wages", "")),
        box17_state_tax=str(data.get("box17_state_tax", "")),
        box18_local_wages=str(data.get("box18_local_wages", "")),
        box19_local_tax=str(data.get("box19_local_tax", "")),
        box20_locality_name=str(data.get("box20_locality_name", "")),
    )


def w2_to_markdown(w2: W2Data) -> str:
    """Convert a :class:`W2Data` instance to structured markdown.

    The output uses headers so the markdown chunker preserves section
    context in each chunk (``strip_headers=False``).
    """
    lines: list[str] = []

    lines.append("# W-2 Wage and Tax Statement")
    lines.append("")

    # Employer info
    lines.append("## Employer Information")
    lines.append(f"- **Employer Name (c):** {w2.employer_name}")
    lines.append(f"- **Employer EIN (b):** {w2.employer_ein}")
    lines.append(f"- **Employer Address:** {w2.employer_address}")
    lines.append(f"- **Control Number (d):** {w2.control_number}")
    lines.append("")

    # Employee info
    lines.append("## Employee Information")
    lines.append(f"- **Employee Name (e/f):** {w2.employee_name}")
    lines.append(f"- **Employee SSN (a):** {w2.employee_ssn}")
    lines.append(f"- **Employee Address:** {w2.employee_address}")
    lines.append("")

    # Compensation
    lines.append("## Compensation")
    lines.append(f"- **Box 1 — Wages, tips, other compensation:** {w2.box1_wages}")
    lines.append(f"- **Box 3 — Social Security wages:** {w2.box3_ss_wages}")
    lines.append(f"- **Box 5 — Medicare wages and tips:** {w2.box5_medicare_wages}")
    lines.append(f"- **Box 7 — Social Security tips:** {w2.box7_ss_tips}")
    lines.append(f"- **Box 8 — Allocated tips:** {w2.box8_allocated_tips}")
    lines.append(f"- **Box 10 — Dependent care benefits:** {w2.box10_dependent_care}")
    lines.append(f"- **Box 11 — Nonqualified plans:** {w2.box11_nonqualified_plans}")
    lines.append("")

    # Taxes withheld
    lines.append("## Taxes Withheld")
    lines.append(f"- **Box 2 — Federal income tax withheld:** {w2.box2_fed_tax_withheld}")
    lines.append(f"- **Box 4 — Social Security tax withheld:** {w2.box4_ss_tax_withheld}")
    lines.append(f"- **Box 6 — Medicare tax withheld:** {w2.box6_medicare_tax_withheld}")
    lines.append("")

    # Box 12
    lines.append("## Box 12 — Codes")
    if w2.box12:
        for entry in w2.box12:
            lines.append(f"- **Code {entry.code}:** {entry.amount}")
    else:
        lines.append("- (none)")
    lines.append("")

    # Box 13
    lines.append("## Box 13 — Checkboxes")
    lines.append(f"- **Statutory employee:** {'Yes' if w2.box13_statutory_employee else 'No'}")
    lines.append(f"- **Retirement plan:** {'Yes' if w2.box13_retirement_plan else 'No'}")
    lines.append(f"- **Third-party sick pay:** {'Yes' if w2.box13_third_party_sick_pay else 'No'}")
    lines.append("")

    # Box 14
    lines.append("## Box 14 — Other")
    lines.append(f"- {w2.box14_other or '(none)'}")
    lines.append("")

    # State / local
    lines.append("## State and Local Taxes")
    lines.append(f"- **Box 15 — State:** {w2.box15_state}")
    lines.append(f"- **Box 15 — Employer state ID:** {w2.box15_employer_state_id}")
    lines.append(f"- **Box 16 — State wages:** {w2.box16_state_wages}")
    lines.append(f"- **Box 17 — State income tax:** {w2.box17_state_tax}")
    lines.append(f"- **Box 18 — Local wages:** {w2.box18_local_wages}")
    lines.append(f"- **Box 19 — Local income tax:** {w2.box19_local_tax}")
    lines.append(f"- **Box 20 — Locality name:** {w2.box20_locality_name}")
    lines.append("")

    return "\n".join(lines)
