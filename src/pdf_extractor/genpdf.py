"""
MuleSoft → MCP Migration Package — PDF Generator
Requires: pip install reportlab
Run:      python generate_pdf.py
Output:   MuleSoft-MCP-Migration-Package.pdf
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from pathlib import Path
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "Data"
OUTPUT_FILE = OUTPUT_DIR / "MuleSoft-MCP-Migration-Package.pdf"
W, H = A4  # 595.27 x 841.89 points

# ── Brand colors ────────────────────────────────────────────────────────────
DARK    = colors.HexColor("#141413")
LIGHT   = colors.HexColor("#faf9f5")
ORANGE  = colors.HexColor("#d97757")
MUTED   = colors.HexColor("#5f5e5a")
FAINT   = colors.HexColor("#d3d1c7")
CREAM   = colors.HexColor("#f5f3ee")


# ── Document content ─────────────────────────────────────────────────────────
DOCUMENTS = [
    {
        "title": "Executive Summary",
        "subtitle": "MuleSoft to MCP Migration Initiative",
        "sections": [
            ("Overview",
             "This document package outlines the strategic, technical, and operational "
             "plan for migrating the organization's MuleSoft API platform to a Model "
             "Context Protocol (MCP) architecture, enabling AI-native agent integration "
             "across all enterprise systems."),
            ("Objectives",
             "1. Reduce annual MuleSoft licensing costs by transitioning to open-source "
             "MCP infrastructure.\n"
             "2. Enable AI agents to autonomously discover and invoke enterprise tools "
             "without custom integration code.\n"
             "3. Establish a centralized, governed tool registry as the AI integration "
             "layer of record.\n"
             "4. Complete full migration within four phases over 12–18 months."),
            ("Expected Outcomes",
             "All enterprise APIs will be registered as MCP tools consumable by any LLM "
             "or agent framework. The organization will achieve significant cost savings, "
             "accelerated AI adoption, and a compounding architectural advantage as new "
             "AI capabilities are deployed."),
            ("Stakeholders",
             "Executive Sponsor: CTO\n"
             "Project Owner: VP of Engineering\n"
             "Technical Lead: Enterprise Architecture Team\n"
             "Key Consumers: AI/ML Team, Product Engineering, IT Operations"),
        ],
    },
    {
        "title": "Business Case",
        "subtitle": "Converting MuleSoft APIs into MCP Tools & Agents",
        "sections": [
            ("The Strategic Problem",
             "MuleSoft was designed for a world where humans wrote integration logic and "
             "APIs were called by applications. That world is being replaced by one where "
             "AI agents select and orchestrate capabilities autonomously. MuleSoft's rigid "
             "flows and heavyweight runtimes are fundamentally misaligned with AI-native "
             "systems."),
            ("Financial Case",
             "MuleSoft enterprise licensing typically costs hundreds of thousands to "
             "millions of dollars annually. MCP servers are open-source with no per-seat "
             "or per-API-call fees. Developer productivity improves significantly as MCP "
             "tools are written in standard code (Python, TypeScript) by any engineer."),
            ("Technical Case",
             "Each MuleSoft concept maps cleanly to an MCP equivalent:\n"
             "  - API endpoint / flow  →  Tool function\n"
             "  - Anypoint Exchange    →  MCP tool registry\n"
             "  - Orchestration flow   →  AI agent reasoning loop\n"
             "  - DataWeave transform  →  LLM-native data handling\n"
             "  - API gateway          →  MCP server with auth middleware"),
            ("Strategic Case",
             "Organizations that build MCP tool registries now gain a compounding "
             "advantage. Every new AI capability automatically has access to the full "
             "library of enterprise tools, rather than requiring custom integration "
             "plumbing for each new use case."),
        ],
    },
    {
        "title": "Technical Architecture",
        "subtitle": "MCP Server Design & Tool Registration",
        "sections": [
            ("Architecture Overview",
             "The target architecture consists of one or more MCP servers acting as a "
             "centralized tool registry. Each former MuleSoft API becomes a discrete, "
             "versioned tool function. AI agents connect to the MCP server and "
             "autonomously select and invoke tools based on task context."),
            ("MCP Server Components",
             "1. Tool Registry: Central catalog of all tool functions with metadata, "
             "schemas, and descriptions.\n"
             "2. Auth Middleware: OAuth 2.0 / API key authentication and tool-level "
             "permission scoping.\n"
             "3. Logging Layer: Structured logs of all agent tool calls for governance "
             "and audit.\n"
             "4. Version Manager: Semantic versioning to support backward compatibility."),
            ("Tool Function Structure",
             "Each tool is defined with:\n"
             "  - name:         Unique identifier (e.g. crm_get_customer)\n"
             "  - description:  Natural language description for agent discovery\n"
             "  - input_schema: JSON Schema for required and optional parameters\n"
             "  - output_schema: JSON Schema defining the response structure\n"
             "  - auth_scope:   Required permission scope for invocation"),
            ("Security Considerations",
             "MCP servers support OAuth, API key auth, and fine-grained tool-level "
             "permission scoping. All invocations are logged with agent identity, "
             "timestamp, inputs, and output status — providing better visibility than "
             "most MuleSoft deployments."),
        ],
    },
    {
        "title": "Migration Phasing Plan",
        "subtitle": "4-Phase Rollout Over 12–18 Months",
        "sections": [
            ("Phase 1 — Wrap APIs as Tools  (Months 1–3)",
             "Identify the 5–10 highest-value MuleSoft APIs. Wrap each as a discrete MCP "
             "tool function with a clear input/output schema. Run in parallel with "
             "existing MuleSoft flows.\n"
             "Deliverable: Working tool functions for priority APIs, validated against "
             "existing integration tests."),
            ("Phase 2 — Register in MCP Server  (Months 3–6)",
             "Stand up an MCP server (open-source or managed). Register all wrapped tools "
             "with descriptions, schemas, and auth. Validate tool discoverability and "
             "invocation.\n"
             "Deliverable: Live MCP server with full tool registry for Phase 1 APIs."),
            ("Phase 3 — Connect AI Agents  (Months 6–12)",
             "Deploy an AI agent (e.g. Claude) connected to the MCP server. Demonstrate "
             "a high-value use case that previously required months of MuleSoft "
             "development. Use this proof-of-concept to close organizational buy-in.\n"
             "Deliverable: Production AI agent use case with measurable business value."),
            ("Phase 4 — Decommission MuleSoft  (Months 12–18)",
             "Migrate remaining APIs to MCP tools. Validate full parity with existing "
             "integration coverage. Sunset Anypoint Platform and realize licensing "
             "savings.\n"
             "Deliverable: Full MCP tool registry, MuleSoft decommissioned, license "
             "costs eliminated."),
        ],
    },
    {
        "title": "Claude Skills Setup Guide",
        "subtitle": "GitHub, Folder Structure & Python Integration",
        "sections": [
            ("Where to Find Skills",
             "The official Anthropic Skills repository:\n"
             "https://github.com/anthropics/skills\n\n"
             "It contains pre-built skills for creative design, development, enterprise "
             "communication, and document creation. Each skill is a self-contained folder "
             "with a SKILL.md file."),
            ("How to Download",
             "Clone the full repository:\n"
             "  git clone https://github.com/anthropics/skills.git\n\n"
             "Sparse-checkout a single skill:\n"
             "  git clone --no-checkout https://github.com/anthropics/skills.git\n"
             "  git sparse-checkout init --cone\n"
             "  git sparse-checkout set skills/template-skill\n"
             "  git checkout main"),
            ("Folder Structure in a Python Project",
             "Place skills inside .claude/skills/ at the project root:\n\n"
             "  my-python-project/\n"
             "  ├── .claude/\n"
             "  │   └── skills/\n"
             "  │       └── my-skill/\n"
             "  │           ├── SKILL.md\n"
             "  │           ├── references/\n"
             "  │           └── scripts/\n"
             "  ├── src/\n"
             "  └── requirements.txt"),
            ("Python Integration",
             "Install the SDK:\n"
             "  pip install anthropic\n\n"
             "Upload and invoke a skill:\n"
             "  client = anthropic.Anthropic()\n"
             "  skill  = client.beta.skills.upload(name='my-skill', content=skill_md)\n"
             "  response = client.beta.messages.create(\n"
             "      model='claude-sonnet-4-6',\n"
             "      skills=[{'type': 'uploaded_skill', 'skill_id': skill.id}],\n"
             "      messages=[{'role': 'user', 'content': 'Your prompt here'}]\n"
             "  )"),
        ],
    },
]


# ── Low-level helpers ────────────────────────────────────────────────────────
def draw_cover(c: canvas.Canvas):
    """Draw a full dark cover page."""
    c.setFillColor(DARK)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # Main title
    c.setFillColor(LIGHT)
    c.setFont("Helvetica-Bold", 34)
    c.drawString(20*mm, H - 80*mm, "MuleSoft \u2192 MCP")
    c.setFont("Helvetica", 34)
    c.drawString(20*mm, H - 94*mm, "Migration Package")

    # Subtitle
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 12)
    c.drawString(20*mm, H - 112*mm, "A collection of 5 documents")
    c.drawString(20*mm, H - 120*mm, "March 2026")

    # Contents heading
    c.setFillColor(ORANGE)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(20*mm, H - 148*mm, "CONTENTS")

    for i, doc in enumerate(DOCUMENTS):
        y = H - (160 + i * 16) * mm
        c.setFillColor(colors.HexColor("#c8c4b6"))
        c.setFont("Helvetica", 10)
        c.drawString(20*mm, y, f"{i+1}.   {doc['title']}")
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 9)
        c.drawString(26*mm, y - 6*mm, doc["subtitle"])

    # Footer
    c.setFillColor(colors.HexColor("#504e48"))
    c.setFont("Helvetica", 8)
    c.drawString(20*mm, 14*mm, "Confidential \u2014 Internal Use Only")
    c.drawRightString(W - 20*mm, 14*mm, "Page 1")


def draw_doc_header(c: canvas.Canvas, doc_num: int, total: int, title: str, page_num: int):
    """Draw the orange header bar on a document page."""
    c.setFillColor(ORANGE)
    c.rect(0, H - 18*mm, W, 18*mm, fill=1, stroke=0)
    c.setFillColor(LIGHT)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(20*mm, H - 11*mm, f"DOCUMENT {doc_num} OF {total}")
    c.setFont("Helvetica", 9)
    c.drawCentredString(W / 2, H - 11*mm, title.upper())
    c.drawRightString(W - 20*mm, H - 11*mm, f"Page {page_num}")


def draw_doc_footer(c: canvas.Canvas, title: str):
    c.setStrokeColor(FAINT)
    c.setLineWidth(0.5)
    c.line(20*mm, 16*mm, W - 20*mm, 16*mm)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8)
    c.drawString(20*mm, 10*mm, "Confidential \u2014 Internal Use Only")
    c.drawCentredString(W / 2, 10*mm, title)


def wrap_text(text: str, font: str, size: float, max_width: float) -> list:
    """Split text into lines that fit within max_width points."""
    from reportlab.pdfbase.pdfmetrics import stringWidth
    words = text.split(" ")
    lines, current = [], ""
    for word in words:
        test = (current + " " + word).strip()
        if stringWidth(test, font, size) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_section(c: canvas.Canvas, heading: str, body: str,
                 y: float, margin: float, content_w: float) -> float:
    """Draw one section heading + body. Returns updated y position."""
    LINE_H   = 5.5 * mm
    HEAD_H   = 10  * mm
    PAD      = 20  * mm   # bottom margin before new page

    # Section heading background
    if y - HEAD_H < PAD:
        c.showPage()
        y = H - 30*mm
    c.setFillColor(CREAM)
    c.rect(margin, y - HEAD_H + 3*mm, content_w, HEAD_H, fill=1, stroke=0)
    c.setFillColor(DARK)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin + 3*mm, y - 4*mm, heading)
    y -= HEAD_H + 2*mm

    # Body text — handle explicit newlines
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor("#323230"))
    for raw_line in body.split("\n"):
        wrapped = wrap_text(raw_line, "Helvetica", 10, content_w) if raw_line.strip() else [""]
        for line in wrapped:
            if y < PAD:
                c.showPage()
                y = H - 30*mm
            c.drawString(margin, y, line)
            y -= LINE_H

    return y - 4*mm


# ── Main builder ─────────────────────────────────────────────────────────────
def build_pdf():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT_FILE), pagesize=A4)    
    c.setTitle("MuleSoft → MCP Migration Package")
    c.setAuthor("Enterprise Architecture Team")
    c.setSubject("MuleSoft to MCP Migration")

    # Cover
    draw_cover(c)
    c.showPage()

    page_num = 2
    margin     = 20 * mm
    content_w  = W - 2 * margin
    total_docs = len(DOCUMENTS)

    for di, document in enumerate(DOCUMENTS):
        title    = document["title"]
        subtitle = document["subtitle"]

        # Header bar
        draw_doc_header(c, di + 1, total_docs, title, page_num)

        # Document title block
        y = H - 34*mm
        c.setFillColor(DARK)
        c.setFont("Helvetica-Bold", 22)
        c.drawString(margin, y, title)
        y -= 8*mm

        c.setFillColor(MUTED)
        c.setFont("Helvetica", 11)
        c.drawString(margin, y, subtitle)
        y -= 6*mm

        # Orange rule
        c.setStrokeColor(ORANGE)
        c.setLineWidth(1)
        c.line(margin, y, W - margin, y)
        y -= 10*mm

        # Sections
        for heading, body in document["sections"]:
            y = draw_section(c, heading, body, y, margin, content_w)

        draw_doc_footer(c, title)
        c.showPage()
        page_num += 1

    c.save()
    print(f"PDF saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    build_pdf()