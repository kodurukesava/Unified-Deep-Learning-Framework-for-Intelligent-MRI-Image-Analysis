
# src/report_generator.py

import os
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PDFImage
)


def generate_pdf_report(analysis, image_path=None):
    output = BytesIO()

    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=42,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#19345c"),
        spaceAfter=18
    ))

    story = [
        Paragraph("MRI IMAGE ANALYSIS REPORT", styles["ReportTitle"]),
        Paragraph(
            "AI-assisted MRI image classification summary",
            styles["Normal"]
        ),
        Spacer(1, 18),
        Paragraph("Patient Information", styles["Heading2"])
    ]

    patient_rows = [
        ["Patient ID", escape(str(analysis["patient_code"]))],
        ["Patient Name", escape(str(analysis["patient_name"]))],
        ["Age", str(analysis.get("patient_age") or "Not provided")],
        ["Analysis Date", escape(str(analysis["analysis_date"]))],
        ["Analysis ID", str(analysis["id"])]
    ]

    patient_table = Table(patient_rows, colWidths=[1.7 * inch, 4.4 * inch])
    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eaf0f8")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 8)
    ]))
    story.extend([patient_table, Spacer(1, 18)])

    story.append(Paragraph("AI Analysis Results", styles["Heading2"]))

    result_rows = [
        ["Model output", escape(str(analysis["prediction"]))],
        ["Model confidence", f'{float(analysis["confidence"]):.2f}%']
    ]

    result_table = Table(result_rows, colWidths=[1.7 * inch, 4.4 * inch])
    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eaf0f8")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8)
    ]))
    story.extend([result_table, Spacer(1, 16)])

    probabilities = analysis.get("probabilities", {})
    if probabilities:
        story.append(
            Paragraph("Class Probabilities", styles["Heading2"])
        )

        rows = [["Class", "Probability"]]
        for class_name, value in probabilities.items():
            rows.append([
                escape(str(class_name)),
                f"{float(value):.2f}%"
            ])

        probability_table = Table(
            rows, colWidths=[3.5 * inch, 2.6 * inch], repeatRows=1
        )
        probability_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#19345c")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 7)
        ]))
        story.extend([probability_table, Spacer(1, 18)])

    if image_path and os.path.isfile(image_path):
        try:
            story.append(Paragraph("Uploaded MRI Image", styles["Heading2"]))
            story.append(PDFImage(image_path, width=3.0 * inch,
                                  height=3.0 * inch, kind="proportional"))
            story.append(Spacer(1, 18))
        except Exception:
            story.append(Paragraph(
                "The uploaded image could not be embedded.",
                styles["Normal"]
            ))

    story.extend([
        Paragraph("Important Medical Disclaimer", styles["Heading2"]),
        Paragraph(
            "This report contains an AI-generated classification result. "
            "Model confidence is not the probability that a patient has "
            "a disease. This report is not a medical diagnosis and must "
            "not be used alone to make treatment decisions. A qualified "
            "healthcare professional should review the MRI and interpret "
            "the findings in the appropriate clinical context.",
            styles["BodyText"]
        )
    ])

    doc.build(story)
    output.seek(0)
    return output
