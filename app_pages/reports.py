
import json
import math
import re
from datetime import datetime, timezone
from io import BytesIO
from xml.sax.saxutils import escape

import streamlit as st

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    Flowable,
    KeepTogether,
    PageBreak,
)

from components.header import render_header
from services.assessment_service import get_user_assessments
from services.health_guidance import build_health_guidance


PROJECT_NAME = (
    "AI-Based Early Health Risk Detection "
    "with Explainable AI (XAI)"
)

NAVY = colors.HexColor("#183B56")
BLUE = colors.HexColor("#2878A8")
LIGHT_BLUE = colors.HexColor("#EEF3F8")
LIGHT_GREY = colors.HexColor("#E2E8F0")
GREY = colors.HexColor("#64748B")
POSITIVE = colors.HexColor("#D97757")
NEGATIVE = colors.HexColor("#3984B8")


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def format_date(value):

    if value is None:
        return "Date unavailable"

    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )
        except ValueError:
            return value

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value.astimezone().strftime(
        "%d %B %Y, %I:%M %p"
    )


def safe_text(value):

    if value is None:
        return "Not available"

    return escape(str(value))


def normalize_dict(value):

    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)

            if isinstance(parsed, dict):
                return parsed

        except (json.JSONDecodeError, TypeError):
            pass

    return {}


def get_class_probabilities(record, explanation):

    saved = explanation.get("class_probabilities") or {}

    try:
        class_1 = float(
            saved.get("Class 1", record.get("model_score"))
        )

        if not math.isfinite(class_1):
            return None

        if not 0 <= class_1 <= 1:
            return None

        class_0 = float(
            saved.get("Class 0", 1.0 - class_1)
        )

        if not math.isfinite(class_0):
            return None

        if not 0 <= class_0 <= 1:
            return None

        if abs(class_0 + class_1 - 1.0) > 1e-5:
            return None

        return {
            "Class 0": class_0,
            "Class 1": class_1,
        }

    except (TypeError, ValueError):
        return None


def get_class_labels(explanation, assessment_type):

    labels = explanation.get("class_labels") or {}

    if assessment_type == "heart":
        default_0 = "No Heart Disease"
        default_1 = "Heart Disease"
    else:
        default_0 = "No Diabetes"
        default_1 = "Diabetes"

    label_0 = labels.get(
        0,
        labels.get("0", default_0)
    )

    label_1 = labels.get(
        1,
        labels.get("1", default_1)
    )

    return str(label_0), str(label_1)


def get_shap_features(explanation):

    features = (
        explanation.get("all_features")
        or explanation.get("top_features")
        or []
    )

    valid = []

    for item in features:

        if not isinstance(item, dict):
            continue

        try:
            contribution = float(item.get("contribution"))

            if not math.isfinite(contribution):
                continue

            valid.append({
                "feature": str(
                    item.get("feature", "Unknown")
                ),
                "contribution": contribution,
                "direction": str(
                    item.get("direction", "")
                ),
            })

        except (TypeError, ValueError):
            continue

    return sorted(
        valid,
        key=lambda item: abs(item["contribution"]),
        reverse=True,
    )


# --------------------------------------------------
# HEALTH GUIDANCE TEXT CLEANER
# --------------------------------------------------

def clean_guidance_text(value):

    if value is None:
        return ""

    text = str(value).strip()

    if not text:
        return ""

    blocked_phrases = (
        "does not rule out",
        "not proof of medical causation",
        "not a medical diagnosis",
        "not a substitute for professional",
        "not a substitute for medical",
        "not proof of",
        "not clinically validated",
        "not clinical evidence",
        "model cannot diagnose",
        "model is not a diagnosis",
        "educational and research use only",
        "for educational purposes only",
        "disclaimer",
        "warning:",
        "caution:",
        "model output should not",
        "prediction should not be used",
        "shap explains model behaviour",
        "shap explains model behavior",
    )

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.replace("\n", " ")
    )

    kept = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        if any(
            phrase in sentence.lower()
            for phrase in blocked_phrases
        ):
            continue

        kept.append(sentence)

    return " ".join(kept).strip()


def format_number(value, decimals=6):

    try:
        number = float(value)

        if not math.isfinite(number):
            return "Not available"

        return f"{number:.{decimals}f}"

    except (TypeError, ValueError):
        return "Not available"


# --------------------------------------------------
# PREDICTION GRAPH
# --------------------------------------------------

class PredictionProbabilityChart(Flowable):

    def __init__(
        self,
        class_0,
        class_1,
        label_0,
        label_1,
        width=160 * mm
    ):

        super().__init__()

        self.class_0 = float(class_0)
        self.class_1 = float(class_1)
        self.label_0 = str(label_0)
        self.label_1 = str(label_1)

        self.width = width
        self.height = 95

    def wrap(self, available_width, available_height):

        self.width = min(
            self.width,
            available_width
        )

        return self.width, self.height

    def draw(self):

        canvas = self.canv

        left = 48 * mm
        right = self.width - 22 * mm

        chart_width = right - left
        bar_height = 15

        rows = [
            (self.label_1, self.class_1, POSITIVE, 55),
            (self.label_0, self.class_0, NEGATIVE, 25),
        ]

        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(NAVY)

        canvas.drawString(
            0,
            82,
            "Prediction Probability Comparison"
        )

        for label, value, bar_color, y in rows:

            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(
                colors.HexColor("#334155")
            )

            canvas.drawRightString(
                left - 6,
                y + 4,
                label[:25]
            )

            canvas.setFillColor(LIGHT_GREY)

            canvas.roundRect(
                left,
                y,
                chart_width,
                bar_height,
                3,
                stroke=0,
                fill=1,
            )

            canvas.setFillColor(bar_color)

            canvas.roundRect(
                left,
                y,
                max(chart_width * value, 0.5),
                bar_height,
                3,
                stroke=0,
                fill=1,
            )

            canvas.setFillColor(NAVY)
            canvas.setFont("Helvetica-Bold", 8)

            value_x = min(
                left + chart_width * value + 5,
                right - 1
            )

            canvas.drawString(
                value_x,
                y + 4,
                f"{value * 100:.2f}%"
            )

        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GREY)

        canvas.drawString(left, 8, "0%")
        canvas.drawRightString(right, 8, "100%")


# --------------------------------------------------
# SHAP GRAPH
# --------------------------------------------------

class SHAPContributionChart(Flowable):

    def __init__(self, features, width=160 * mm):

        super().__init__()

        self.features = features
        self.width = width
        self.row_height = 15
        self.top_space = 24
        self.bottom_space = 22

        self.height = (
            self.top_space
            + len(features) * self.row_height
            + self.bottom_space
        )

    def wrap(self, available_width, available_height):

        self.width = min(
            self.width,
            available_width
        )

        return self.width, self.height

    def draw(self):

        canvas = self.canv

        label_width = 47 * mm
        right_padding = 8 * mm

        chart_left = label_width
        chart_right = self.width - right_padding

        chart_width = chart_right - chart_left
        center_x = chart_left + chart_width / 2

        max_value = max(
            abs(item["contribution"])
            for item in self.features
        )

        if max_value == 0:
            max_value = 1.0

        usable_half_width = chart_width / 2 - 5

        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(NAVY)

        canvas.drawString(
            0,
            self.height - 12,
            "SHAP Feature Contributions"
        )

        canvas.setStrokeColor(GREY)
        canvas.setLineWidth(0.8)

        canvas.line(
            center_x,
            self.bottom_space - 2,
            center_x,
            self.height - self.top_space + 4
        )

        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GREY)

        canvas.drawString(
            chart_left,
            5,
            "Lower model output"
        )

        canvas.drawRightString(
            chart_right,
            5,
            "Higher model output"
        )

        for index, item in enumerate(self.features):

            y = (
                self.height
                - self.top_space
                - (index + 1) * self.row_height
            )

            contribution = item["contribution"]
            label = item["feature"]

            if len(label) > 22:
                label = label[:20] + "..."

            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(
                colors.HexColor("#334155")
            )

            canvas.drawString(
                0,
                y + 3,
                label
            )

            bar_width = (
                abs(contribution)
                / max_value
                * usable_half_width
            )

            if contribution >= 0:
                bar_color = POSITIVE
                bar_x = center_x
            else:
                bar_color = NEGATIVE
                bar_x = center_x - bar_width

            canvas.setFillColor(bar_color)

            canvas.roundRect(
                bar_x,
                y,
                max(bar_width, 0.5),
                9,
                2,
                stroke=0,
                fill=1,
            )

            canvas.setFillColor(GREY)
            canvas.setFont("Helvetica", 6.5)

            if contribution >= 0:

                canvas.drawString(
                    bar_x + bar_width + 3,
                    y + 2,
                    f"{contribution:+.4f}"
                )

            else:

                canvas.drawRightString(
                    bar_x - 3,
                    y + 2,
                    f"{contribution:+.4f}"
                )


# --------------------------------------------------
# PAGE FOOTER
# --------------------------------------------------

def add_page_footer(canvas, doc):

    canvas.saveState()

    page_width, _ = A4

    canvas.setStrokeColor(LIGHT_GREY)

    canvas.line(
        18 * mm,
        13 * mm,
        page_width - 18 * mm,
        13 * mm
    )

    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(GREY)

    canvas.drawString(
        18 * mm,
        8 * mm,
        "AI Health XAI"
    )

    canvas.drawRightString(
        page_width - 18 * mm,
        8 * mm,
        f"Page {doc.page}"
    )

    canvas.restoreState()


# --------------------------------------------------
# PDF GENERATION
# --------------------------------------------------

def generate_pdf(record, user):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=15 * mm,
        bottomMargin=19 * mm,
        title="AI Health XAI Assessment Report",
        author="AI Health XAI",
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="BrandTitle",
            parent=styles["Title"],
            fontSize=19,
            leading=23,
            alignment=1,
            textColor=NAVY,
            spaceAfter=4,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ProjectSubtitle",
            parent=styles["Normal"],
            fontSize=8,
            leading=11,
            alignment=1,
            textColor=GREY,
            spaceAfter=8,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontSize=11,
            leading=14,
            textColor=NAVY,
            spaceBefore=9,
            spaceAfter=5,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SmallText",
            parent=styles["Normal"],
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#334155"),
        )
    )

    styles.add(
        ParagraphStyle(
            name="GuidanceHeading",
            parent=styles["Heading3"],
            fontSize=9,
            leading=11,
            textColor=BLUE,
            spaceBefore=6,
            spaceAfter=3,
        )
    )

    story = []

    assessment_type = record.get(
        "assessment_type",
        "unknown"
    )

    assessment_name = (
        "Heart Assessment"
        if assessment_type == "heart"
        else "Diabetes Assessment"
        if assessment_type == "diabetes"
        else "Assessment"
    )

    input_data = normalize_dict(
        record.get("input_data")
    )

    explanation = normalize_dict(
        record.get("explanation")
    )

    features = get_shap_features(explanation)

    prediction = record.get("prediction")
    score = record.get("model_score")
    model_key = record.get("model_key", "unknown")

    probabilities = get_class_probabilities(
        record,
        explanation
    )

    label_0, label_1 = get_class_labels(
        explanation,
        assessment_type
    )

    # ----------------------------------------------
    # HEADER
    # ----------------------------------------------

    story.append(
        Paragraph(
            "AI HEALTH XAI",
            styles["BrandTitle"]
        )
    )

    story.append(
        Paragraph(
            safe_text(PROJECT_NAME),
            styles["ProjectSubtitle"]
        )
    )

    story.append(
        HRFlowable(
            width="100%",
            thickness=1.2,
            color=NAVY,
        )
    )

    story.append(Spacer(1, 5))

    story.append(
        Paragraph(
            safe_text(assessment_name) + " Report",
            styles["Heading1"]
        )
    )

    # ----------------------------------------------
    # ASSESSMENT INFORMATION
    # ----------------------------------------------

    story.append(
        Paragraph(
            "Assessment Information",
            styles["SectionHeading"]
        )
    )

    info_data = [
        ["Assessment ID", str(record.get("id", "N/A"))],
        ["User", safe_text(user.get("full_name", "User"))],
        ["Email", safe_text(user.get("email", "N/A"))],
        ["Assessment Type", assessment_name],
        ["Model", safe_text(model_key)],
        [
            "Date",
            safe_text(format_date(record.get("created_at")))
        ],
    ]

    info_table = Table(
        info_data,
        colWidths=[45 * mm, 115 * mm],
    )

    info_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
            ("GRID", (0, 0), (-1, -1), 0.4, LIGHT_GREY),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )

    story.append(info_table)

    # ----------------------------------------------
    # PREDICTION SUMMARY
    # ----------------------------------------------

    story.append(
        Paragraph(
            "Prediction Summary",
            styles["SectionHeading"]
        )
    )

    if prediction in (0, 1) and probabilities is not None:

        predicted_label = (
            label_1 if int(prediction) == 1 else label_0
        )

        predicted_probability = probabilities[
            "Class 1" if int(prediction) == 1 else "Class 0"
        ]

        other_label = (
            label_0 if int(prediction) == 1 else label_1
        )

        other_probability = probabilities[
            "Class 0" if int(prediction) == 1 else "Class 1"
        ]

        result_rows = [
            ["Prediction", "Probability"],
            [
                safe_text(predicted_label),
                f"{predicted_probability * 100:.2f}%"
            ],
            [
                safe_text(other_label),
                f"{other_probability * 100:.2f}%"
            ],
        ]

    else:

        predicted_label = (
            f"Class {prediction}"
            if prediction in (0, 1)
            else "Not available"
        )

        try:
            score_text = (
                f"{float(score) * 100:.2f}%"
                if score is not None
                else "Not available"
            )
        except (TypeError, ValueError):
            score_text = "Not available"

        result_rows = [
            ["Prediction", "Class 1 Model Score"],
            [predicted_label, score_text],
        ]

    result_table = Table(
        result_rows,
        colWidths=[80 * mm, 80 * mm],
    )

    result_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 1), (-1, -1), LIGHT_BLUE),
            ("GRID", (0, 0), (-1, -1), 0.4, LIGHT_GREY),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("PADDING", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )

    story.append(result_table)

    # ----------------------------------------------
    # PREDICTION GRAPH
    # ----------------------------------------------

    story.append(
        Paragraph(
            "Disease vs No Disease",
            styles["SectionHeading"]
        )
    )

    if probabilities is not None:

        story.append(
            PredictionProbabilityChart(
                class_0=probabilities["Class 0"],
                class_1=probabilities["Class 1"],
                label_0=label_0,
                label_1=label_1,
            )
        )

    # ----------------------------------------------
    # INPUT DATA
    # ----------------------------------------------

    story.append(
        Paragraph(
            "Assessment Input Data",
            styles["SectionHeading"]
        )
    )

    if input_data:

        input_rows = [
            ["Feature", "Entered Value"]
        ]

        for key, value in input_data.items():

            input_rows.append([
                Paragraph(
                    safe_text(key),
                    styles["SmallText"]
                ),
                Paragraph(
                    safe_text(value),
                    styles["SmallText"]
                ),
            ])

        input_table = Table(
            input_rows,
            colWidths=[80 * mm, 80 * mm],
            repeatRows=1,
        )

        input_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, LIGHT_GREY),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("PADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, LIGHT_BLUE]
                ),
            ])
        )

        story.append(input_table)

    else:

        story.append(
            Paragraph(
                "Input data unavailable.",
                styles["SmallText"]
            )
        )

    # ----------------------------------------------
    # SHAP ANALYSIS
    # ----------------------------------------------

    if features:

        story.append(PageBreak())

        story.append(
            Paragraph(
                "Explainable AI (XAI) Analysis",
                styles["SectionHeading"]
            )
        )

        story.append(
            Paragraph(
                "SHAP Feature Contribution Graph",
                styles["Heading3"]
            )
        )

        story.append(Spacer(1, 5))

        story.append(
            SHAPContributionChart(features)
        )

        story.append(Spacer(1, 5))

        story.append(
            Paragraph(
                "Feature Contribution Details",
                styles["Heading3"]
            )
        )

        xai_rows = [
            ["Feature", "Contribution", "Direction"]
        ]

        for item in features:

            xai_rows.append([
                Paragraph(
                    safe_text(item["feature"]),
                    styles["SmallText"]
                ),
                Paragraph(
                    format_number(
                        item["contribution"],
                        6
                    ),
                    styles["SmallText"]
                ),
                Paragraph(
                    safe_text(item["direction"]),
                    styles["SmallText"]
                ),
            ])

        xai_table = Table(
            xai_rows,
            colWidths=[48 * mm, 43 * mm, 69 * mm],
            repeatRows=1,
        )

        xai_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, LIGHT_GREY),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("PADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, LIGHT_BLUE]
                ),
            ])
        )

        story.append(xai_table)



    # ----------------------------------------------
    # HEALTH GUIDANCE
    # ----------------------------------------------

    if prediction in (0, 1) and assessment_type in (
        "heart",
        "diabetes"
    ):

        guidance = build_health_guidance(
            assessment_type=assessment_type,
            model_key=model_key,
            prediction=int(prediction),
            model_score=float(score or 0),
            input_data=input_data,
            xai_result=explanation,
        )

        story.append(PageBreak())

        story.append(
            Paragraph(
                "Health Guidance & Next Steps",
                styles["SectionHeading"]
            )
        )

        # Clean summary
        summary = clean_guidance_text(
            guidance.get("summary", "")
        )

        if summary:

            story.append(
                Paragraph(
                    "What This Result Means",
                    styles["GuidanceHeading"]
                )
            )

            story.append(
                Paragraph(
                    safe_text(summary),
                    styles["SmallText"]
                )
            )

        # Clean section headings and recommendations
        for section in guidance.get("sections", []):

            title = clean_guidance_text(
                section.get("title", "")
            )

            clean_items = []

            for item in section.get("items", []):

                cleaned_item = clean_guidance_text(item)

                if cleaned_item:
                    clean_items.append(cleaned_item)

            if not clean_items:
                continue

            section_content = []

            if title:

                section_content.append(
                    Paragraph(
                        safe_text(title),
                        styles["GuidanceHeading"]
                    )
                )

            for item in clean_items:

                section_content.append(
                    Paragraph(
                        "&#8226; " + safe_text(item),
                        styles["SmallText"]
                    )
                )

                section_content.append(
                    Spacer(1, 3)
                )

            if section_content:
                story.append(
                    KeepTogether(section_content)
                )

        # Clean separate alerts
        clean_alerts = []

        for alert in guidance.get("alerts", []):

            cleaned_alert = clean_guidance_text(alert)

            if cleaned_alert:
                clean_alerts.append(cleaned_alert)

        if clean_alerts:

            story.append(
                Paragraph(
                    "Additional Recommendations",
                    styles["GuidanceHeading"]
                )
            )

            for alert in clean_alerts:

                story.append(
                    Paragraph(
                        "&#8226; " + safe_text(alert),
                        styles["SmallText"]
                    )
                )

                story.append(Spacer(1, 3))

    # ----------------------------------------------
    # BUILD PDF
    # ----------------------------------------------

    doc.build(
        story,
        onFirstPage=add_page_footer,
        onLaterPages=add_page_footer,
    )

    buffer.seek(0)

    return buffer.getvalue()


# --------------------------------------------------
# REPORTS PAGE
# --------------------------------------------------

def show_reports():

    render_header("Reports")

    st.subheader("Assessment Reports")

    st.caption(
        "Generate and download detailed PDF reports "
        "from your saved assessments."
    )

    user = st.session_state.get("user")

    if not user or not user.get("id"):
        st.error("Please log in to access your reports.")
        return

    try:

        with st.spinner("Loading saved assessments..."):

            records = get_user_assessments(
                user_id=int(user["id"]),
                limit=100,
            )

    except Exception:
        st.error("Unable to load saved assessments.")
        return

    if records is None:
        st.error("Assessment records could not be retrieved.")
        return

    total = len(records)

    heart_count = sum(
        1 for item in records
        if item.get("assessment_type") == "heart"
    )

    diabetes_count = sum(
        1 for item in records
        if item.get("assessment_type") == "diabetes"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Available Reports", total)
    col2.metric("Heart Reports", heart_count)
    col3.metric("Diabetes Reports", diabetes_count)

    st.divider()

    if not records:

        st.info(
            "No saved assessments found. Complete and save an "
            "assessment before generating a PDF report."
        )
        return

    st.markdown("### Generate Individual Report")

    filter_option = st.selectbox(
        "Select report type",
        ["All Assessments", "Heart", "Diabetes"],
        key="reports_filter",
    )

    type_map = {
        "All Assessments": None,
        "Heart": "heart",
        "Diabetes": "diabetes",
    }

    selected_type = type_map[filter_option]

    filtered_records = [
        item for item in records
        if (
            selected_type is None
            or item.get("assessment_type") == selected_type
        )
    ]

    filtered_records.sort(
        key=lambda item: str(
            item.get("created_at") or ""
        ),
        reverse=True,
    )

    if not filtered_records:

        st.info("No records available for this report type.")
        return

    record_options = {}

    for item in filtered_records:

        record_type = item.get(
            "assessment_type",
            "unknown"
        ).title()

        label = (
            f"ID {item.get('id')} | {record_type} | "
            f"{item.get('model_key')} | "
            f"{format_date(item.get('created_at'))}"
        )

        record_options[label] = item

    selected_label = st.selectbox(
        "Choose an assessment record",
        list(record_options.keys()),
        key="selected_report_record",
    )

    selected_record = record_options[selected_label]

    st.divider()

    st.markdown("### Selected Assessment")

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "**Assessment:**",
            selected_record.get(
                "assessment_type", ""
            ).title()
        )

        st.write(
            "**Model:**",
            selected_record.get("model_key")
        )

        st.write(
            "**Date:**",
            format_date(
                selected_record.get("created_at")
            )
        )

    with col2:

        explanation = normalize_dict(
            selected_record.get("explanation")
        )

        prediction = selected_record.get("prediction")

        label_0, label_1 = get_class_labels(
            explanation,
            selected_record.get("assessment_type")
        )

        predicted_label = (
            label_1 if prediction == 1 else label_0
        ) if prediction in (0, 1) else "Not available"

        st.write(
            "**Prediction:**",
            predicted_label
        )

        probabilities = get_class_probabilities(
            selected_record,
            explanation
        )

        if probabilities is not None and prediction in (0, 1):

            predicted_probability = probabilities[
                "Class 1" if prediction == 1 else "Class 0"
            ]

            st.write(
                "**Predicted Class Probability:**",
                f"{predicted_probability * 100:.2f}%"
            )

        else:

            st.write(
                "**Predicted Class Probability:**",
                "Not available"
            )

    # ----------------------------------------------
    # GENERATE PDF
    # ----------------------------------------------

    if st.button(
        "Generate PDF Report",
        type="primary",
        use_container_width=True,
    ):

        try:

            with st.spinner("Generating PDF report..."):

                pdf_bytes = generate_pdf(
                    selected_record,
                    user,
                )

            st.session_state["generated_report"] = {
                "bytes": pdf_bytes,
                "record_id": selected_record.get("id"),
                "filename": (
                    f"AI_Health_XAI_Report_"
                    f"{selected_record.get('id')}.pdf"
                ),
            }

            st.success("PDF report generated successfully.")

        except Exception as exc:

            st.error(f"PDF generation failed: {exc}")

    # ----------------------------------------------
    # DOWNLOAD
    # ----------------------------------------------

    generated = st.session_state.get(
        "generated_report"
    )

    if (
        generated
        and generated.get("record_id")
        == selected_record.get("id")
    ):

        st.download_button(
            label="Download PDF Report",
            data=generated["bytes"],
            file_name=generated["filename"],
            mime="application/pdf",
            use_container_width=True,
        )

    st.divider()


show_reports()
