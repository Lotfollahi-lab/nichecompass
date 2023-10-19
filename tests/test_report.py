import pytest
import nichecompass.report


def test_report_item_text_render():
    report_item = nichecompass.report.ReportItemText(content="hello")
    assert report_item.render() == "<div>\n    <p>hello</p>\n</div>"


def test_report_section_render():
    report_item = nichecompass.report.ReportItemText(content="hello")
    report_section = nichecompass.report.ReportSection(title="title", description="description")
    report_section.add_item(report_item)
    assert report_section.render() == ("<div>\n    <h1>title</h1>\n    <p>description</p>\n    \n        <div>\n    "
                                       "<p>hello</p>\n</div>\n    \n</div>")


def test_report_render():
    report_item = nichecompass.report.ReportItemText(content="hello")
    report_section = nichecompass.report.ReportSection(title="title", description="description")
    report_section.add_item(report_item)
    report = nichecompass.report.Report()
    report.add_section(report_section)
    assert report.render() == ("<html>\n<head>\n    <title>NicheCompass run report</title>\n</head>\n<body>\n    \n    "
                               "    <div>\n    <h1>title</h1>\n    <p>description</p>\n    \n        <div>\n    "
                               "<p>hello</p>\n</div>\n    \n</div>\n    </body>\n</html>")
