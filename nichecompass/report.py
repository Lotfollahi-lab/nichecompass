import uuid
from abc import ABC, abstractmethod
from jinja2 import Environment, PackageLoader, select_autoescape
import matplotlib.pyplot as plt
from io import BytesIO
import base64

env = Environment(
    loader=PackageLoader("nichecompass"),
    autoescape=select_autoescape()
)


class Report:
    """Object that stores the report from a NicheCompass run"""

    def __init__(self):
        self.id = uuid.uuid4()
        self.title = "NicheCompass run report"
        self.sections = []

    def add_section(self, report_section):
        self.sections.append(report_section)

    def render(self, jinja_env=env):
        template = jinja_env.get_template("report.html")
        rendered_sections = [x.render() for x in self.sections]
        rendered_report = template.render(
            report_title=self.title,
            rendered_sections=rendered_sections)
        return rendered_report


class ReportSection:
    """Object that stores a section of a report"""

    def __init__(self, title, description):
        self.title = title
        self.description = description
        self.report_items = []

    def add_item(self, report_item):
        self.report_items.append(report_item)

    def render(self, jinja_env=env):
        template = jinja_env.get_template("section.html")
        rendered_items = [x.render() for x in self.report_items]
        rendered_section = template.render(
            section_title=self.title,
            section_description=self.description,
            rendered_items=rendered_items
        )
        return rendered_section


class ReportItem(ABC):
    """Object that stores an item of a report"""

    @abstractmethod
    def render(self):
        pass


class ReportItemText(ReportItem):
    """A simple text report item"""

    def __init__(self, content):
        self.content = content

    def render(self, jinja_env=env):
        template = jinja_env.get_template("item_text.html")
        rendered_item = template.render(
            item_content=self.content
        )
        return rendered_item


class ReportItemImage(ReportItem):
    """A simple image report item for a matplotlib figure"""

    def __init__(self, fig, alt, caption):
        self.fig = fig
        self.alt = alt
        self.caption = caption

    def render(self, jinja_env=env):
        image_bytes = BytesIO()
        self.fig.savefig(image_bytes, format='png')
        base64_encoded_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        template = jinja_env.get_template("item_image.html")
        rendered_item = template.render(
            img_src=base64_encoded_image,
            img_alt=self.alt,
            caption=self.caption
        )
        return rendered_item

