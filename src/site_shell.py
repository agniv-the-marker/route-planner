"""Shared page furniture for the interactive and static pages."""


def header(page: str = "text") -> str:
    mode_link = ('<a href="/draw">drawing mode ↗</a>' if page == "text"
                 else '<a href="/">text mode ↗</a>')
    debug_links = ('<button type="button" data-open="debug">debug</button>'
                   '<button type="button" data-open="dev">dev</button>') if page == "text" else ""
    return (
        '<header id="masthead" class="site-header">'
        '<a class="wordmark" href="/"><img src="/drawing-assets/bike-mark.webp" alt="">'
        'route sculptor</a><nav aria-label="Main navigation">'
        f'{mode_link}<a href="/about">about</a>{debug_links}</nav></header>'
    )


def footer() -> str:
    return (
        # The spacer absorbs leftover height so a short page still ends with the footer
        # at the bottom, and keeps a fixed gap once the page is tall enough to scroll.
        '<div class="footer-spacer" aria-hidden="true"></div>'
        '<footer id="credits"><span>© '
        '<a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> · '
        '<a href="/about">terrain &amp; credits</a></span>'
        '<a href="https://agniv.me">agniv.me</a></footer>'
    )


def render_page(template: str, page: str) -> str:
    return template.replace("{{HEADER}}", header(page)).replace("{{FOOTER}}", footer())
