"""Typeset the single Markdown source as the version 1.0.0 Russian note."""
from __future__ import annotations

import html
from pathlib import Path
import re

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                               Table, TableStyle, Image, KeepTogether)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = ROOT / 'output/pdf/local-recovery-note-ru.pdf'
FONT_DIR = Path('C:/Windows/Fonts')
for name, filename in (('Arial', 'arial.ttf'), ('Arial-Bold', 'arialbd.ttf'),
                       ('Arial-Italic', 'ariali.ttf'), ('Arial-BoldItalic', 'arialbi.ttf')):
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / filename)))
pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='Arial-Bold',
                              italic='Arial-Italic', boldItalic='Arial-BoldItalic')
INK = colors.HexColor('#243638')
ACCENT = colors.HexColor('#32685B')
MUTED = colors.HexColor('#526469')
RULE = colors.HexColor('#CDD7D3')
WASH = colors.HexColor('#EDF2EF')
WIDTH, HEIGHT = A4
LEFT = RIGHT = 48
BODY_WIDTH = WIDTH - LEFT - RIGHT

STYLES = {
    'cover-title': ParagraphStyle('cover-title', fontName='Arial-Bold', fontSize=31, leading=37,
                                  textColor=INK, spaceAfter=23),
    'cover-subtitle': ParagraphStyle('cover-subtitle', fontName='Arial', fontSize=18, leading=26,
                                     textColor=ACCENT, spaceAfter=48),
    'cover-meta': ParagraphStyle('cover-meta', fontName='Arial', fontSize=11, leading=16,
                                 textColor=MUTED, spaceAfter=27),
    'cover-author': ParagraphStyle('cover-author', fontName='Arial-Bold', fontSize=14, leading=21,
                                   textColor=INK, spaceAfter=14),
    'cover-note': ParagraphStyle('cover-note', fontName='Arial', fontSize=10.5, leading=16,
                                 textColor=MUTED, spaceAfter=10),
    'body': ParagraphStyle('body', fontName='Arial', fontSize=10.2, leading=14.1,
                           textColor=INK, spaceAfter=8, allowWidows=0, allowOrphans=0),
    'title': ParagraphStyle('title', fontName='Arial-Bold', fontSize=25, leading=29,
                            textColor=INK, spaceAfter=14),
    'h2': ParagraphStyle('h2', fontName='Arial-Bold', fontSize=17, leading=21,
                         textColor=INK, spaceBefore=2, spaceAfter=12, keepWithNext=True),
    'subtitle': ParagraphStyle('subtitle', fontName='Arial', fontSize=14, leading=19,
                               textColor=ACCENT, spaceAfter=14, keepWithNext=True),
    'h3': ParagraphStyle('h3', fontName='Arial-Bold', fontSize=11.5, leading=15,
                         textColor=ACCENT, spaceBefore=9, spaceAfter=7, keepWithNext=True),
    'caption': ParagraphStyle('caption', fontName='Arial', fontSize=8.7, leading=11.6,
                              textColor=MUTED, spaceAfter=9),
    'cell': ParagraphStyle('cell', fontName='Arial', fontSize=9.1, leading=12,
                           textColor=INK),
    'headcell': ParagraphStyle('headcell', fontName='Arial-Bold', fontSize=8.8,
                               leading=11.3, textColor=INK),
    'reference': ParagraphStyle('reference', fontName='Arial', fontSize=8.4, leading=11.2,
                                textColor=MUTED, spaceAfter=5),
}


def inline(text):
    text = html.escape(text, quote=False)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)',
                  lambda m: f'<link href="{m[2]}" color="#32685B">{m[1]}</link>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`([^`]+)`', r'<font size="8.5">\1</font>', text)
    return text


def paragraph(text, style='body'):
    return Paragraph(inline(text), STYLES[style])


def table(lines):
    rows = [[s.strip() for s in line.strip().strip('|').split('|')] for line in lines]
    rows = [r for r in rows if not all(re.fullmatch(r':?-+:?', c) for c in r)]
    n = len(rows[0])
    if rows[0][0] == 'Серия':
        widths = [.20, .49, .31]
    elif n == 2:
        widths = [.66, .34]
    elif n == 4:
        widths = [.43, .20, .19, .18]
    else:
        widths = [1/n] * n
    content = [[paragraph(c, 'headcell' if i == 0 else 'cell') for c in r] for i, r in enumerate(rows)]
    t = Table(content, colWidths=[BODY_WIDTH*x for x in widths], repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), WASH),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LINEBELOW', (0, 0), (-1, 0), .7, RULE),
        ('LINEBELOW', (0, 1), (-1, -1), .35, RULE),
    ]))
    return [t, Spacer(1, 10)]


def decorate_cover(canvas, doc):
    canvas.saveState()
    canvas.setFont('Arial', 9)
    canvas.setFillColor(ACCENT)
    canvas.drawString(LEFT, HEIGHT-64, 'ИССЛЕДОВАТЕЛЬСКАЯ ЗАМЕТКА')
    canvas.setStrokeColor(ACCENT)
    canvas.setLineWidth(1.6)
    canvas.line(LEFT, HEIGHT-82, LEFT+64, HEIGHT-82)
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(.5)
    canvas.line(LEFT, 104, WIDTH-RIGHT, 104)
    canvas.setFont('Arial', 10)
    canvas.setFillColor(MUTED)
    canvas.drawString(LEFT, 83, 'Версия 1.0.0')
    canvas.drawString(LEFT, 65, 'Заметка и воспроизводимый комплект')
    canvas.restoreState()


def decorate(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(.5)
    canvas.line(LEFT, HEIGHT-36, WIDTH-RIGHT, HEIGHT-36)
    canvas.setFillColor(MUTED)
    canvas.setFont('Arial', 8)
    canvas.drawString(LEFT, HEIGHT-27, 'Локальное восстановление ресурсной сети')
    canvas.drawRightString(WIDTH-RIGHT, HEIGHT-27, 'ВЕРСИЯ 1.0.0  |  17.11.2026')
    canvas.line(LEFT, 38, WIDTH-RIGHT, 38)
    canvas.drawString(LEFT, 25, 'Вычислительные результаты  |  исходные серии 08.10.2026')
    canvas.drawRightString(WIDTH-RIGHT, 25, str(doc.page - 1))
    canvas.restoreState()


def main():
    source = (HERE / 'note.ru.md').read_text(encoding='utf-8')
    lines = source.splitlines()
    story = [Spacer(1, 88)]
    cover = True
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == '<!-- pagebreak -->':
            story.append(PageBreak()); cover = False; i += 1; continue
        if line.startswith('|'):
            group=[]
            while i < len(lines) and lines[i].strip().startswith('|'):
                group.append(lines[i]); i += 1
            story.extend(table(group)); continue
        if line.startswith('!['):
            m = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line)
            path = HERE / m[2]
            with PILImage.open(path) as im:
                w, h = im.size
            story.append(Image(str(path), width=BODY_WIDTH, height=BODY_WIDTH*h/w))
            story.append(Spacer(1, 9)); i += 1; continue
        if line.startswith('# '):
            story.append(paragraph(line[2:], 'cover-title' if cover else 'title')); i += 1; continue
        if line.startswith('## '):
            style = 'cover-subtitle' if cover else 'subtitle' if not line[3:4].isdigit() else 'h2'
            story.append(paragraph(line[3:], style)); i += 1; continue
        if line.startswith('### '):
            story.append(paragraph(line[4:], 'h3')); i += 1; continue
        if line.startswith('- '):
            p = Paragraph('- ' + inline(line[2:]), STYLES['body'])
            story.append(p); i += 1; continue
        group = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(('#', '|', '![', '<!--', '- ')):
            group.append(lines[i].strip()); i += 1
        text = ' '.join(group)
        style = 'reference' if re.match(r'^\[[DL]\d\]', text) else 'caption' if text.startswith('Рисунок 1.') else 'body'
        if cover:
            style = 'cover-author' if text.startswith('**SymbiosisK') else 'cover-meta' if text.startswith('Исследовательская заметка.') else 'cover-note'
        story.append(paragraph(text, style))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=RIGHT, leftMargin=LEFT,
                            topMargin=53, bottomMargin=53, title='Локальное восстановление ресурсной сети',
                            author='SymbiosisK (Katia) & ChatGPT / Codex (OpenAI)', subject='Исследовательская заметка, версия 1.0.0; три серии 8 октября 2026')
    doc.build(story, onFirstPage=decorate_cover, onLaterPages=decorate)
    print(OUTPUT)


if __name__ == '__main__':
    main()
