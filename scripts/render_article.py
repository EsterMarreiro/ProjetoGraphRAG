"""Gera a fonte SBC do texto revisado; não inventa metadados de autoria."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def escape(text: str) -> str:
    return "".join(
        {
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
            "\\": r"\textbackslash{}",
        }.get(char, char)
        for char in text
    )


def inline(text: str) -> str:
    text = escape(text)
    for original, key in [
        ("Lewis et al. (2020)", "lewis2020"),
        ("Edge et al. (2025)", "edge2025"),
        ("Liang et al. (2024)", "liang2024"),
        ("Cormack, Clarke e Buettcher (2009)", "cormack2009"),
    ]:
        text = text.replace(original, r"\cite{" + key + "}")
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    return re.sub(r"`([^`]+)`", r"\\texttt{\1}", text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authors-json", type=Path)
    args = parser.parse_args()
    authors = "Autoria em confirmação"
    address = "Ciência da Computação"
    if args.authors_json:
        raw = json.loads(args.authors_json.read_text())
        names = raw.get("authors", [])
        if not 2 <= len(names) <= 5 or not all(
            isinstance(n, str) and n.strip() for n in names
        ):
            raise ValueError("Informe de 2 a 5 autores reais")
        authors = ", ".join(names)
        address = raw["institution"]
    text = (ROOT / "docs/ARTIGO_TECNICO.md").read_text()
    lines = text.splitlines()
    title = lines[0].removeprefix("# ")
    output = [
        r"\documentclass[12pt]{article}",
        r"\usepackage{sbc-template}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[brazil]{babel}",
        r"\usepackage{url}",
        r"\usepackage{graphicx}",
        r"\sloppy",
        r"\title{" + escape(title) + "}",
        r"\author{" + escape(authors) + "}",
        r"\address{" + escape(address) + "}",
        r"\begin{document}",
        r"\maketitle",
    ]
    for section in re.split(r"^## ", "\n".join(lines[1:]), flags=re.MULTILINE)[1:]:
        heading, _, content = section.partition("\n")
        if heading == "REFERÊNCIAS":
            break
        if heading in ("Resumo", "Abstract"):
            env = "resumo" if heading == "Resumo" else "abstract"
            body, _, keywords = content.partition("\n**")
            output.extend(
                [r"\begin{" + env + "}", inline(body.strip()), r"\end{" + env + "}"]
            )
            if keywords:
                output.append(inline("**" + keywords.strip()) + "\n")
            continue
        heading = re.sub(r"^\d+\s+", "", heading)
        output.append(r"\section{" + escape(heading) + "}")
        for paragraph in content.strip().split("\n\n"):
            if paragraph.startswith("<!-- table:"):
                caption_line, _, body = paragraph.partition("\n")
                caption = (
                    caption_line.removeprefix("<!-- table:").removesuffix("-->").strip()
                )
                rows = [
                    [cell.strip() for cell in line.strip().strip("|").split("|")]
                    for line in body.splitlines()
                    if line.strip()
                ]
                rows = [
                    row
                    for row in rows
                    if not all(re.fullmatch(r"[-: ]+", cell) for cell in row)
                ]
                columns = "l" + "r" * (len(rows[0]) - 1)
                output.extend(
                    [
                        r"\begin{table}[ht]",
                        r"\centering",
                        r"\caption{" + escape(caption) + "}",
                        r"\begin{tabular}{" + columns + "}",
                        r"\hline",
                    ]
                )
                for index, row in enumerate(rows):
                    output.append(" & ".join(inline(cell) for cell in row) + r" \\")
                    if index == 0:
                        output.append(r"\hline")
                output.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
                continue
            figure = re.fullmatch(r"!\[(.+)\]\(artigo/([^)]+)\)", paragraph)
            if figure:
                caption, filename = figure.groups()
                output.extend(
                    [
                        r"\begin{figure}[ht]",
                        r"\centering",
                        r"\includegraphics[width=\textwidth]{" + filename + "}",
                        r"\caption{" + escape(caption) + "}",
                        r"\end{figure}",
                    ]
                )
            elif paragraph.startswith("### "):
                title = re.sub(r"^### \d+\.\d+\s+", "", paragraph)
                output.append(r"\subsection{" + escape(title) + "}")
            else:
                output.append(inline(paragraph) + "\n")
    output.extend(
        [r"\bibliographystyle{sbc}", r"\bibliography{referencias}", r"\end{document}"]
    )
    (ROOT / "docs/artigo/main.tex").write_text("\n".join(output) + "\n")


if __name__ == "__main__":
    main()
