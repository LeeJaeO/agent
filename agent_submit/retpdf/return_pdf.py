from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.generate_company_json import update_company_json_from_rules


BASE_DIR = Path(__file__).resolve().parent
JSON_DIR = BASE_DIR / "json"
HTML_DIR = BASE_DIR / "html"
PDF_DIR = BASE_DIR / "pdf"
DOCS_DIR = BASE_DIR / "docs"
MAP_DIR = BASE_DIR / "map"
DEFAULT_HTML_INPUT = HTML_DIR / "quote_filled.html"
DEFAULT_TEMPLATE_INPUT = HTML_DIR / "quote_template.html"
DEFAULT_CONDA_ENV = "pdf"
DEFAULT_FORMAT_DATA = JSON_DIR / "quote_format.json"
DEFAULT_CUSTOMER_DATA = JSON_DIR / "quote_customer.json"
DEFAULT_COMPANY_DATA = JSON_DIR / "quote_company.json"
DEFAULT_RULES_MAP = MAP_DIR / "quote_rules.json"
HTML_SUFFIXES = {".html", ".htm"}
PLACEHOLDER_PATTERN = re.compile(r"{{\s*([a-zA-Z0-9_]+)\s*}}")
GENERATED_CONTEXT_KEYS = {
    "detail_sections_html",
    "stored_items_html",
    "discarded_items_html",
}


def collect_html_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in HTML_SUFFIXES
    )


def resolve_existing_path(raw_path: str, *, search_dirs: list[Path], kind: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"{kind} 파일을 찾지 못했습니다: {resolved}")

    if candidate.parent != Path("."):
        resolved = (BASE_DIR / candidate).resolve()
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"{kind} 파일을 찾지 못했습니다: {resolved}")

    search_candidates = [(directory / candidate.name).resolve() for directory in search_dirs]
    search_candidates.append((BASE_DIR / candidate.name).resolve())
    for resolved in search_candidates:
        if resolved.exists():
            return resolved

    searched_text = ", ".join(str(path) for path in search_candidates)
    raise FileNotFoundError(f"{kind} 파일을 찾지 못했습니다: {raw_path}\n검색 위치: {searched_text}")


def resolve_output_path(
    raw_path: str | None,
    *,
    default_path: Path,
    default_dir: Path,
    suffixes: set[str] | None = None,
    label: str,
) -> Path:
    if raw_path is None:
        return default_path.resolve()

    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    elif candidate.parent != Path("."):
        resolved = (BASE_DIR / candidate).resolve()
    else:
        resolved = (default_dir / candidate.name).resolve()

    if suffixes and resolved.suffix.lower() not in suffixes:
        suffix_text = ", ".join(sorted(suffixes))
        raise ValueError(f"{label} 경로는 다음 확장자여야 합니다: {suffix_text} ({resolved})")

    return resolved


def resolve_input_path(raw_path: str) -> Path:
    candidate = resolve_existing_path(raw_path, search_dirs=[HTML_DIR, BASE_DIR], kind="HTML")
    if candidate.suffix.lower() not in HTML_SUFFIXES:
        raise ValueError(f"HTML 파일만 변환할 수 있습니다: {candidate}")
    return candidate


def resolve_json_path(raw_path: str) -> Path:
    candidate = resolve_existing_path(raw_path, search_dirs=[JSON_DIR, BASE_DIR], kind="JSON 데이터")
    if candidate.suffix.lower() != ".json":
        raise ValueError(f"JSON 파일만 입력할 수 있습니다: {candidate}")
    return candidate


def resolve_rules_map_path(raw_path: str) -> Path:
    candidate = resolve_existing_path(raw_path, search_dirs=[MAP_DIR, BASE_DIR], kind="매핑 테이블")
    if candidate.suffix.lower() != ".json":
        raise ValueError(f"매핑 테이블은 JSON 파일이어야 합니다: {candidate}")
    return candidate


def resolve_pdf_output_path(raw_path: str | None, html_path: Path) -> Path:
    default_path = PDF_DIR / f"{html_path.stem}_rendered.pdf"
    return resolve_output_path(
        raw_path,
        default_path=default_path,
        default_dir=PDF_DIR,
        suffixes={".pdf"},
        label="PDF 출력",
    )


def resolve_generated_html_path(raw_path: str | None, template_path: Path) -> Path:
    stem = template_path.stem
    if stem.endswith("_template"):
        stem = stem[: -len("_template")]
    default_path = HTML_DIR / f"{stem}_filled.html"
    return resolve_output_path(
        raw_path,
        default_path=default_path,
        default_dir=HTML_DIR,
        suffixes=HTML_SUFFIXES,
        label="생성 HTML",
    )


def escape_text(value: object) -> str:
    return html.escape(str(value), quote=True)


def validate_html_for_weasyprint(html_path: Path) -> None:
    html_text = html_path.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", html_text)

    js_render_patterns = [
        r'<div[^>]+id=["\']app["\'][^>]*></div>',
        r'<div[^>]+id=["\']root["\'][^>]*></div>',
        r'document\.getElementById\(["\']app["\']\)\.innerHTML\s*=',
        r'document\.getElementById\(["\']root["\']\)\.innerHTML\s*=',
        r'innerHTML\s*=',
        r'insertAdjacentHTML\s*\(',
    ]
    if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in js_render_patterns):
        raise RuntimeError(
            "이 HTML은 JavaScript로 본문을 그리는 구조라서 WeasyPrint로는 렌더링할 수 없습니다.\n"
            f"파일: {html_path}\n"
            "예: <div id=\"app\"></div> 뒤에 script가 innerHTML로 내용을 주입하는 형태\n"
            "해결 방법: 최종 결과가 이미 들어있는 정적 HTML로 바꾸거나, 서버 측에서 HTML을 먼저 렌더링한 뒤 PDF로 변환하세요."
        )


def load_quote_data(data_path: Path) -> dict:
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"JSON 최상위 구조는 객체여야 합니다: {data_path}")
    return raw


def merge_quote_data(data_paths: list[Path]) -> dict:
    merged: dict = {}
    for data_path in data_paths:
        raw = load_quote_data(data_path)
        merged.update(raw)
    return merged


def render_detail_rows_html(rows: object) -> str:
    if rows in (None, []):
        return '<tr><td colspan="2" style="height:22px;"></td></tr>'
    if not isinstance(rows, list):
        raise ValueError("detail_rows 는 배열이어야 합니다.")

    rendered_rows = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("detail_rows 의 각 항목은 객체여야 합니다.")
        label = escape_text(row.get("label", ""))
        amount = escape_text(row.get("amount", ""))
        rendered_rows.append(
            "<tr>"
            f"<td>{label}</td>"
            f"<td style=\"text-align:right;\">{amount}</td>"
            "</tr>"
        )
    return "".join(rendered_rows)


def render_detail_sections_html(detail_sections: object) -> str:
    if detail_sections in (None, []):
        fallback_rows = render_detail_rows_html(None)
        return (
            '<div class="detail-group">'
            '<div class="detail-group-title">세부항목</div>'
            f'<table class="detail-table">{fallback_rows}</table>'
            "</div>"
        )

    if not isinstance(detail_sections, list):
        raise ValueError("detail_sections 는 배열이어야 합니다.")

    blocks = []
    for section in detail_sections:
        if not isinstance(section, dict):
            raise ValueError("detail_sections 의 각 항목은 객체여야 합니다.")
        title = escape_text(section.get("title", "세부항목"))
        rows_html = render_detail_rows_html(section.get("rows"))
        blocks.append(
            '<div class="detail-group">'
            f'<div class="detail-group-title">{title}</div>'
            f'<table class="detail-table">{rows_html}</table>'
            "</div>"
        )
    return "".join(blocks)


def render_list_items_html(items: object) -> str:
    if items in (None, []):
        return "<li>없음</li>"
    if not isinstance(items, list):
        raise ValueError("리스트 항목은 배열이어야 합니다.")
    return "".join(f"<li>{escape_text(item)}</li>" for item in items)


def extract_template_placeholders(template_text: str) -> set[str]:
    return {match.group(1) for match in PLACEHOLDER_PATTERN.finditer(template_text)}


def build_quote_context(template_text: str, data: dict) -> dict[str, str]:
    template_fields = extract_template_placeholders(template_text)
    required_fields = sorted(template_fields - GENERATED_CONTEXT_KEYS)
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        missing_text = ", ".join(missing_fields)
        raise KeyError(f"JSON에 필요한 항목이 없습니다: {missing_text}")

    context = {field: escape_text(data[field]) for field in required_fields}
    context["detail_sections_html"] = render_detail_sections_html(
        data.get("detail_sections")
        if data.get("detail_sections") is not None
        else [{"title": "세부항목", "rows": data.get("detail_rows")}]
    )
    context["stored_items_html"] = render_list_items_html(data.get("stored_items"))
    context["discarded_items_html"] = render_list_items_html(data.get("discarded_items"))
    return context


def render_template_text(template_text: str, context: dict[str, str]) -> str:
    missing_placeholders: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        value = context.get(key)
        if value is None:
            missing_placeholders.add(key)
            return match.group(0)
        return value

    rendered = PLACEHOLDER_PATTERN.sub(replace, template_text)
    if missing_placeholders:
        missing_text = ", ".join(sorted(missing_placeholders))
        raise KeyError(f"템플릿에 대응되는 값이 없습니다: {missing_text}")
    return rendered


def render_quote_html(template_path: Path, data_paths: list[Path]) -> str:
    template_text = template_path.read_text(encoding="utf-8")
    quote_data = merge_quote_data(data_paths)
    context = build_quote_context(template_text, quote_data)
    return render_template_text(template_text, context)


def collect_data_paths(args: argparse.Namespace) -> list[Path]:
    data_paths: list[Path] = []

    for raw_path in [args.format_data, args.customer_data, args.company_data]:
        if raw_path:
            data_paths.append(resolve_json_path(raw_path))

    for raw_path in args.data_file:
        data_paths.append(resolve_json_path(raw_path))

    return data_paths


def write_html_file(output_path: Path, html_text: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")


def render_pdf_with_local_weasyprint(html_path: Path, pdf_path: Path) -> None:
    from weasyprint import HTML

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(filename=str(html_path), base_url=str(html_path.parent)).write_pdf(str(pdf_path))


def render_pdf_with_conda_weasyprint(
    html_path: Path,
    pdf_path: Path,
    conda_env: str,
) -> None:
    conda_executable = shutil.which("conda")
    if not conda_executable:
        raise RuntimeError(
            "현재 Python에는 weasyprint가 없고, conda 실행 파일도 찾지 못했습니다.\n"
            f"'{conda_env}' 환경에서 실행하거나 weasyprint를 현재 환경에 설치해주세요."
        )

    helper_code = (
        "import sys; "
        "from pathlib import Path; "
        "from weasyprint import HTML; "
        "html_path = Path(sys.argv[1]).resolve(); "
        "pdf_path = Path(sys.argv[2]).resolve(); "
        "pdf_path.parent.mkdir(parents=True, exist_ok=True); "
        "HTML(filename=str(html_path), base_url=str(html_path.parent)).write_pdf(str(pdf_path))"
    )
    cmd = [
        conda_executable,
        "run",
        "-n",
        conda_env,
        "python3",
        "-c",
        helper_code,
        str(html_path),
        str(pdf_path),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "conda 환경의 weasyprint로 PDF를 생성하지 못했습니다.\n"
            f"conda env: {conda_env}\n"
            f"html: {html_path}\n"
            f"stdout: {completed.stdout.strip()}\n"
            f"stderr: {completed.stderr.strip()}"
        )


def render_pdf(html_path: Path, pdf_path: Path, conda_env: str) -> str:
    try:
        render_pdf_with_local_weasyprint(html_path, pdf_path)
        return f"현재 Python 환경({Path(sys.executable)})의 weasyprint 사용"
    except ModuleNotFoundError:
        render_pdf_with_conda_weasyprint(html_path, pdf_path, conda_env)
        return f"conda 환경 '{conda_env}'의 weasyprint 사용"


def print_available_html_files() -> None:
    html_files = collect_html_files(HTML_DIR)
    if not html_files:
        print(f"HTML 파일 없음: {HTML_DIR}")
        return

    print(f"HTML 파일 목록: {HTML_DIR}")
    for path in html_files:
        print(f"- {path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HTML을 PDF로 변환하거나, quote 템플릿과 JSON으로 채워진 HTML/PDF를 생성합니다."
    )
    parser.add_argument(
        "html_input",
        nargs="?",
        default=DEFAULT_HTML_INPUT.name,
        help="직접 PDF로 변환할 HTML 파일 경로. JSON 데이터 모드 사용 시 무시됩니다.",
    )
    parser.add_argument(
        "--data-file",
        action="append",
        default=[],
        help="quote 템플릿에 채워 넣을 JSON 데이터 파일 경로. 여러 번 지정할 수 있습니다.",
    )
    parser.add_argument(
        "--format-data",
        help=f"quote 포맷용 JSON 파일 경로. 예: {DEFAULT_FORMAT_DATA.name}",
    )
    parser.add_argument(
        "--customer-data",
        help=f"고객 입력용 JSON 파일 경로. 예: {DEFAULT_CUSTOMER_DATA.name}",
    )
    parser.add_argument(
        "--company-data",
        help=f"업체 입력용 JSON 파일 경로. 예: {DEFAULT_COMPANY_DATA.name}",
    )
    parser.add_argument(
        "--rules-map",
        default=DEFAULT_RULES_MAP.name,
        help=f"자동 company 계산에 사용할 매핑 테이블 JSON 경로. 예: {DEFAULT_RULES_MAP.name}",
    )
    parser.add_argument(
        "--template",
        default=DEFAULT_TEMPLATE_INPUT.name,
        help="데이터 입력 모드에서 사용할 HTML 템플릿 파일 경로",
    )
    parser.add_argument(
        "--html-output",
        help="데이터 입력 모드에서 생성할 채워진 HTML 파일 경로",
    )
    parser.add_argument(
        "--pdf-output",
        help="생성할 PDF 경로. 생략하면 <html파일명>_rendered.pdf 로 저장합니다.",
    )
    parser.add_argument(
        "--conda-env",
        default=DEFAULT_CONDA_ENV,
        help="현재 Python에 weasyprint가 없을 때 사용할 conda 환경 이름",
    )
    parser.add_argument(
        "--list-html",
        action="store_true",
        help="같은 디렉토리의 HTML 파일 목록만 출력하고 종료합니다.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()

        if args.list_html:
            print_available_html_files()
            return

        generated_html_path: Path | None = None
        data_paths = collect_data_paths(args)
        if data_paths:
            if args.customer_data and args.company_data:
                customer_data_path = resolve_json_path(args.customer_data)
                company_data_path = resolve_json_path(args.company_data)
                rules_map_path = resolve_rules_map_path(args.rules_map)
                update_company_json_from_rules(
                    customer_json_path=customer_data_path,
                    company_json_path=company_data_path,
                    rules_json_path=rules_map_path,
                )
                print(f"업체 JSON 자동 계산 완료: {company_data_path}")
            template_path = resolve_input_path(args.template)
            generated_html_path = resolve_generated_html_path(args.html_output, template_path)
            html_text = render_quote_html(template_path, data_paths)
            write_html_file(generated_html_path, html_text)
            html_path = generated_html_path
        else:
            if args.html_output:
                raise ValueError("--html-output 옵션은 JSON 데이터 모드와 함께 사용해주세요.")
            html_path = resolve_input_path(args.html_input)

        validate_html_for_weasyprint(html_path)
        pdf_path = resolve_pdf_output_path(args.pdf_output, html_path)
        render_message = render_pdf(html_path, pdf_path, args.conda_env)

        if generated_html_path is not None:
            print(f"생성 HTML: {generated_html_path}")
        print(f"HTML 입력: {html_path}")
        print(f"PDF 출력: {pdf_path}")
        print(f"렌더링 엔진: {render_message}")
    except Exception as exc:
        print(f"오류: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
