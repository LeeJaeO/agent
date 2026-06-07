from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalize_text(value: object) -> str:
    return str(value).strip().replace(" ", "")


def load_json(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"JSON 최상위 구조는 객체여야 합니다: {path}")
    return raw


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_manwon(text: str) -> int:
    normalized = str(text).strip().replace("만원", "").replace(",", "")
    return int(float(normalized))


def format_manwon(value: int) -> str:
    return f"{value}만원"


def pick_range_value(rule: object, selection_policy: str) -> int:
    if isinstance(rule, (int, float)):
        return int(round(rule))

    if not isinstance(rule, dict):
        raise ValueError(f"지원하지 않는 금액 규칙 형식입니다: {rule}")

    min_value = int(rule["min_manwon"])
    max_value = int(rule.get("max_manwon", min_value))

    if selection_policy == "min":
        return min_value
    if selection_policy == "max":
        return max_value
    if selection_policy == "midpoint":
        return int(round((min_value + max_value) / 2))

    raise ValueError(f"지원하지 않는 amount selection_policy 입니다: {selection_policy}")


def pick_staff_value(staff_rule: object, work_type: str) -> str:
    if isinstance(staff_rule, str):
        return staff_rule

    if not isinstance(staff_rule, dict):
        raise ValueError(f"지원하지 않는 인원 규칙 형식입니다: {staff_rule}")

    if work_type in staff_rule:
        return str(staff_rule[work_type])
    if "default" in staff_rule:
        return str(staff_rule["default"])

    raise KeyError(f"인원 추천표에 work_type '{work_type}' 또는 default 값이 없습니다.")


def find_rule_by_normalized_key(mapping: dict, raw_key: str) -> tuple[str, object]:
    normalized_target = normalize_text(raw_key)
    for key, value in mapping.items():
        if normalize_text(key) == normalized_target:
            return str(key), value
    raise KeyError(f"규칙표에 '{raw_key}' 에 해당하는 항목이 없습니다.")


def resolve_directional_amount_rule(rule: object, direction: str) -> object:
    if isinstance(rule, (int, float)):
        return rule

    if not isinstance(rule, dict):
        raise ValueError(f"지원하지 않는 방향별 금액 규칙 형식입니다: {rule}")

    if direction in rule:
        return rule[direction]

    if "min_manwon" in rule:
        return rule

    return {"min_manwon": 0, "max_manwon": 0}


def calculate_deposit_total(move_subtotal: int, rules: dict) -> str:
    deposit_rules = rules.get("deposit_recommendation", [])
    for rule in deposit_rules:
        min_total = int(rule.get("min_total_manwon", 0))
        max_total = rule.get("max_total_manwon")
        if move_subtotal < min_total:
            continue
        if max_total is not None and move_subtotal > int(max_total):
            continue
        return format_manwon(int(rule["deposit_manwon"]))

    raise KeyError(f"계약금 추천표에 해당하는 구간이 없습니다: {move_subtotal}만원")


def build_detail_section(title: str, rows: list[dict]) -> dict:
    return {
        "title": title,
        "rows": rows,
    }


def calculate_direction_quote(
    *,
    direction: str,
    truck_type: str,
    work_type: str,
    rules: dict,
) -> tuple[str, str, list[dict]]:
    amount_policy = str(rules.get("selection_policy", {}).get("amount", "midpoint"))
    staff_rules = rules["staff_recommendation"][truck_type]
    base_rules = rules["base_amount_recommendation"][truck_type][direction]
    adjustment_rules = rules.get("work_type_adjustments", {})
    matched_work_type, work_adjustment_rule = find_rule_by_normalized_key(
        adjustment_rules,
        work_type,
    )
    work_adjustment = resolve_directional_amount_rule(work_adjustment_rule, direction)

    staff_text = pick_staff_value(staff_rules, work_type)
    adjustment_value = pick_range_value(work_adjustment, amount_policy)
    amount_value = pick_range_value(base_rules, amount_policy) + adjustment_value

    detail_rows: list[dict] = []
    label = str(work_adjustment_rule.get("label", matched_work_type))
    if adjustment_value != 0:
        detail_rows.append(
            {
                "label": label,
                "amount": format_manwon(adjustment_value),
            }
        )

    return staff_text, format_manwon(amount_value), detail_rows


def update_company_payload(customer_data: dict, company_data: dict, rules: dict) -> dict:
    updated = dict(company_data)
    truck_type = str(updated["truck_type"])
    inbound_work_type = str(customer_data["inbound_work_type"])
    outbound_work_type = str(customer_data["outbound_work_type"])

    inbound_staff, inbound_total, inbound_detail_rows = calculate_direction_quote(
        direction="inbound",
        truck_type=truck_type,
        work_type=inbound_work_type,
        rules=rules,
    )
    outbound_staff, outbound_total, outbound_detail_rows = calculate_direction_quote(
        direction="outbound",
        truck_type=truck_type,
        work_type=outbound_work_type,
        rules=rules,
    )

    updated["inbound_staff"] = inbound_staff
    updated["outbound_staff"] = outbound_staff
    updated["inbound_total"] = inbound_total
    updated["outbound_total"] = outbound_total

    move_subtotal = parse_manwon(inbound_total) + parse_manwon(outbound_total)
    deposit_total = calculate_deposit_total(move_subtotal, rules)
    updated["deposit_total"] = deposit_total
    updated["grand_total"] = format_manwon(
        move_subtotal + parse_manwon(deposit_total)
    )
    updated["detail_sections"] = [
        build_detail_section("입고 세부항목", inbound_detail_rows),
        build_detail_section("출고 세부항목", outbound_detail_rows),
    ]

    return updated


def update_company_json_from_rules(
    *,
    customer_json_path: Path,
    company_json_path: Path,
    rules_json_path: Path,
) -> dict:
    customer_data = load_json(customer_json_path)
    company_data = load_json(company_json_path)
    rules = load_json(rules_json_path)

    updated = update_company_payload(customer_data, company_data, rules)
    write_json(company_json_path, updated)
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="고객 정보와 매핑 규칙을 바탕으로 quote_company.json 을 자동 계산합니다."
    )
    parser.add_argument("customer_json", help="고객 입력 JSON 경로")
    parser.add_argument("company_json", help="업체 JSON 경로")
    parser.add_argument("rules_json", help="매핑 테이블 JSON 경로")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    update_company_json_from_rules(
        customer_json_path=Path(args.customer_json).expanduser().resolve(),
        company_json_path=Path(args.company_json).expanduser().resolve(),
        rules_json_path=Path(args.rules_json).expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
