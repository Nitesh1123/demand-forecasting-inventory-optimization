"""Generate the DemandForecasting Power BI project (.pbip) as TMDL + PBIR text files.

Run from the repository root after 04_powerbi_export.py:
    python powerbi/build_pbip.py
Then open powerbi/DemandForecasting.pbip in Power BI Desktop and click Refresh.
"""
import json
import os
import shutil
import uuid
from pathlib import Path

PROJECT_NAME = "DemandForecasting"
POWERBI_DIRECTORY = Path(__file__).resolve().parent
DATA_FOLDER = (POWERBI_DIRECTORY.parent / "outputs" / "powerbi").resolve()
SCHEMA_ROOT = "https://developer.microsoft.com/json-schemas/fabric/item/report/definition"

M_TYPES = {"int64": "Int64.Type", "double": "type number", "string": "type text", "dateTime": "type date"}

# column name -> data type, per exported CSV
TABLES = {
    "fact_sales": {"date": "dateTime", "store": "int64", "item": "int64", "sales": "int64"},
    "fact_forecast": {
        "date": "dateTime",
        "store": "int64",
        "item": "int64",
        "actual": "double",
        "forecast": "double",
        "baseline_forecast": "double",
        "absolute_error": "double",
        "baseline_absolute_error": "double",
    },
    "fact_inventory": {
        "item": "int64",
        "store": "int64",
        "forecast_demand": "double",
        "actual_demand": "double",
        "total_item_supply": "int64",
        "pulp_allocation": "int64",
        "proportional_allocation": "int64",
        "equal_split_allocation": "int64",
        "pulp_actual_unmet": "double",
        "proportional_actual_unmet": "double",
        "equal_split_actual_unmet": "double",
        "forecast_mean_daily_demand": "double",
        "residual_std_daily": "double",
        "safety_stock": "double",
        "reorder_point": "double",
    },
    "dim_date": {
        "date": "dateTime",
        "year": "int64",
        "quarter": "string",
        "month_number": "int64",
        "month_name": "string",
        "year_month": "string",
        "month_start": "dateTime",
        "week_of_year": "int64",
        "day_of_week_number": "int64",
        "day_of_week_name": "string",
        "is_weekend": "int64",
        "is_us_holiday": "int64",
        "period": "string",
    },
    "dim_store": {"store": "int64", "store_name": "string"},
    "dim_item": {"item": "int64", "item_name": "string", "abc_class": "string"},
    "model_metrics": {"model": "string", "wape": "double", "rmse": "double", "bias": "double"},
}
# What-if parameter tables: name -> (column data type, format, DAX table expression, default value)
WHAT_IF_TABLES = {
    # Whole-number series divided by 100, so slider values compare exactly (0.01 steps drift in floating point).
    "Service Level": ("double", "0%", 'SELECTCOLUMNS ( GENERATESERIES ( 80, 99, 1 ), "Value", [Value] / 100 )', "0.95"),
    "Lead Time": ("int64", "0", "GENERATESERIES ( 1, 28, 1 )", "7"),
}
SORT_BY = {
    ("dim_date", "month_name"): "month_number",
    ("dim_date", "day_of_week_name"): "day_of_week_number",
    ("dim_store", "store_name"): "store",
    ("dim_item", "item_name"): "item",
}
KEY_COLUMNS = {"date", "store", "item", "year", "month_number", "week_of_year", "day_of_week_number", "month_start"}

PERCENT = "0.0%"
UNITS = "#,0"
TEXT = None
GOOD_COLOR, WARN_COLOR, BAD_COLOR = "#C8E6C9", "#FFF3C4", "#FFCDD2"
MEASURES = {
    "fact_sales": [
        ("Total Sales", "SUM ( fact_sales[sales] )", UNITS),
        ("Sales LY", "CALCULATE ( [Total Sales], SAMEPERIODLASTYEAR ( dim_date[date] ) )", UNITS),
        (
            "YoY Growth %",
            "VAR LatestYear = MAX ( dim_date[year] ) "
            "VAR CurrentSales = CALCULATE ( [Total Sales], dim_date[year] = LatestYear ) "
            "VAR PreviousSales = CALCULATE ( [Total Sales], dim_date[year] = LatestYear - 1 ) "
            "RETURN DIVIDE ( CurrentSales - PreviousSales, PreviousSales )",
            PERCENT,
        ),
        ("Avg Daily Sales", "AVERAGEX ( VALUES ( dim_date[date] ), [Total Sales] )", UNITS),
    ],
    "fact_forecast": [
        ("Actual Units", "SUM ( fact_forecast[actual] )", UNITS),
        ("Forecast Units", "SUM ( fact_forecast[forecast] )", UNITS),
        ("Baseline Units", "SUM ( fact_forecast[baseline_forecast] )", UNITS),
        ("WAPE", "DIVIDE ( SUM ( fact_forecast[absolute_error] ), [Actual Units] )", PERCENT),
        ("Baseline WAPE", "DIVIDE ( SUM ( fact_forecast[baseline_absolute_error] ), [Actual Units] )", PERCENT),
        ("Forecast Accuracy", "1 - [WAPE]", PERCENT),
        ("Forecast Bias", "DIVIDE ( [Forecast Units] - [Actual Units], [Actual Units] )", PERCENT),
        (
            "Error Reduction",
            "1 - DIVIDE ( SUM ( fact_forecast[absolute_error] ), SUM ( fact_forecast[baseline_absolute_error] ) )",
            PERCENT,
        ),
        (
            "WAPE Color",
            f'SWITCH ( TRUE (), ISBLANK ( [WAPE] ), BLANK (), [WAPE] <= 0.12, "{GOOD_COLOR}", '
            f'[WAPE] <= 0.18, "{WARN_COLOR}", "{BAD_COLOR}" )',
            TEXT,
        ),
        (
            "Forecast Chart Title",
            'VAR StoreCount = COUNTROWS ( VALUES ( dim_store[store_name] ) ) '
            'VAR ItemCount = COUNTROWS ( VALUES ( dim_item[item_name] ) ) '
            'VAR Stores = IF ( NOT ISFILTERED ( dim_store ), "all stores", '
            'IF ( StoreCount <= 3, CONCATENATEX ( VALUES ( dim_store[store_name] ), dim_store[store_name], ", " ), '
            'StoreCount & " stores" ) ) '
            'VAR Items = IF ( NOT ISFILTERED ( dim_item ), "all items", '
            'IF ( ItemCount <= 3, CONCATENATEX ( VALUES ( dim_item[item_name] ), dim_item[item_name], ", " ), '
            'ItemCount & " items" ) ) '
            'RETURN "Daily actual vs forecast: " & Stores & " | " & Items',
            TEXT,
        ),
        (
            "Forecast Insight",
            'VAR WorstItem = MAXX ( TOPN ( 1, ADDCOLUMNS ( VALUES ( dim_item[item_name] ), "@wape", [WAPE] ), '
            '[@wape], DESC ), dim_item[item_name] ) '
            'RETURN "LightGBM cuts forecast error by " & FORMAT ( [Error Reduction], "0.0%" ) '
            '& " vs. the seasonal-naive baseline (WAPE " & FORMAT ( [WAPE], "0.0%" ) & " vs " '
            '& FORMAT ( [Baseline WAPE], "0.0%" ) & "). Hardest item to forecast: " & WorstItem '
            '& " (WAPE " & FORMAT ( CALCULATE ( [WAPE], dim_item[item_name] = WorstItem ), "0.0%" ) & ")."',
            TEXT,
        ),
    ],
    "fact_inventory": [
        ("Inventory Demand", "SUM ( fact_inventory[actual_demand] )", UNITS),
        ("Allocated Supply", "SUM ( fact_inventory[pulp_allocation] )", UNITS),
        ("Unmet Demand (Optimised)", "SUM ( fact_inventory[pulp_actual_unmet] )", UNITS),
        ("Unmet Demand (Proportional)", "SUM ( fact_inventory[proportional_actual_unmet] )", UNITS),
        ("Unmet Demand (Equal Split)", "SUM ( fact_inventory[equal_split_actual_unmet] )", UNITS),
        ("Fill Rate (Optimised)", "1 - DIVIDE ( [Unmet Demand (Optimised)], [Inventory Demand] )", PERCENT),
        ("Fill Rate (Proportional)", "1 - DIVIDE ( [Unmet Demand (Proportional)], [Inventory Demand] )", PERCENT),
        ("Fill Rate (Equal Split)", "1 - DIVIDE ( [Unmet Demand (Equal Split)], [Inventory Demand] )", PERCENT),
        (
            "Worst Fill Rate (Optimised)",
            "MINX ( fact_inventory, 1 - DIVIDE ( fact_inventory[pulp_actual_unmet], fact_inventory[actual_demand] ) )",
            PERCENT,
        ),
        (
            "Worst Fill Rate (Equal Split)",
            "MINX ( fact_inventory, 1 - DIVIDE ( fact_inventory[equal_split_actual_unmet], fact_inventory[actual_demand] ) )",
            PERCENT,
        ),
        (
            "Fill Rate Color",
            f'SWITCH ( TRUE (), ISBLANK ( [Fill Rate (Optimised)] ), BLANK (), [Fill Rate (Optimised)] >= 0.8, '
            f'"{GOOD_COLOR}", [Fill Rate (Optimised)] >= 0.7, "{WARN_COLOR}", "{BAD_COLOR}" )',
            TEXT,
        ),
        (
            "Safety Stock (What-if)",
            "SUMX ( fact_inventory, NORM.S.INV ( [Service Level Value] ) * fact_inventory[residual_std_daily] "
            "* SQRT ( [Lead Time Value] ) )",
            UNITS,
        ),
        (
            "Reorder Point (What-if)",
            "SUMX ( fact_inventory, fact_inventory[forecast_mean_daily_demand] * [Lead Time Value] ) "
            "+ [Safety Stock (What-if)]",
            UNITS,
        ),
        (
            "Inventory Insight",
            '"With supply capped at 80% of forecast, the optimised allocation keeps every store-item at or above "'
            ' & FORMAT ( [Worst Fill Rate (Optimised)], "0%" ) & " fill rate (equal split: "'
            ' & FORMAT ( [Worst Fill Rate (Equal Split)], "0%" ) & "). At a " & FORMAT ( [Service Level Value], "0%" )'
            ' & " service level and " & [Lead Time Value] & "-day lead time, safety stock is "'
            ' & FORMAT ( [Safety Stock (What-if)], "#,0" ) & " units."',
            TEXT,
        ),
    ],
    "Service Level": [("Service Level Value", "SELECTEDVALUE ( 'Service Level'[Service Level], 0.95 )", "0%")],
    "Lead Time": [("Lead Time Value", "SELECTEDVALUE ( 'Lead Time'[Lead Time], 7 )", "0")],
}
RELATIONSHIPS = [
    ("fact_sales.date", "dim_date.date"),
    ("fact_sales.store", "dim_store.store"),
    ("fact_sales.item", "dim_item.item"),
    ("fact_forecast.date", "dim_date.date"),
    ("fact_forecast.store", "dim_store.store"),
    ("fact_forecast.item", "dim_item.item"),
    ("fact_inventory.store", "dim_store.store"),
    ("fact_inventory.item", "dim_item.item"),
]


def stable_id(*parts):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "/".join([PROJECT_NAME, *parts])))


def tmdl_name(name):
    return f"'{name}'" if not name.isidentifier() else name


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(content, str):
        content = json.dumps(content, indent=2)
    path.write_text(content, encoding="utf-8")


def measure_lines(table_name):
    lines = []
    for measure_name, expression, format_string in MEASURES.get(table_name, []):
        lines.append(f"\tmeasure {tmdl_name(measure_name)} = {expression}")
        if format_string:
            lines.append(f"\t\tformatString: {format_string}")
        lines += [f"\t\tlineageTag: {stable_id(table_name, measure_name)}", ""]
    return lines


def table_tmdl(table_name, columns):
    lines = [f"table {table_name}", f"\tlineageTag: {stable_id(table_name)}"]
    if table_name == "dim_date":
        lines.append("\tdataCategory: Time")
    lines.append("")
    lines += measure_lines(table_name)
    for column_name, data_type in columns.items():
        lines.append(f"\tcolumn {column_name}")
        lines.append(f"\t\tdataType: {data_type}")
        if table_name == "dim_date" and column_name == "date":
            lines.append("\t\tisKey")
        if data_type == "dateTime":
            lines.append("\t\tformatString: " + ("mmm yyyy" if column_name == "month_start" else "yyyy-mm-dd"))
        elif data_type == "double":
            lines.append("\t\tformatString: #,0.00")
        elif data_type == "int64":
            lines.append("\t\tformatString: 0")
        lines.append(f"\t\tlineageTag: {stable_id(table_name, column_name)}")
        summarize = "none" if column_name in KEY_COLUMNS or data_type == "string" else "sum"
        lines.append(f"\t\tsummarizeBy: {summarize}")
        lines.append(f"\t\tsourceColumn: {column_name}")
        if (table_name, column_name) in SORT_BY:
            lines.append(f"\t\tsortByColumn: {SORT_BY[(table_name, column_name)]}")
        lines.append("")
    type_pairs = ", ".join(f'{{"{name}", {M_TYPES[data_type]}}}' for name, data_type in columns.items())
    lines += [
        f"\tpartition {table_name} = m",
        "\t\tmode: import",
        "\t\tsource =",
        "\t\t\t\tlet",
        f'\t\t\t\t    Source = Csv.Document(File.Contents(DataFolder & "{table_name}.csv"), '
        "[Delimiter = \",\", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),",
        "\t\t\t\t    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),",
        f'\t\t\t\t    Typed = Table.TransformColumnTypes(Promoted, {{{type_pairs}}}, "en-US")',
        "\t\t\t\tin",
        "\t\t\t\t    Typed",
        "",
    ]
    return "\n".join(lines)


def what_if_tmdl(table_name, data_type, format_string, expression):
    lines = [f"table {tmdl_name(table_name)}", f"\tlineageTag: {stable_id(table_name)}", ""]
    lines += measure_lines(table_name)
    lines += [
        f"\tcolumn {tmdl_name(table_name)}",
        f"\t\tdataType: {data_type}",
        f"\t\tformatString: {format_string}",
        f"\t\tlineageTag: {stable_id(table_name, 'column')}",
        "\t\tsummarizeBy: none",
        "\t\tsourceColumn: [Value]",
        "",
        # Marks the table as a what-if parameter so Power BI renders a slider.
        "\t\textendedProperty ParameterMetadata =",
        "\t\t\t\t{",
        '\t\t\t\t  "version": 0',
        "\t\t\t\t}",
        "",
        f"\tpartition {tmdl_name(table_name)} = calculated",
        "\t\tmode: import",
        f"\t\tsource = {expression}",
        "",
    ]
    return "\n".join(lines)


def build_semantic_model(model_directory):
    definition = model_directory / "definition"
    write(model_directory / "definition.pbism", {"version": "4.0", "settings": {}})
    write(definition / "database.tmdl", "database\n\tcompatibilityLevel: 1600\n")
    model_lines = [
        "model Model",
        "\tculture: en-US",
        "\tdefaultPowerBIDataSourceVersion: powerBI_V3",
        "\tsourceQueryCulture: en-US",
        "\tdataAccessOptions",
        "\t\tlegacyRedirects",
        "\t\treturnErrorValuesAsNull",
        "",
    ]
    model_lines += [f"ref table {tmdl_name(table_name)}" for table_name in [*TABLES, *WHAT_IF_TABLES]]
    write(definition / "model.tmdl", "\n".join(model_lines) + "\n")
    data_folder = str(DATA_FOLDER).rstrip("\\/") + os.sep
    write(
        definition / "expressions.tmdl",
        f'expression DataFolder = "{data_folder}" '
        'meta [IsParameterQuery = true, Type = "Text", IsParameterQueryRequired = true]\n'
        f"\tlineageTag: {stable_id('DataFolder')}\n",
    )
    relationship_lines = []
    for from_column, to_column in RELATIONSHIPS:
        relationship_lines += [
            f"relationship {stable_id(from_column, to_column)}",
            f"\tfromColumn: {from_column}",
            f"\ttoColumn: {to_column}",
            "",
        ]
    write(definition / "relationships.tmdl", "\n".join(relationship_lines))
    for table_name, columns in TABLES.items():
        write(definition / "tables" / f"{table_name}.tmdl", table_tmdl(table_name, columns))
    for table_name, (data_type, format_string, expression, _) in WHAT_IF_TABLES.items():
        write(
            definition / "tables" / f"{table_name}.tmdl",
            what_if_tmdl(table_name, data_type, format_string, expression),
        )


# ---------- report helpers ----------


def field_ref(entity, name, kind):
    return {kind: {"Expression": {"SourceRef": {"Entity": entity}}, "Property": name}}


def projection(entity, name, kind):
    return {"field": field_ref(entity, name, kind), "queryRef": f"{entity}.{name}", "nativeQueryRef": name}


def measure(entity, name):
    return projection(entity, name, "Measure")


def column(entity, name):
    return projection(entity, name, "Column")


def literal(value):
    return {"expr": {"Literal": {"Value": value}}}


def text_literal(value):
    return literal("'" + value.replace("'", "''") + "'")


def measure_expr(entity, name):
    return {"expr": field_ref(entity, name, "Measure")}


def visual(name, visual_type, position, roles=None, title=None, objects=None, container_objects=None, sort=None):
    x, y, width, height = position
    content = {"visualType": visual_type, "drillFilterOtherVisuals": True}
    if roles:
        query_state = {role: {"projections": projections} for role, projections in roles.items()}
        content["query"] = {"queryState": query_state}
        if sort:
            content["query"]["sortDefinition"] = {
                "sort": [{"field": field_ref(*sort[0], "Column"), "direction": sort[1]}],
                "isDefaultSort": True,
            }
    if objects:
        content["objects"] = objects
    container_objects = dict(container_objects or {})
    if title is not None:
        title_text = measure_expr(*title) if isinstance(title, tuple) else text_literal(title)
        container_objects["title"] = [{"properties": {"show": literal("true"), "text": title_text}}]
    if container_objects:
        content["visualContainerObjects"] = container_objects
    return {
        "$schema": f"{SCHEMA_ROOT}/visualContainer/1.0.0/schema.json",
        "name": name,
        "position": {"x": x, "y": y, "z": 0, "width": width, "height": height, "tabOrder": 0},
        "visual": content,
    }


def textbox(name, text, position, font_size="18pt"):
    paragraphs = [{"textRuns": [{"value": text, "textStyle": {"fontWeight": "bold", "fontSize": font_size}}]}]
    return visual(name, "textbox", position, objects={"general": [{"properties": {"paragraphs": paragraphs}}]})


def card(name, entity, measure_name, position):
    return visual(name, "card", position, roles={"Values": [measure(entity, measure_name)]})


def insight_card(name, entity, measure_name, position):
    return visual(
        name,
        "card",
        position,
        roles={"Values": [measure(entity, measure_name)]},
        objects={
            "categoryLabels": [{"properties": {"show": literal("false")}}],
            "labels": [{"properties": {"fontSize": literal("11D"), "color": {"solid": {"color": text_literal("#1F2937")}}}}],
            "wordWrap": [{"properties": {"show": literal("true")}}],
        },
        container_objects={"background": [{"properties": {"color": {"solid": {"color": text_literal("#E6F4FB")}}}}]},
    )


def slicer(name, entity, column_name, position, mode="Basic", horizontal=False, title=None):
    # The visual title names the slicer, so the slicer's own header (the raw column name) is hidden.
    objects = {
        "data": [{"properties": {"mode": text_literal(mode)}}],
        "header": [{"properties": {"show": literal("false")}}],
    }
    if horizontal:
        objects["general"] = [{"properties": {"orientation": literal("1D")}}]
    return visual(name, "slicer", position, roles={"Values": [column(entity, column_name)]}, objects=objects, title=title)


def navigator(name):
    return visual(
        name,
        "pageNavigator",
        (700, 8, 560, 44),
        objects={"pages": [{"properties": {"showHiddenPages": literal("false")}}]},
    )


def back_button(name):
    return visual(
        name,
        "actionButton",
        (20, 12, 40, 36),
        objects={"icon": [{"properties": {"shapeType": text_literal("back")}, "selector": {"id": "default"}}]},
        container_objects={"visualLink": [{"properties": {"show": literal("true"), "type": text_literal("Back")}}]},
    )


def color_rule(entity, column_name, color_measure):
    """Background colour of one table column, driven by a DAX colour measure."""
    return {
        "properties": {"backColor": {"solid": {"color": measure_expr(entity, color_measure)}}},
        "selector": {"data": [{"dataViewWildcard": {"matchingOption": 1}}], "metadata": f"{entity}.{column_name}"},
    }


def table(name, position, values, title, color_rules=()):
    objects = {"values": [color_rule(*rule) for rule in color_rules]} if color_rules else None
    return visual(name, "tableEx", position, roles={"Values": values}, title=title, objects=objects)


ROW_SLICERS = 60
ROW_CARDS = 130
ROW_INSIGHT = 232
ROW_CHARTS = 292
CARD_WIDTH = 200
CARD_HEIGHT = 92


def card_row(prefix, cards, y=ROW_CARDS):
    return [
        card(f"{prefix}Card{index}", entity, name, (20 + index * 210, y, CARD_WIDTH, CARD_HEIGHT))
        for index, (entity, name) in enumerate(cards)
    ]


PAGES = [
    {
        "name": "overview",
        "display": "Executive Overview",
        "visuals": [
            textbox("ovTitle", "Demand Forecasting & Inventory", (20, 8, 660, 44)),
            navigator("ovNav"),
            slicer("ovYear", "dim_date", "year", (20, ROW_SLICERS, 460, 62), horizontal=True, title="Year"),
            slicer("ovStore", "dim_store", "store_name", (500, ROW_SLICERS, 220, 62), mode="Dropdown", title="Store"),
            slicer("ovAbc", "dim_item", "abc_class", (740, ROW_SLICERS, 300, 62), horizontal=True, title="ABC class"),
            *card_row(
                "ov",
                [
                    ("fact_sales", "Total Sales"),
                    ("fact_sales", "YoY Growth %"),
                    ("fact_forecast", "Forecast Accuracy"),
                    ("fact_forecast", "WAPE"),
                    ("fact_forecast", "Error Reduction"),
                    ("fact_inventory", "Fill Rate (Optimised)"),
                ],
            ),
            insight_card("ovInsight", "fact_forecast", "Forecast Insight", (20, ROW_INSIGHT, 1240, 50)),
            visual(
                "ovTrend",
                "lineChart",
                (20, ROW_CHARTS, 760, 418),
                roles={
                    "Category": [column("dim_date", "month_start")],
                    "Y": [measure("fact_sales", "Total Sales"), measure("fact_forecast", "Forecast Units")],
                },
                title="Monthly sales with the 90-day LightGBM forecast",
            ),
            visual(
                "ovWeekday",
                "clusteredColumnChart",
                (800, ROW_CHARTS, 460, 200),
                roles={
                    "Category": [column("dim_date", "day_of_week_name")],
                    "Y": [measure("fact_sales", "Avg Daily Sales")],
                },
                title="Weekly seasonality: average daily sales",
                sort=(("dim_date", "day_of_week_name"), "Ascending"),
            ),
            visual(
                "ovStores",
                "clusteredBarChart",
                (800, ROW_CHARTS + 218, 460, 200),
                roles={"Category": [column("dim_store", "store_name")], "Y": [measure("fact_sales", "Total Sales")]},
                title="Sales by store (right-click a bar to drill through)",
            ),
        ],
    },
    {
        "name": "forecast",
        "display": "Forecast Accuracy",
        "visuals": [
            textbox("fcTitle", "Forecast Accuracy (90-day test window)", (20, 8, 660, 44)),
            navigator("fcNav"),
            slicer("fcStore", "dim_store", "store_name", (20, ROW_SLICERS, 240, 62), mode="Dropdown", title="Store"),
            slicer("fcItem", "dim_item", "item_name", (280, ROW_SLICERS, 240, 62), mode="Dropdown", title="Item"),
            slicer("fcAbc", "dim_item", "abc_class", (540, ROW_SLICERS, 300, 62), horizontal=True, title="ABC class"),
            *card_row(
                "fc",
                [
                    ("fact_forecast", "Forecast Accuracy"),
                    ("fact_forecast", "WAPE"),
                    ("fact_forecast", "Baseline WAPE"),
                    ("fact_forecast", "Forecast Bias"),
                    ("fact_forecast", "Actual Units"),
                    ("fact_forecast", "Forecast Units"),
                ],
            ),
            visual(
                "fcTrend",
                "lineChart",
                (20, ROW_INSIGHT, 1240, 250),
                roles={
                    "Category": [column("dim_date", "date")],
                    "Y": [
                        measure("fact_forecast", "Actual Units"),
                        measure("fact_forecast", "Forecast Units"),
                        measure("fact_forecast", "Baseline Units"),
                    ],
                },
                title=("fact_forecast", "Forecast Chart Title"),
            ),
            visual(
                "fcItems",
                "clusteredColumnChart",
                (20, 492, 700, 218),
                roles={
                    "Category": [column("dim_item", "item_name")],
                    "Y": [measure("fact_forecast", "WAPE"), measure("fact_forecast", "Baseline WAPE")],
                },
                title="WAPE by item: LightGBM vs baseline (click a bar to filter)",
            ),
            table(
                "fcTable",
                (740, 492, 520, 218),
                [
                    column("dim_store", "store_name"),
                    measure("fact_forecast", "Actual Units"),
                    measure("fact_forecast", "Forecast Units"),
                    measure("fact_forecast", "WAPE"),
                    measure("fact_forecast", "Forecast Bias"),
                ],
                "Accuracy by store (green = WAPE under 12%)",
                color_rules=[("fact_forecast", "WAPE", "WAPE Color")],
            ),
        ],
    },
    {
        "name": "inventory",
        "display": "Inventory & Allocation",
        "visuals": [
            textbox("invTitle", "Inventory Allocation & What-if", (20, 8, 660, 44)),
            navigator("invNav"),
            slicer("invAbc", "dim_item", "abc_class", (20, ROW_SLICERS, 260, 62), horizontal=True, title="ABC class"),
            slicer("invStore", "dim_store", "store_name", (300, ROW_SLICERS, 220, 62), mode="Dropdown", title="Store"),
            slicer(
                "invService",
                "Service Level",
                "Service Level",
                (540, ROW_SLICERS, 350, 62),
                mode="Single",
                title="What-if: service level",
            ),
            slicer(
                "invLead",
                "Lead Time",
                "Lead Time",
                (910, ROW_SLICERS, 350, 62),
                mode="Single",
                title="What-if: lead time (days)",
            ),
            *card_row(
                "inv",
                [
                    ("fact_inventory", "Fill Rate (Optimised)"),
                    ("fact_inventory", "Fill Rate (Equal Split)"),
                    ("fact_inventory", "Worst Fill Rate (Optimised)"),
                    ("fact_inventory", "Unmet Demand (Optimised)"),
                    ("fact_inventory", "Safety Stock (What-if)"),
                    ("fact_inventory", "Reorder Point (What-if)"),
                ],
            ),
            insight_card("invInsight", "fact_inventory", "Inventory Insight", (20, ROW_INSIGHT, 1240, 50)),
            visual(
                "invStrategy",
                "clusteredColumnChart",
                (20, ROW_CHARTS, 600, 418),
                roles={
                    "Category": [column("dim_store", "store_name")],
                    "Y": [
                        measure("fact_inventory", "Unmet Demand (Optimised)"),
                        measure("fact_inventory", "Unmet Demand (Proportional)"),
                        measure("fact_inventory", "Unmet Demand (Equal Split)"),
                    ],
                },
                title="Unmet demand by store and allocation strategy",
            ),
            table(
                "invTable",
                (640, ROW_CHARTS, 620, 418),
                [
                    column("dim_item", "item_name"),
                    column("dim_item", "abc_class"),
                    measure("fact_inventory", "Inventory Demand"),
                    measure("fact_inventory", "Allocated Supply"),
                    measure("fact_inventory", "Fill Rate (Optimised)"),
                    measure("fact_inventory", "Safety Stock (What-if)"),
                    measure("fact_inventory", "Reorder Point (What-if)"),
                ],
                "Item replenishment plan (moves with the what-if sliders)",
                color_rules=[("fact_inventory", "Fill Rate (Optimised)", "Fill Rate Color")],
            ),
        ],
    },
    {
        "name": "detail",
        "display": "Store-Item Detail",
        "drillthrough": [("dim_store", "store_name"), ("dim_item", "item_name")],
        "visuals": [
            back_button("dtBack"),
            textbox("dtTitle", "Store-Item Detail (drill-through)", (70, 8, 700, 44)),
            *card_row(
                "dt",
                [
                    ("fact_forecast", "Actual Units"),
                    ("fact_forecast", "Forecast Units"),
                    ("fact_forecast", "WAPE"),
                    ("fact_forecast", "Forecast Bias"),
                    ("fact_inventory", "Fill Rate (Optimised)"),
                    ("fact_inventory", "Reorder Point (What-if)"),
                ],
                y=60,
            ),
            visual(
                "dtTrend",
                "lineChart",
                (20, 165, 1240, 320),
                roles={
                    "Category": [column("dim_date", "date")],
                    "Y": [
                        measure("fact_forecast", "Actual Units"),
                        measure("fact_forecast", "Forecast Units"),
                        measure("fact_forecast", "Baseline Units"),
                    ],
                },
                title=("fact_forecast", "Forecast Chart Title"),
            ),
            table(
                "dtTable",
                (20, 495, 1240, 215),
                [
                    column("dim_store", "store_name"),
                    column("dim_item", "item_name"),
                    measure("fact_inventory", "Inventory Demand"),
                    measure("fact_inventory", "Allocated Supply"),
                    measure("fact_inventory", "Unmet Demand (Optimised)"),
                    measure("fact_inventory", "Fill Rate (Optimised)"),
                    measure("fact_inventory", "Safety Stock (What-if)"),
                    measure("fact_inventory", "Reorder Point (What-if)"),
                ],
                "Allocation and replenishment",
                color_rules=[("fact_inventory", "Fill Rate (Optimised)", "Fill Rate Color")],
            ),
        ],
    },
]


def page_definition(page):
    definition = {
        "$schema": f"{SCHEMA_ROOT}/page/1.0.0/schema.json",
        "name": page["name"],
        "displayName": page["display"],
        "displayOption": "FitToPage",
        "height": 720,
        "width": 1280,
    }
    if page.get("drillthrough"):
        definition["pageBinding"] = {
            "name": stable_id(page["name"], "binding"),
            "type": "Drillthrough",
            "referenceScope": "Default",
            "parameters": [
                {
                    "name": f"{page['name']}_{entity}_{column_name}_parameter",
                    "boundFilter": f"{page['name']}_{entity}_{column_name}",
                    "fieldExpr": field_ref(entity, column_name, "Column"),
                }
                for entity, column_name in page["drillthrough"]
            ],
        }
        definition["filterConfig"] = {
            "filters": [
                {
                    "name": f"{page['name']}_{entity}_{column_name}",
                    "field": field_ref(entity, column_name, "Column"),
                    "type": "Categorical",
                    "howCreated": "Drillthrough",
                }
                for entity, column_name in page["drillthrough"]
            ]
        }
        definition["visibility"] = "HiddenInViewMode"
    return definition


def build_report(report_directory):
    definition = report_directory / "definition"
    write(
        report_directory / "definition.pbir",
        {"version": "4.0", "datasetReference": {"byPath": {"path": f"../{PROJECT_NAME}.SemanticModel"}}},
    )
    write(definition / "version.json", {"$schema": f"{SCHEMA_ROOT}/versionMetadata/1.0.0/schema.json", "version": "2.0.0"})
    write(
        definition / "report.json",
        {
            "$schema": f"{SCHEMA_ROOT}/report/1.0.0/schema.json",
            "layoutOptimization": "None",
            "themeCollection": {
                "customTheme": {"name": "DemandTheme.json", "reportVersionAtImport": "5.55", "type": "RegisteredResources"}
            },
            "resourcePackages": [
                {
                    "name": "RegisteredResources",
                    "type": "RegisteredResources",
                    "items": [{"name": "DemandTheme.json", "path": "DemandTheme.json", "type": "CustomTheme"}],
                }
            ],
        },
    )
    write(
        report_directory / "StaticResources" / "RegisteredResources" / "DemandTheme.json",
        {
            "name": "Demand Forecasting",
            "dataColors": ["#0096D6", "#F28C28", "#7A7A7A", "#2E7D32", "#C62828", "#6A1B9A"],
            "background": "#FFFFFF",
            "foreground": "#1F2937",
            "tableAccent": "#0096D6",
        },
    )
    write(
        definition / "pages" / "pages.json",
        {
            "$schema": f"{SCHEMA_ROOT}/pagesMetadata/1.0.0/schema.json",
            "pageOrder": [page["name"] for page in PAGES],
            "activePageName": PAGES[0]["name"],
        },
    )
    for page in PAGES:
        page_directory = definition / "pages" / page["name"]
        write(page_directory / "page.json", page_definition(page))
        for tab_order, visual_definition in enumerate(page["visuals"]):
            visual_definition["position"]["tabOrder"] = tab_order
            visual_definition["position"]["z"] = tab_order
            write(page_directory / "visuals" / visual_definition["name"] / "visual.json", visual_definition)


model_directory = POWERBI_DIRECTORY / f"{PROJECT_NAME}.SemanticModel"
report_directory = POWERBI_DIRECTORY / f"{PROJECT_NAME}.Report"
for directory in (model_directory, report_directory):
    if directory.exists():
        shutil.rmtree(directory)
build_semantic_model(model_directory)
build_report(report_directory)
write(
    POWERBI_DIRECTORY / f"{PROJECT_NAME}.pbip",
    {
        "version": "1.0",
        "artifacts": [{"report": {"path": f"{PROJECT_NAME}.Report"}}],
        "settings": {"enableAutoRecovery": True},
    },
)
print(f"Power BI project written to {POWERBI_DIRECTORY / (PROJECT_NAME + '.pbip')}")
print(f"DataFolder parameter: {DATA_FOLDER}")
