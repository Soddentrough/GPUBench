#!/usr/bin/env python3
"""Format GPUBench JSON benchmark results into pretty terminal tables using rich."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def format_status(status: str) -> Text:
    status_lower = status.lower()
    if status_lower in ("completed", "passed", "success", "ok"):
        return Text(status.upper(), style="bold green")
    if status_lower in ("unsupported", "skipped"):
        return Text(status.upper(), style="dim yellow")
    if status_lower in ("failed", "error"):
        return Text(status.upper(), style="bold red")
    return Text(status, style="cyan")


def format_value(val_str: str, status: str) -> Text:
    if status.lower() == "unsupported":
        return Text("N/A", style="dim italic")
    return Text(str(val_str), style="bold cyan")


def display_detailed_results(data: dict[str, Any], console: Console) -> None:
    timestamp = data.get("timestamp")
    time_str = (
        datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        if timestamp
        else "Unknown"
    )
    backend = data.get("backend", "Unknown")

    header_text = Text()
    header_text.append("Timestamp: ", style="bold")
    header_text.append(f"{time_str} ({timestamp})\n", style="white")
    header_text.append("Backend:   ", style="bold")
    header_text.append(f"{backend}\n", style="magenta")

    devices = data.get("devices", [])
    if devices:
        header_text.append("Devices:   ", style="bold")
        header_text.append(", ".join(devices), style="green")

    console.print(
        Panel(
            header_text,
            title="[bold yellow]GPUBench Execution Summary[/bold yellow]",
            border_style="bright_blue",
            expand=False,
        )
    )

    profiles = data.get("device_profiles", [])
    if profiles:
        prof_table = Table(
            title="[bold green]Detected GPU Profiles[/bold green]",
            box=ROUNDED,
            header_style="bold green",
            show_lines=True,
        )
        prof_table.add_column("GPU", style="bold cyan")
        prof_table.add_column("PCI / Backend", style="white")
        prof_table.add_column("Driver & API", style="dim")
        prof_table.add_column("VRAM & Workgroup", style="yellow")
        prof_table.add_column("Hardware Capabilities", style="magenta")

        for p in profiles:
            dev_str = f"{p.get('device_name', 'Unknown')}\n(Index: {p.get('device_index', 0)})"
            pci_str = f"Backend: {p.get('backend', '-')}\nVendor: {p.get('vendor_id', '-')}\nDevice: {p.get('device_id', '-')}"
            drv_str = f"Driver: {p.get('driver_name', '-')}\nInfo: {p.get('driver_info', '-')}\nVer: {p.get('driver_version', '-')}\nAPI: {p.get('api_version', '-')}"
            mem_str = f"VRAM: {p.get('vram_total_mb', 0)} MB\nSubgroup: {p.get('subgroup_size', '-')}\nMax WG: {p.get('max_workgroup_size', '-')}"
            
            caps = []
            caps.append(f"Ray Tracing: {'[bold green]Yes[/bold green]' if p.get('ray_tracing_supported') else '[dim]No[/dim]'}")
            caps.append(f"Hardware SER: {'[bold green]Yes[/bold green]' if p.get('ser_supported') else '[dim]No[/dim]'}")
            caps.append(f"Work Graphs: {'[bold green]Yes[/bold green]' if p.get('work_graphs_supported') else '[dim]No[/dim]'}")
            caps.append(f"Coop Matrix: {'[bold green]Yes[/bold green]' if p.get('cooperative_matrix_supported') else '[dim]No[/dim]'}")
            caps.append(f"FP16: {'[bold green]Yes[/bold green]' if p.get('float16_supported') else '[dim]No[/dim]'}, INT8: {'[bold green]Yes[/bold green]' if p.get('int8_supported') else '[dim]No[/dim]'}")
            cap_str = "\n".join(caps)

            prof_table.add_row(dev_str, pci_str, drv_str, mem_str, cap_str)

        console.print(prof_table)

    results = data.get("results", [])
    if not results:
        console.print("[yellow]No benchmark results found in JSON data.[/yellow]")
        return

    if results and isinstance(results[0], dict) and "benchmark" in results[0]:
        # CLI ResultData format
        table = Table(
            title="[bold]Benchmark Results[/bold]",
            box=ROUNDED,
            header_style="bold magenta",
            show_lines=False,
        )
        table.add_column("Benchmark", style="bold white")
        table.add_column("Backend", style="cyan")
        table.add_column("Subcategory", style="dim")
        table.add_column("Result / Diagnostic", justify="right", style="bold cyan")
        table.add_column("Status", justify="center", no_wrap=True)

        for r in results:
            bench = r.get("benchmark", "")
            backend_name = r.get("backend", "")
            subcat = r.get("subcategory", "") or r.get("component", "")
            unsupp = r.get("unsupported", False)
            if unsupp:
                status_styled = Text("UNSUPPORTED", style="dim yellow")
                reason = r.get("unsupported_reason", "N/A")
                val_styled = Text(reason, style="dim italic")
            else:
                status_styled = Text("COMPLETED", style="bold green")
                val_str = f"{r.get('value', 0.0):.2f} {r.get('metric', '')}"
                val_styled = Text(val_str, style="bold cyan")
            table.add_row(bench, backend_name, subcat, val_styled, status_styled)

        console.print(table)
        return

    for result in results:
        device_id = result.get("device_id", 0)
        device_name = result.get("device_name", f"Device {device_id}")
        benchmarks = result.get("benchmarks", [])

        console.print(f"\n[bold underline cyan]Device {device_id}: {device_name}[/bold underline cyan]\n")

        # Group benchmarks by category
        categories: dict[str, list[dict[str, Any]]] = {}
        for bm in benchmarks:
            cat = bm.get("category", "GENERAL")
            categories.setdefault(cat, []).append(bm)

        for cat_name, items in categories.items():
            table = Table(
                title=f"[bold]{cat_name}[/bold]",
                box=ROUNDED,
                header_style="bold magenta",
                show_lines=False,
            )

            table.add_column("Benchmark", style="bold white")
            table.add_column("Approach / Implementation", style="dim")
            table.add_column("Result / Score", justify="right", style="bold cyan")
            table.add_column("Status", justify="center", no_wrap=True)

            for item in items:
                label = item.get("label", item.get("id", "Unknown"))
                approach = item.get("approach", "-")
                val = item.get("value", str(item.get("numeric", "-")))
                status = item.get("status", "completed")

                status_styled = format_status(status)
                val_styled = format_value(val, status)

                table.add_row(
                    label,
                    approach,
                    val_styled,
                    status_styled,
                )

            console.print(table)


def display_simple_results(data: dict[str, Any], console: Console) -> None:
    api = data.get("compute_api", "Unknown")
    hardware = data.get("hardware", "Unknown")

    header_text = Text()
    header_text.append("Compute API: ", style="bold")
    header_text.append(f"{api}\n", style="magenta")
    header_text.append("Hardware:    ", style="bold")
    header_text.append(f"{hardware}", style="green")

    console.print(
        Panel(
            header_text,
            title="[bold yellow]GPUBench Summary[/bold yellow]",
            border_style="bright_blue",
            expand=False,
        )
    )

    results = data.get("results", {})
    if not isinstance(results, dict) or not results:
        console.print("[yellow]No benchmark results found in JSON data.[/yellow]")
        return

    for group_name, metrics in results.items():
        table = Table(
            title=f"[bold]{group_name.replace('_', ' ').upper()}[/bold]",
            box=ROUNDED,
            header_style="bold magenta",
            show_lines=False,
        )

        table.add_column("Metric", style="bold white")
        table.add_column("Value", justify="right", style="bold cyan")

        if isinstance(metrics, dict):
            for key, val in metrics.items():
                table.add_row(key, str(val))
        else:
            table.add_row("Value", str(metrics))

        console.print(table)


def display_results(data: dict[str, Any], console: Console) -> None:
    # Determine schema variant:
    # Schema A: has 'results' as a list of device objects (e.g. gpubench_results_<timestamp>.json)
    # Schema B: has 'results' as a dict of categorized metrics (e.g. gpubench_results.json)
    results = data.get("results")
    if isinstance(results, list):
        display_detailed_results(data, console)
    elif isinstance(results, dict):
        display_simple_results(data, console)
    else:
        console.print(Panel(json.dumps(data, indent=2), title="JSON Data"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format GPUBench results JSON into pretty terminal tables."
    )
    parser.add_argument(
        "json_paths",
        nargs="*",
        default=["gpubench_results_1788511996.json"],
        help="Path(s) to GPUBench JSON results file(s)",
    )
    args = parser.parse_args()

    console = Console()

    for path_str in args.json_paths:
        path = Path(path_str)
        if not path.is_file():
            console.print(f"[bold red]Error:[/bold red] File '{path}' not found.")
            continue

        if len(args.json_paths) > 1:
            console.rule(f"[bold cyan]{path.name}[/bold cyan]")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            display_results(data, console)
        except Exception as e:
            console.print(f"[bold red]Error reading '{path}':[/bold red] {e}")


if __name__ == "__main__":
    main()
