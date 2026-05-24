"""
Steam-specific utility functions.

Handles:
  - Parsing Python2-style dict files (ast.literal_eval)
  - Genre tag normalization
  - Playtime normalization
  - Item text representation construction
"""

import ast
import re
from typing import Dict, List, Optional, Any


def parse_steam_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single line from Steam JSON file (Python2 dict format with u'').

    Returns dict or None if parsing fails.
    """
    line = line.strip()
    if not line:
        return None
    try:
        return ast.literal_eval(line)
    except (ValueError, SyntaxError):
        # Try replacing Python2-isms
        try:
            # Replace u'...' with '...'
            fixed = re.sub(r"\bu'([^']*)'", r"'\1'", line)
            fixed = re.sub(r'\bu"([^"]*)"', r'"\1"', fixed)
            fixed = fixed.replace('True', 'true').replace('False', 'false').replace('None', 'null')
            import json
            return json.loads(fixed)
        except (json.JSONDecodeError, ValueError):
            pass
        return None


def parse_steam_file(filepath: str, max_lines: Optional[int] = None) -> List[Dict[str, Any]]:
    """Parse entire Steam data file. Returns list of dicts."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            record = parse_steam_line(line)
            if record:
                records.append(record)
            if i % 500000 == 0 and i > 0:
                print(f"  Parsed {i} lines, {len(records)} valid records")
    return records


def normalize_genres(genres: List[str]) -> List[str]:
    """Normalize Steam genre tags."""
    # Remove duplicates while preserving order, strip whitespace
    seen = set()
    result = []
    for g in genres:
        g = g.strip()
        if g and g not in seen:
            seen.add(g)
            result.append(g)
    return result


def build_item_text(item: Dict[str, Any], include_description: bool = False) -> str:
    """Build a text representation of a Steam game for LLM/embedding use.

    Format: "Title: {title} | Genres: {genres} | Tags: {tags} | Developer: {dev} | Price: {price}"
    """
    title = item.get('title', item.get('app_name', 'Unknown'))
    genres = ', '.join(item.get('genres', [])) if item.get('genres') else 'Unknown'
    tags = ', '.join(item.get('tags', [])[:10]) if item.get('tags') else ''
    developer = item.get('developer', 'Unknown')
    price = item.get('price', 'N/A')

    parts = [f"Title: {title}", f"Genres: {genres}"]
    if tags:
        parts.append(f"Tags: {tags}")
    parts.append(f"Developer: {developer}")
    parts.append(f"Price: {price}")

    return " | ".join(parts)


def normalize_hours(hours: float, max_hours: float = 10000.0) -> float:
    """Log-normalize playtime hours, cap at max_hours."""
    import math
    hours = min(float(hours), max_hours)
    return math.log1p(hours)


def timestamp_to_unix(date_str: str) -> Optional[float]:
    """Convert Steam date string to unix timestamp.

    Handles formats: '2018-01-04', 'Posted November 5, 2011.'
    """
    import time
    from datetime import datetime

    # Try ISO format first
    try:
        dt = datetime.strptime(date_str.strip(), '%Y-%m-%d')
        return dt.timestamp()
    except ValueError:
        pass

    # Try "Posted Month DD, YYYY." format
    match = re.match(r'Posted\s+(\w+)\s+(\d+),?\s*(\d{4})', date_str.strip())
    if match:
        month_str, day, year = match.groups()
        try:
            dt = datetime.strptime(f'{month_str} {day} {year}', '%B %d %Y')
            return dt.timestamp()
        except ValueError:
            pass

    # Try other common formats
    for fmt in ['%Y-%m-%d %H:%M:%S', '%m/%d/%Y', '%B %d, %Y']:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.timestamp()
        except ValueError:
            continue

    return None
