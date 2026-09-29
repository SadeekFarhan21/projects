#!/usr/bin/env python3
"""WakaTime MCP Server — exposes WakaTime API endpoints as MCP tools."""

import base64
import os
from typing import Optional

import requests
from mcp.server.fastmcp import FastMCP

WAKATIME_API_BASE = "https://api.wakatime.com/api/v1"

mcp = FastMCP("WakaTime")


def _auth_headers() -> dict[str, str]:
    api_key = os.environ.get("WAKATIME_API_KEY")
    if not api_key:
        raise RuntimeError("WAKATIME_API_KEY environment variable is not set")
    encoded = base64.b64encode(api_key.encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


def _get(path: str, params: Optional[dict] = None) -> dict:
    url = f"{WAKATIME_API_BASE}{path}"
    resp = requests.get(url, headers=_auth_headers(), params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------

@mcp.tool()
def get_current_user() -> dict:
    """Get the authenticated user's profile (name, timezone, plan, etc.)."""
    return _get("/users/current")


# ---------------------------------------------------------------------------
# Coding time overview
# ---------------------------------------------------------------------------

@mcp.tool()
def get_all_time_stats(project: Optional[str] = None) -> dict:
    """Total coding time logged since the account was created.

    Args:
        project: Optional project name to narrow results to a single project.
    """
    params: dict = {}
    if project:
        params["project"] = project
    return _get("/users/current/all_time_since_today", params or None)


@mcp.tool()
def get_status_bar_today() -> dict:
    """Today's coding activity — same data shown in IDE status bars."""
    return _get("/users/current/status_bar/today")


# ---------------------------------------------------------------------------
# Stats (aggregated by range)
# ---------------------------------------------------------------------------

@mcp.tool()
def get_stats(range: str = "last_7_days") -> dict:
    """Aggregated coding stats for a time range.

    Args:
        range: One of last_7_days, last_30_days, last_6_months, last_year,
               all_time, a 4-digit year (e.g. "2024"), or a year-month
               (e.g. "2024-03").
    """
    return _get(f"/users/current/stats/{range}")


# ---------------------------------------------------------------------------
# Summaries (daily breakdown)
# ---------------------------------------------------------------------------

@mcp.tool()
def get_summaries(
    start: str,
    end: str,
    project: Optional[str] = None,
    branches: Optional[str] = None,
    timezone: Optional[str] = None,
) -> dict:
    """Daily coding summaries for a date range.

    Args:
        start: Start date in YYYY-MM-DD format.
        end:   End date in YYYY-MM-DD format.
        project:  Filter to a single project.
        branches: Comma-separated branch names to filter by.
        timezone: Olson timezone string (defaults to user's timezone).
    """
    params: dict = {"start": start, "end": end}
    if project:
        params["project"] = project
    if branches:
        params["branches"] = branches
    if timezone:
        params["timezone"] = timezone
    return _get("/users/current/summaries", params)


# ---------------------------------------------------------------------------
# Durations
# ---------------------------------------------------------------------------

@mcp.tool()
def get_durations(
    date: str,
    project: Optional[str] = None,
    branches: Optional[str] = None,
    timezone: Optional[str] = None,
    slice_by: Optional[str] = None,
) -> dict:
    """Coding durations for a single day (heartbeats joined into sessions).

    Args:
        date:     Date in YYYY-MM-DD format.
        project:  Filter to a single project.
        branches: Comma-separated branch names.
        timezone: Olson timezone string.
        slice_by: Group durations by this key. One of: project, entity,
                  language, dependencies, os, editor, category, machine.
    """
    params: dict = {"date": date}
    if project:
        params["project"] = project
    if branches:
        params["branches"] = branches
    if timezone:
        params["timezone"] = timezone
    if slice_by:
        params["slice_by"] = slice_by
    return _get("/users/current/durations", params)


# ---------------------------------------------------------------------------
# Heartbeats
# ---------------------------------------------------------------------------

@mcp.tool()
def get_heartbeats(date: str) -> dict:
    """Raw heartbeats sent by IDE plugins for a specific day.

    Args:
        date: Date in YYYY-MM-DD format.
    """
    return _get("/users/current/heartbeats", {"date": date})


# ---------------------------------------------------------------------------
# Projects & commits
# ---------------------------------------------------------------------------

@mcp.tool()
def get_projects(query: Optional[str] = None) -> dict:
    """List WakaTime projects for the current user.

    Args:
        query: Optional search string to filter project names.
    """
    params: dict = {}
    if query:
        params["q"] = query
    return _get("/users/current/projects", params or None)


@mcp.tool()
def get_commits(
    project: str,
    author: Optional[str] = None,
    branch: Optional[str] = None,
    page: Optional[int] = None,
) -> dict:
    """Commits for a WakaTime project with time spent coding per commit.

    Args:
        project: Project name (must match exactly).
        author:  Filter by author username.
        branch:  Filter by branch name.
        page:    Page number for pagination.
    """
    params: dict = {}
    if author:
        params["author"] = author
    if branch:
        params["branch"] = branch
    if page is not None:
        params["page"] = page
    return _get(f"/users/current/projects/{project}/commits", params or None)


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------

@mcp.tool()
def get_goals() -> dict:
    """List the current user's coding goals and their pass/fail status."""
    return _get("/users/current/goals")


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------

@mcp.tool()
def get_insights(insight_type: str, range: str = "last_7_days") -> dict:
    """Detailed insights about coding habits for a time range.

    Args:
        insight_type: One of: weekdays, days, ai_days, best_day,
                      daily_average, projects, languages, editors,
                      categories, machines, operating_systems.
        range: One of: last_7_days, last_30_days, last_6_months, last_year.
    """
    return _get(f"/users/current/insights/{insight_type}/{range}")


# ---------------------------------------------------------------------------
# Leaderboards
# ---------------------------------------------------------------------------

@mcp.tool()
def get_leaders(
    language: Optional[str] = None,
    country_code: Optional[str] = None,
    page: Optional[int] = None,
) -> dict:
    """Public leaderboard — users ranked by coding activity.

    Args:
        language:     Filter by programming language.
        country_code: Two-character ISO country code filter.
        page:         Page number for pagination.
    """
    params: dict = {}
    if language:
        params["language"] = language
    if country_code:
        params["country_code"] = country_code
    if page is not None:
        params["page"] = page
    return _get("/leaders", params or None)


# ---------------------------------------------------------------------------
# Machines & metadata
# ---------------------------------------------------------------------------

@mcp.tool()
def get_machine_names() -> dict:
    """List machines (computers) the user has logged activity from."""
    return _get("/users/current/machine_names")


@mcp.tool()
def get_editors() -> dict:
    """List all WakaTime IDE plugins and their latest versions."""
    return _get("/editors")


@mcp.tool()
def get_program_languages() -> dict:
    """List all programming languages supported by WakaTime."""
    return _get("/program_languages")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
