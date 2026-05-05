---
name: file-dependabot-cves
description: Fetch Dependabot alerts, cross-reference against LCORE Jira tickets, and file tickets for gaps
---

Audit Dependabot vulnerabilities for `$repo` (default: `lightspeed-core/rag-content`) and cross-reference them against existing LCORE Jira tickets.

## Step 1: Fetch Dependabot alerts

Fetch open Dependabot alerts using:

```
gh api "repos/$repo/dependabot/alerts" --paginate --jq '.[] | select(.state == "open") | {number, state, severity: .security_vulnerability.severity, package: .security_vulnerability.package.name, ecosystem: .security_vulnerability.package.ecosystem, summary: .security_advisory.summary, cve: (.security_advisory.cve_id // "N/A"), created: .created_at, fixed_version: (.security_vulnerability.first_patched_version.identifier // "N/A")}'
```

If `$repo` is not provided, default to `lightspeed-core/rag-content`. Deduplicate results by CVE + package name (multiple alerts for the same CVE across different manifest files count as one unique vulnerability).

## Step 2: Present severity summary

Present a summary table with counts by severity (Critical, High, Medium, Low), then a breakdown table grouped by package: Package | Severity | CVE | Summary | Fix Version.

## Step 3: Search LCORE Jira for existing coverage

!Requires JIRA Atlassian MCP

Search LCORE Jira for existing tickets per affected package. Batch package names into OR clauses to minimize API calls:

- By summary: `project = LCORE AND (summary ~ "pkg1" OR summary ~ "pkg2" ...)`
- By CVE label: `project = LCORE AND labels in ("CVE-XXXX-XXXXX", ...)`

Fields: `summary,status,assignee,priority,labels`. Limit: 50. Paginate if needed.

## Step 4: Cross-reference and classify

Cross-reference each Dependabot alert to its LCORE ticket(s) and classify as:
- **Covered**: open/in-progress ticket exists
- **Closed**: ticket done
- **Missing**: no ticket

Present:
1. A coverage table: Vulnerability | Sev. | Dependabot # | LCORE Ticket(s) | Status | Assignee
2. A **gaps table** listing only the missing vulnerabilities with their GitHub alert title (from `security_advisory.summary`), severity, CVE, and fix version
3. Key findings: coverage ratio, unassigned high/critical items, duplicate tickets that could be consolidated

## Step 5: Verify gaps

For each gap, cross-reference in JIRA (full-text search by CVE ID) and GitHub (confirm alert is still open) to verify it is a real missing issue. Drop false positives (e.g., already-closed tickets, stale alerts).

## Step 6: Ask user which gaps to file

Ask the user:
- Whether they want to create LCORE tickets for the missing vulnerabilities
- Which severity levels to include (e.g. "only medium and above", "all", or specific ones)
- The target fix version (look up available versions from `jira_get_project_versions` for LCORE)
- The component to assign (look up available components from `jira_get_project_components` for LCORE)

## Step 7: Fetch full advisory details and draft tickets

For each vulnerability the user wants to file, fetch the full Dependabot advisory description:

```
gh api "repos/$repo/dependabot/alerts/$alert_number" --jq '{summary: .security_advisory.summary, description: .security_advisory.description, cve: (.security_advisory.cve_id // "N/A"), remediation: (.security_vulnerability.first_patched_version.identifier // "No fix available"), vulnerable_range: .security_vulnerability.vulnerable_version_range}'
```

Structure each ticket as:

| Field | Value |
|-------|-------|
| **Project** | LCORE |
| **Type** | Vulnerability |
| **Title** | The original GitHub advisory summary (from `security_advisory.summary`) |
| **Component** | As chosen by user |
| **Fix Version** | As chosen by user |
| **Labels** | The CVE ID (if available), Security |
| **Description** | The full `security_advisory.description` from Dependabot, followed by `**Remediation:** Upgrade <package> to >= <fix_version>` (or "No upstream fix available yet" if no fix exists, including the vulnerable range). |

Present all drafted tickets in a table to the user for review before creating them.

## Step 8: Find the parent CVE epic

Search for the parent epic: `project = LCORE AND issuetype = Epic AND summary ~ "CVE" AND summary ~ "rag-content" ORDER BY created DESC`. Pick the one whose fix version matches the user's chosen fix version. If ambiguous or none found, ask the user.

## Step 9: Create tickets after user confirmation

Only after the user explicitly confirms the drafts, create the tickets using `jira_create_issue` with:
- `project_key`: LCORE
- `issue_type`: Vulnerability
- `summary`: the GitHub advisory title
- `description`: as structured above
- `components`: user's chosen component
- `additional_fields`: `{"fixVersions": [{"id": "<version_id>"}], "labels": ["<CVE-ID>", "Security"], "parent": "<EPIC_KEY>"}`

Omit `parent` only if the user chose to skip it.

Report back the created ticket keys and links.
