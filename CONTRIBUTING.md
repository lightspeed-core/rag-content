# CONTRIBUTING

<!-- the following line is used by tool to autogenerate Table of Content when the document is changed -->
<!-- vim-markdown-toc GFM -->

* [TLDR;](#tldr)
* [Prerequisites](#prerequisites)
    * [Tooling installation](#tooling-installation)
* [Setting up your development environment](#setting-up-your-development-environment)
* [PR description](#pr-description)
* [Pull request size](#pull-request-size)
* [Definition of Done](#definition-of-done)
    * [A deliverable is to be considered “done” when](#a-deliverable-is-to-be-considered-done-when)
* [AI assistants](#ai-assistants)
    * [“Mark” code with substantial AI-generated portions.](#mark-code-with-substantial-ai-generated-portions)
    * [Copyright and licence notices](#copyright-and-licence-notices)
* [Automation](#automation)
    * [Pre-commit hook settings](#pre-commit-hook-settings)
* [Code coverage measurement](#code-coverage-measurement)
* [Linters](#linters)
    * [Type hints checks](#type-hints-checks)
    * [Ruff](#ruff)
    * [Pylint](#pylint)
    * [Security checks](#security-checks)
* [Code style](#code-style)
    * [Function Standards](#function-standards)
        * [Documentation](#documentation)
        * [Type annotations](#type-annotations)
        * [Naming conventions](#naming-conventions)
        * [Async functions](#async-functions)
        * [Error handling](#error-handling)
    * [Formatting rules](#formatting-rules)
    * [Docstrings style](#docstrings-style)

<!-- vim-markdown-toc -->

## TLDR;

1. Create your own fork of the repo
2. Make changes to the code in your fork
3. Run unit tests and integration tests
4. Check the code with linters
5. Submit PR from your fork to main branch of the project repo


## Prerequisites

- git
- Python 3.12 or 3.13
- pip

The development requires at least [Python 3.12](https://docs.python.org/3/whatsnew/3.12.html) due to significant improvement on performance, optimizations which benefit modern ML, AI, LLM, NL stacks, and improved asynchronous processing capabilities. It is also possible to use Python 3.13.



### Tooling installation

1. `pip install --user uv`
1. `uv --version` -- should return no error



## Setting up your development environment

```bash
# clone your fork
git clone https://github.com/YOUR-GIT-PROFILE/rag-content.git

# move into the directory
cd rag-content

# setup your devel environment with uv
uv sync --group dev

# Now you can run test commands trough make targets, or prefix the rest of commands with `uv run`, eg. `uv run make test` or do `uv venv`, which creates virtual environment and prints activation command, and run commands inside venv.

# code formatting
# (this is also run automatically as part of pre-commit hook if configured)
make format

# code style and docstring style
# (this is also run automatically as part of pre-commit hook if configured)
make verify

# check type hints
# (this is also run automatically as part of pre-commit hook)
make check-types
```

Happy hacking!


## PR description

* Jira ticket needs to be added into PR title, for example: `LCORE-740: type hints for models unit tests`
* Fill-in all relevant information in the PR template
    - unused parts of PR template (like information about testing etc.) can be deleted
* Please note that CodeRabbitAI will create a summary of your pull request


## Pull request size

* Keep pull requests small. Aim for about 200 lines when possible and generally
  stay under 500 lines. Research shows smaller PRs are reviewed and merged more
  quickly, reduce reviewer cognitive load, make testing easier, and lower the
  risk of introducing bugs. If a change must be larger, break it into a clear
  sequence of smaller commits or dependent PRs, include a concise summary of
  the intent and scope, and highlight the critical files or areas to review.

* Make each pull request address a single feature or bug fix. Combining
  multiple unrelated changes in one PR makes reviews harder and increases the
  chance of overlooked issues. Break larger work into smaller, self-contained
  units that can be reviewed and merged independently. For bigger tasks,
  sequence dependent PRs, provide clear descriptions of scope and intent, and
  call out which files or behaviors need careful review.

* Keep the count of modified files small in addition to limiting total lines
  changed. Touching many files increases reviewer overhead, makes it harder to
  understand the scope, and raises the chance of missed regressions. When a
  change necessarily spans multiple files, group related edits into logical,
  incremental PRs; move large, nonfunctional refactors into separate commits or
  PRs; and add a clear summary and guided review notes that point reviewers to
  the most important files and risk areas.

* When you split a larger task into several smaller PRs, craft detailed commit
  messages that explain what each change does and how it fits into the broader
  effort. Include the rationale, any trade-offs, and links or references to
  related PRs or tickets. Good messages preserve context across multiple
  submissions, speed up reviews, and make it easier for future maintainers to
  trace the evolution of the code.

* Ensure each pull request is a self-contained unit that can be merged
  independently. An atomic PR has a single, clear objective—such as fixing a
  bug or adding one feature—and includes only the changes necessary to achieve
  that goal. Avoid coupling the PR to other unmerged branches or relying on
  external context; if work must be split, sequence dependent PRs so each stage
  is reviewable on its own. Don’t mix unrelated change types (for example, bug
  fixes, refactors, and new features) in the same PR. When appropriate,
  separate large refactors or formatting changes into their own PRs, and add
  concise descriptions that state the PR’s intent and any necessary migration
  or rollout steps.



## Definition of Done

### A deliverable is to be considered “done” when

* Code is complete, commented, and merged to the relevant release branch
* User facing documentation written (where relevant)
* Acceptance criteria in the related Jira ticket (where applicable) are verified and fulfilled
* Pull request title+commit includes Jira number
* Changes are covered by unit tests that run cleanly in the CI environment (where relevant)
* Changes are covered by integration tests that run cleanly in the CI environment (where relevant)
* Changes are covered by E2E tests that run cleanly in the CI environment (where relevant)
* All linters are running cleanly in the CI environment
* Code changes reviewed by at least one peer
* Code changes acked by at least one project owner

## AI assistants

### “Mark” code with substantial AI-generated portions.

Nontrivial and substantial AI-generated or AI-assisted content should be
“marked” in appropriate cases. In deciding how to approach this, consider
adopting one or more of the following recommendations. (This assumes you have
not concluded that a suggestion is a match to some existing third-party code.) 

In a commit message, or in a pull request/merge request description field,
identify the code assistant that you used, perhaps elaborating on how it was
used. You may wish to use a trailer like “Assisted-by:” or “Generated-by:”. For
example:

```
Assisted-by: <name of code assistant>
```

In a source file comment, indicate the use of the code assistant. For example:

```
Generated by: <name of code assistant>
```

### Copyright and licence notices

If the contents of an entire file or files in PR were substantially generated
by a code assistant with little to no creative input or modification by you
(which should typically not be the case), copyright protection may be limited,
but it is particularly appropriate to mark the contents of the file as
recommended above.


### Formatting rules

Code formatting rules are checked by __Black__. More info can be found on [https://black.readthedocs.io/en/stable/](https://black.readthedocs.io/en/stable/).

### Docstrings style
We are using [Google's docstring style](https://google.github.io/styleguide/pyguide.html).

Here is simple example:
```python
def function_with_pep484_type_annotations(param1: int, param2: str) -> bool:
    """Example function with PEP 484 type annotations.
    
    Args:
        param1: The first parameter.
        param2: The second parameter.
    
    Returns:
        The return value. True for success, False otherwise.
    
    Raises:
        ValueError: If the first parameter does not contain proper model name
    """
```

For further guidance, see the rest of our codebase, or check sources online. There are many, e.g. [this one](https://gist.github.com/redlotus/3bc387c2591e3e908c9b63b97b11d24e).


