# Contributing

Thank you for your interest in contributing to this project.

This document describes the basic rules for contributing to the repository.

## How to contribute

1. Fork the repository.

2. Clone your fork:

```bash
git clone https://github.com/your-username/WHA.git
```

3. Add the original repository as `upstream`:

```bash
git remote add upstream https://github.com/AgushaOS/WHA.git
```

4. Sync your local repository with the original one:

```bash
git fetch upstream
git checkout main
git merge upstream/main
```

5. Create a separate branch for your changes:

```bash
git checkout -b feat/your-change-name
```

6. Make your changes.

7. Check that the project still builds and works correctly. Run tests if they are available.

8. Commit your changes:

```bash
git add <changed-files>
git commit -m "feat: describe your changes"
```

9. Push the branch to your fork:

```bash
git push origin feat/your-change-name
```

10. Open a Pull Request to the original repository.

## Branch naming

Use the following format:

```text
type/short-change-description
```

Examples:

```text
docs/...        - for documentation changes
fix/...         - for bug fixes
feat/...        - for new functionality
refactor/...    - for code changes without changing behavior
test/...        - for tests
chore/...       - for maintenance and miscellaneous changes
```

## Commit naming

Use the following format:

```text
type: short change description
```

Examples:

```text
docs: add contributing guide
fix: correct decoder output
feat: add bitrate comparison
refactor: simplify entropy encoder
test: add decoder tests
chore: update gitignore
```


## Pull Request rules

Before opening a Pull Request, please make sure that:

* your changes are related to one specific task or improvement;
* the code builds successfully;
* unnecessary generated files, temporary files, and local environment files are not included;
* the commit message briefly explains what was changed;
* the Pull Request description explains the purpose of the changes.

## Code style

Please try to keep the existing code style of the project.

When adding new code:

* use clear names for variables, functions, and files;
* avoid unnecessary changes in unrelated files;
* keep formatting consistent with the surrounding code;
* prefer simple and readable solutions.

## Reporting issues

If you found a bug or have an idea for improvement, you can open an Issue.

Please include:

* a short description of the problem or idea;
* steps to reproduce the bug, if applicable;
* expected behavior;
* actual behavior;
* screenshots or logs, if they help explain the problem.

## Notes

Small fixes, documentation improvements, and code cleanup are welcome.

Thank you for helping improve the project!
