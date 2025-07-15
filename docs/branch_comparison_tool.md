# Branch Comparison Tool

This tool compares two git branches with different commit histories but potentially the same file tree structure. It generates tasks for every difference found, including added, deleted, and modified files.

## Features

- **Comprehensive Comparison**: Compares all files between two branches
- **Task Generation**: Creates structured tasks for each difference found
- **Multiple Change Types**: Handles added, deleted, and modified files
- **Detailed Diff Content**: Provides diff content for each change
- **JSON Output**: Saves tasks in structured JSON format for further processing
- **Priority Assignment**: Assigns priorities based on change type
- **Verbose Mode**: Optional detailed output for debugging

## Installation

The tool is part of the KBLaM package. Ensure you have installed the package:

```bash
pip install -e .
```

## Usage

### Basic Usage

```bash
python compare_branches.py <branch1> <branch2>
```

Example:
```bash
python compare_branches.py julian import-repoB
```

### Command Line Options

- `branch1`: First branch name (e.g., 'julian')
- `branch2`: Second branch name (e.g., 'import-repoB')
- `--repo-path`: Path to the git repository (default: current directory)
- `--output`: Output file for tasks (default: branch_comparison_tasks.json)
- `--verbose`: Enable verbose output with detailed diff content

### Advanced Usage

```bash
# Compare branches with custom output file
python compare_branches.py julian import-repoB --output my_comparison.json

# Enable verbose mode for detailed diff content
python compare_branches.py julian import-repoB --verbose

# Specify a different repository path
python compare_branches.py julian import-repoB --repo-path /path/to/repo
```

## Output Format

The tool generates a JSON file containing an array of task objects. Each task has the following structure:

```json
{
  "task_id": "DIFF-0001",
  "title": "New file added: example.py",
  "description": "Detailed description of the change...",
  "file_path": "path/to/file.py",
  "change_type": "added|deleted|modified",
  "priority": "high|medium|low",
  "diff_content": "Actual diff content or file content"
}
```

### Change Types

1. **Added**: Files that exist in branch2 but not in branch1
2. **Deleted**: Files that exist in branch1 but not in branch2
3. **Modified**: Files that exist in both branches but have different content

### Priority Levels

- **High**: Added and deleted files (require immediate attention)
- **Medium**: Modified files (require review and merging)

## Example Output

```bash
$ python compare_branches.py julian import-repoB

Comparing branches 'julian' and 'import-repoB'...
Found 4 differences
Tasks saved to: branch_comparison_tasks.json

=== Branch Comparison Summary ===
Total tasks created: 4

Tasks by change type:
  added: 2
  deleted: 1
  modified: 1

Tasks by priority:
  high: 3
  medium: 1

=== Task Details ===

[DIFF-0001] New file added: import_readme.md
  Priority: high
  Type: added
  File: import_readme.md

[DIFF-0002] New file added: src/kblam/import_feature.py
  Priority: high
  Type: added
  File: src/kblam/import_feature.py

[DIFF-0003] File deleted: to_be_deleted.txt
  Priority: high
  Type: deleted
  File: to_be_deleted.txt

[DIFF-0004] File modified: src/kblam/julian_feature.py
  Priority: medium
  Type: modified
  File: src/kblam/julian_feature.py
```

## Integration with Development Workflow

The generated tasks can be integrated into various development workflows:

1. **Manual Review**: Use the JSON output to manually review each change
2. **Automated Processing**: Parse the JSON to automatically apply certain types of changes
3. **Issue Tracking**: Import tasks into issue tracking systems
4. **Code Review**: Use the diff content for code review processes

## Error Handling

The tool handles various error conditions:

- **Missing Branches**: Validates that both branches exist before comparison
- **Network Issues**: Works entirely with local git repository
- **Permission Issues**: Handles git access permission problems
- **Large Files**: Handles large files efficiently by using git commands

## Limitations

- Requires git repository with the specified branches
- Does not handle binary files content comparison (shows as modified only)
- Requires appropriate git permissions for branch access
- Large repositories may take longer to process

## Testing

Run the test suite to verify functionality:

```bash
python -m pytest tests/test_branch_comparator.py -v
```

## Contributing

This tool is part of the KBLaM project. Please follow the project's contribution guidelines when making changes.