#!/usr/bin/env python3
"""
Branch Comparator Tool for KBLaM

This tool compares two git branches with different commit histories but potentially
the same file tree structure. It generates tasks for every difference found.

Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class FileDifference:
    """Represents a difference between two files or file states."""
    
    file_path: str
    change_type: str  # 'added', 'deleted', 'modified'
    diff_content: Optional[str] = None
    old_size: Optional[int] = None
    new_size: Optional[int] = None


@dataclass
class Task:
    """Represents a task to be created for a file difference."""
    
    task_id: str
    title: str
    description: str
    file_path: str
    change_type: str
    priority: str = "medium"
    diff_content: Optional[str] = None


class BranchComparator:
    """Compares two git branches and generates tasks for differences."""
    
    def __init__(self, repo_path: str = "."):
        """Initialize the comparator with a repository path."""
        self.repo_path = Path(repo_path).resolve()
        self.tasks: List[Task] = []
    
    def _run_git_command(self, cmd: List[str]) -> str:
        """Run a git command and return the output."""
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Git command failed: {' '.join(cmd)}\nError: {e.stderr}")
    
    def _ensure_branch_exists(self, branch_name: str) -> bool:
        """Check if a branch exists locally or remotely."""
        try:
            # Check if branch exists locally
            self._run_git_command(["git", "rev-parse", "--verify", branch_name])
            return True
        except RuntimeError:
            try:
                # Check if branch exists remotely
                self._run_git_command(["git", "rev-parse", "--verify", f"origin/{branch_name}"])
                return True
            except RuntimeError:
                return False
    
    def _get_file_list(self, branch: str) -> Set[str]:
        """Get the list of files in a branch."""
        try:
            output = self._run_git_command(["git", "ls-tree", "-r", "--name-only", branch])
            return set(output.split('\n')) if output else set()
        except RuntimeError:
            return set()
    
    def _get_file_diff(self, file_path: str, branch1: str, branch2: str) -> str:
        """Get the diff content for a file between two branches."""
        try:
            diff_output = self._run_git_command([
                "git", "diff", branch1, branch2, "--", file_path
            ])
            return diff_output
        except RuntimeError:
            # Try alternative approach
            try:
                # Get the file content from each branch separately
                content1 = self._get_file_content(file_path, branch1)
                content2 = self._get_file_content(file_path, branch2)
                
                if content1 is None and content2 is not None:
                    return f"File added in {branch2}:\n{content2}"
                elif content1 is not None and content2 is None:
                    return f"File deleted from {branch1}:\n{content1}"
                elif content1 is not None and content2 is not None:
                    # Create a simple diff manually
                    lines1 = content1.split('\n')
                    lines2 = content2.split('\n')
                    diff_lines = []
                    diff_lines.append(f"--- {branch1}:{file_path}")
                    diff_lines.append(f"+++ {branch2}:{file_path}")
                    
                    max_lines = max(len(lines1), len(lines2))
                    for i in range(max_lines):
                        line1 = lines1[i] if i < len(lines1) else ""
                        line2 = lines2[i] if i < len(lines2) else ""
                        
                        if line1 != line2:
                            if line1:
                                diff_lines.append(f"- {line1}")
                            if line2:
                                diff_lines.append(f"+ {line2}")
                        else:
                            diff_lines.append(f"  {line1}")
                    
                    return '\n'.join(diff_lines)
                else:
                    return "Unable to generate diff"
            except RuntimeError:
                return "Unable to generate diff"
    
    def _get_file_content(self, file_path: str, branch: str) -> Optional[str]:
        """Get the content of a file from a specific branch."""
        try:
            content = self._run_git_command(["git", "show", f"{branch}:{file_path}"])
            return content
        except RuntimeError:
            return None
    
    def _get_file_size(self, file_path: str, branch: str) -> Optional[int]:
        """Get the size of a file in a specific branch."""
        try:
            output = self._run_git_command(["git", "cat-file", "-s", f"{branch}:{file_path}"])
            return int(output)
        except (RuntimeError, ValueError):
            return None
    
    def compare_branches(self, branch1: str, branch2: str) -> List[FileDifference]:
        """Compare two branches and return list of differences."""
        differences = []
        
        # Ensure both branches exist
        if not self._ensure_branch_exists(branch1):
            raise ValueError(f"Branch '{branch1}' does not exist")
        if not self._ensure_branch_exists(branch2):
            raise ValueError(f"Branch '{branch2}' does not exist")
        
        # Get file lists from both branches
        files_branch1 = self._get_file_list(branch1)
        files_branch2 = self._get_file_list(branch2)
        
        # Find added files (in branch2 but not in branch1)
        added_files = files_branch2 - files_branch1
        for file_path in added_files:
            diff_content = self._get_file_content(file_path, branch2)
            differences.append(FileDifference(
                file_path=file_path,
                change_type="added",
                diff_content=diff_content,
                new_size=self._get_file_size(file_path, branch2)
            ))
        
        # Find deleted files (in branch1 but not in branch2)
        deleted_files = files_branch1 - files_branch2
        for file_path in deleted_files:
            diff_content = self._get_file_content(file_path, branch1)
            differences.append(FileDifference(
                file_path=file_path,
                change_type="deleted",
                diff_content=diff_content,
                old_size=self._get_file_size(file_path, branch1)
            ))
        
        # Find modified files (exist in both branches but with different content)
        common_files = files_branch1 & files_branch2
        for file_path in common_files:
            try:
                # Compare file hashes to check if they're different
                hash1 = self._run_git_command(["git", "rev-parse", f"{branch1}:{file_path}"])
                hash2 = self._run_git_command(["git", "rev-parse", f"{branch2}:{file_path}"])
                
                if hash1 != hash2:
                    diff_content = self._get_file_diff(file_path, branch1, branch2)
                    differences.append(FileDifference(
                        file_path=file_path,
                        change_type="modified",
                        diff_content=diff_content,
                        old_size=self._get_file_size(file_path, branch1),
                        new_size=self._get_file_size(file_path, branch2)
                    ))
            except RuntimeError:
                # Skip files that can't be compared
                continue
        
        return differences
    
    def generate_tasks(self, differences: List[FileDifference], branch1: str, branch2: str) -> List[Task]:
        """Generate tasks for each difference found."""
        tasks = []
        
        for i, diff in enumerate(differences):
            task_id = f"DIFF-{i+1:04d}"
            
            if diff.change_type == "added":
                title = f"New file added: {diff.file_path}"
                description = (
                    f"File '{diff.file_path}' was added in branch '{branch2}' "
                    f"but does not exist in branch '{branch1}'.\n"
                    f"File size: {diff.new_size} bytes\n"
                    f"Action required: Review the new file and decide if it should be integrated."
                )
                priority = "high"
                
            elif diff.change_type == "deleted":
                title = f"File deleted: {diff.file_path}"
                description = (
                    f"File '{diff.file_path}' exists in branch '{branch1}' "
                    f"but was deleted in branch '{branch2}'.\n"
                    f"Original file size: {diff.old_size} bytes\n"
                    f"Action required: Review if the file deletion should be applied."
                )
                priority = "high"
                
            elif diff.change_type == "modified":
                title = f"File modified: {diff.file_path}"
                description = (
                    f"File '{diff.file_path}' has different content between "
                    f"branch '{branch1}' and branch '{branch2}'.\n"
                    f"Size change: {diff.old_size} -> {diff.new_size} bytes\n"
                    f"Action required: Review the changes and merge appropriately."
                )
                priority = "medium"
            
            task = Task(
                task_id=task_id,
                title=title,
                description=description,
                file_path=diff.file_path,
                change_type=diff.change_type,
                priority=priority,
                diff_content=diff.diff_content
            )
            
            tasks.append(task)
        
        return tasks
    
    def save_tasks_to_file(self, tasks: List[Task], output_file: str) -> None:
        """Save tasks to a JSON file."""
        tasks_data = []
        for task in tasks:
            task_data = {
                "task_id": task.task_id,
                "title": task.title,
                "description": task.description,
                "file_path": task.file_path,
                "change_type": task.change_type,
                "priority": task.priority,
                "diff_content": task.diff_content
            }
            tasks_data.append(task_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tasks_data, f, indent=2, ensure_ascii=False)
    
    def print_summary(self, tasks: List[Task]) -> None:
        """Print a summary of the tasks generated."""
        print(f"\n=== Branch Comparison Summary ===")
        print(f"Total tasks created: {len(tasks)}")
        
        by_type = {}
        by_priority = {}
        
        for task in tasks:
            by_type[task.change_type] = by_type.get(task.change_type, 0) + 1
            by_priority[task.priority] = by_priority.get(task.priority, 0) + 1
        
        print(f"\nTasks by change type:")
        for change_type, count in by_type.items():
            print(f"  {change_type}: {count}")
        
        print(f"\nTasks by priority:")
        for priority, count in by_priority.items():
            print(f"  {priority}: {count}")
        
        print(f"\n=== Task Details ===")
        for task in tasks:
            print(f"\n[{task.task_id}] {task.title}")
            print(f"  Priority: {task.priority}")
            print(f"  Type: {task.change_type}")
            print(f"  File: {task.file_path}")


def main():
    """Main entry point for the branch comparator tool."""
    parser = argparse.ArgumentParser(
        description="Compare two git branches and generate tasks for differences"
    )
    parser.add_argument("branch1", help="First branch name (e.g., 'julian')")
    parser.add_argument("branch2", help="Second branch name (e.g., 'import-repoB')")
    parser.add_argument(
        "--repo-path", 
        default=".", 
        help="Path to the git repository (default: current directory)"
    )
    parser.add_argument(
        "--output", 
        default="branch_comparison_tasks.json",
        help="Output file for tasks (default: branch_comparison_tasks.json)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        comparator = BranchComparator(args.repo_path)
        
        print(f"Comparing branches '{args.branch1}' and '{args.branch2}'...")
        differences = comparator.compare_branches(args.branch1, args.branch2)
        
        print(f"Found {len(differences)} differences")
        
        tasks = comparator.generate_tasks(differences, args.branch1, args.branch2)
        
        # Save tasks to file
        comparator.save_tasks_to_file(tasks, args.output)
        print(f"Tasks saved to: {args.output}")
        
        # Print summary
        comparator.print_summary(tasks)
        
        if args.verbose:
            print(f"\n=== Detailed Diff Content ===")
            for task in tasks:
                print(f"\n--- {task.task_id}: {task.file_path} ---")
                if task.diff_content:
                    print(task.diff_content[:500] + "..." if len(task.diff_content) > 500 else task.diff_content)
                else:
                    print("No diff content available")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()